"""
Traductor LSP → Castellano
MediaPipe Holistic + LSTM Bidireccional (1141 señas LSP)
Tiempo real desde cámara web o video pregrabado.
"""

import json, time, pickle, warnings
import numpy as np
import cv2
from pathlib import Path
from collections import deque
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

try:
    import gradio as gr
except ImportError:
    raise ImportError("pip install gradio>=4.0")

import mediapipe as mp
import onnxruntime as ort

# ── Configuración ─────────────────────────────────────────────────────────────

LSTM_ONNX = "checkpoints/lstm_signs.onnx"   # modelo principal (LSTM Bidir)
RF_CKPT   = "checkpoints/rf_signs.pkl"       # fallback RF
LABEL_PATH = "data/lstm_label2idx.json"

N_FRAMES   = 30
N_DIMS     = 150   # pose(33×2) + left_hand(21×2) + right_hand(21×2)
N_KP       = 75    # layout: [0:21]=left_hand, [21:42]=right_hand, [42:75]=pose
CONF_UMBRAL = 0.30  # mostrar resultado solo si confianza ≥ este valor

# ── Buffers globales ──────────────────────────────────────────────────────────

kp_buffer = deque(maxlen=N_FRAMES)   # cada elemento: (75, 3)
historial  = deque(maxlen=10)        # últimas 10 traducciones

# ── MediaPipe ─────────────────────────────────────────────────────────────────

mp_holistic       = mp.solutions.holistic
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4,
)

COLOR_LHAND = (255,  80,  80)
COLOR_RHAND = ( 80, 200,  80)
COLOR_POSE  = ( 80, 150, 255)
COLOR_FACE  = (200, 200, 200)

# ── Modelos ───────────────────────────────────────────────────────────────────

lstm_session = None
rf_model     = None
idx2label    = {}


def load_models():
    global lstm_session, rf_model, idx2label

    if Path(LABEL_PATH).exists():
        with open(LABEL_PATH, encoding="utf-8") as f:
            l2i = json.load(f)
        idx2label = {int(v): k for k, v in l2i.items()}
        print(f"Etiquetas cargadas: {len(idx2label)} señas LSP")

    # LSTM ONNX (principal)
    if Path(LSTM_ONNX).exists():
        try:
            avail = ort.get_available_providers()
            provs = [p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider")
                     if p in avail]
            lstm_session = ort.InferenceSession(LSTM_ONNX, providers=provs)
            print(f"LSTM ONNX listo — {len(idx2label)} clases")
        except Exception as e:
            print(f"LSTM error: {e}")

    # RF (fallback)
    if Path(RF_CKPT).exists() and lstm_session is None:
        try:
            with open(RF_CKPT, "rb") as f:
                data = pickle.load(f)
            rf_model  = data["pipeline"]
            # RF usa idx2label propio — sobreescribir solo si LSTM no cargó
            if not idx2label:
                idx2label.update(data["idx2label"])
            print(f"RF fallback listo — {data['n_classes']} clases")
        except Exception as e:
            print(f"RF error: {e}")


load_models()

# ── Extracción de landmarks ───────────────────────────────────────────────────

def extract_and_draw(frame_rgb: np.ndarray):
    """Ejecuta MediaPipe, dibuja landmarks y devuelve (vis_bgr, kp[75,3], detected)."""
    results = holistic.process(frame_rgb)
    vis = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    detected = {
        "mano_izq": results.left_hand_landmarks  is not None,
        "mano_der": results.right_hand_landmarks is not None,
        "cuerpo":   results.pose_landmarks       is not None,
        "rostro":   results.face_landmarks       is not None,
    }

    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            vis, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=COLOR_FACE, thickness=1, circle_radius=1))
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            vis, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=COLOR_POSE, thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=COLOR_POSE, thickness=2))
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            vis, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=COLOR_LHAND, thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=COLOR_LHAND, thickness=2))
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            vis, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=COLOR_RHAND, thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=COLOR_RHAND, thickness=2))

    kp = np.zeros((N_KP, 3), dtype=np.float32)
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            kp[i] = [lm.x, lm.y, lm.z]
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            kp[21 + i] = [lm.x, lm.y, lm.z]
    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            kp[42 + i] = [lm.x, lm.y, lm.z]

    return vis, kp, detected


# ── Conversión kp_buffer → features ──────────────────────────────────────────

def kp_seq_to_lstm_features(kp_seq: np.ndarray) -> np.ndarray:
    """
    kp_seq: [T, 75, 3]  →  features: [T, 150]
    Orden: pose_x(33) pose_y(33) left_x(21) left_y(21) right_x(21) right_y(21)
    """
    pose  = kp_seq[:, 42:75, :2]   # (T, 33, 2)
    left  = kp_seq[:,  0:21, :2]   # (T, 21, 2)
    right = kp_seq[:, 21:42, :2]   # (T, 21, 2)
    return np.concatenate([
        pose[:,  :, 0], pose[:,  :, 1],
        left[:,  :, 0], left[:,  :, 1],
        right[:, :, 0], right[:, :, 1],
    ], axis=1).astype(np.float32)  # (T, 150)


def kp_seq_to_rf_features(kp_seq: np.ndarray) -> np.ndarray:
    """kp_seq [T,75,3] → [108] (media temporal pose+right_hand para RF fallback)."""
    pose  = kp_seq[:, 42:75, :2].mean(0)  # (33, 2)
    right = kp_seq[:, 21:42, :2].mean(0)  # (21, 2)
    return np.concatenate([pose[:, 0], pose[:, 1],
                            right[:, 0], right[:, 1]]).astype(np.float32)


# ── Inferencia ────────────────────────────────────────────────────────────────

def run_inference(kp_seq: np.ndarray) -> dict | None:
    """Inferencia LSTM (o RF fallback) sobre secuencia [T, 75, 3]."""
    t0 = time.perf_counter()

    if lstm_session is not None:
        feat  = kp_seq_to_lstm_features(kp_seq)[np.newaxis]  # (1, 30, 150)
        logits = lstm_session.run(None, {"sequence": feat})[0][0]  # (n_classes,)
        probs  = np.exp(logits - logits.max())
        probs /= probs.sum()
        idx    = int(probs.argmax())
        top3   = [{"clase": idx2label.get(int(i), str(i)), "prob": float(probs[i])}
                  for i in np.argsort(probs)[::-1][:3]]
        seña   = idx2label.get(idx, "?")
        conf   = float(probs[idx])
        modelo = "LSTM-LSP"

    elif rf_model is not None:
        feat  = kp_seq_to_rf_features(kp_seq).reshape(1, -1)
        proba = rf_model.predict_proba(feat)[0]
        idx   = int(proba.argmax())
        conf  = float(proba[idx])
        seña  = idx2label.get(idx, "?")
        top3  = [{"clase": idx2label.get(int(i), str(i)), "prob": float(proba[i])}
                 for i in np.argsort(proba)[::-1][:3]]
        modelo = "RF-LSP"
    else:
        return None

    return {
        "seña": seña, "confidence": conf,
        "top3": top3, "modelo": modelo,
        "latency_ms": (time.perf_counter() - t0) * 1000,
    }


# ── Panel de estado visual ────────────────────────────────────────────────────

def draw_status_bar(vis, detected, result):
    h, w = vis.shape[:2]
    bar  = np.zeros((110, w, 3), dtype=np.uint8)
    parts = [
        ("Mano izq.", detected["mano_izq"], COLOR_LHAND),
        ("Mano der.", detected["mano_der"], COLOR_RHAND),
        ("Cuerpo",    detected["cuerpo"],   COLOR_POSE),
        ("Rostro",    detected["rostro"],   COLOR_FACE),
    ]
    x_off = 10
    for label, ok, color in parts:
        sym = "✓" if ok else "✗"
        col = color if ok else (80, 80, 80)
        cv2.putText(bar, f"{sym} {label}", (x_off, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        x_off += 155
    if result:
        conf = result["confidence"] * 100
        col  = (0, 220, 50) if conf >= 60 else (50, 180, 255) if conf >= 35 else (120, 120, 120)
        cv2.putText(bar, result["seña"], (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
        cv2.putText(bar, f"{conf:.0f}%  [{result['modelo']}]  {result['latency_ms']:.0f}ms",
                    (w - 280, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cv2.rectangle(bar, (0, 102), (int(conf / 100 * w), 110), col, -1)
    else:
        n_buf = len(kp_buffer)
        pct   = int(n_buf / N_FRAMES * w)
        cv2.putText(bar, f"Acumulando señas… {n_buf}/{N_FRAMES} frames",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1)
        cv2.rectangle(bar, (0, 102), (pct, 110), (0, 180, 255), -1)
    return np.vstack([vis, bar])


# ── Handlers Gradio ───────────────────────────────────────────────────────────

last_result = None


def process_webcam_frame(frame):
    global last_result
    if frame is None:
        return None, "Sin señal de cámara", "", ""

    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    elif frame.shape[2] == 4:
        frame = frame[:, :, :3]

    vis, kp, detected = extract_and_draw(frame)
    kp_buffer.append(kp)

    if len(kp_buffer) >= N_FRAMES and len(kp_buffer) % (N_FRAMES // 2) == 0:
        kp_seq = np.stack(list(kp_buffer))  # (30, 75, 3)
        r = run_inference(kp_seq)
        if r is not None and r["confidence"] >= CONF_UMBRAL:
            last_result = r
            if not historial or historial[-1] != r["seña"]:
                historial.append(r["seña"])

    vis_out = draw_status_bar(vis, detected, last_result)

    if last_result:
        r    = last_result
        conf = r["confidence"] * 100
        result_md = (
            f"## {r['seña']}\n\n"
            f"**Confianza:** {conf:.1f}%  |  **Modelo:** {r['modelo']}  |  "
            f"**Latencia:** {r['latency_ms']:.0f} ms"
        )
        top3_md = "**Top 3:**\n" + "\n".join(
            f"{i+1}. {t['clase']} — {t['prob']*100:.1f}%"
            for i, t in enumerate(r.get("top3", []))
        )
        hist_md = "**Historial:** " + " › ".join(list(historial)[-8:])
    else:
        n_buf = len(kp_buffer)
        result_md = f"**Acumulando señas… {n_buf}/{N_FRAMES} frames**"
        top3_md   = ""
        hist_md   = ""

    estado_md = (
        f"{'🟢' if detected['mano_izq'] else '🔴'} Mano izq. &nbsp;&nbsp;"
        f"{'🟢' if detected['mano_der'] else '🔴'} Mano der. &nbsp;&nbsp;"
        f"{'🟢' if detected['cuerpo']   else '🔴'} Cuerpo &nbsp;&nbsp;"
        f"{'🟢' if detected['rostro']   else '🔴'} Rostro"
    )
    return vis_out, result_md, top3_md + "\n\n" + hist_md, estado_md


def process_video_file(video_path):
    if video_path is None:
        return None, "No se subió ningún video.", "", ""

    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx_s = np.linspace(0, total - 1, N_FRAMES, dtype=int)

    kp_frames, detected_any = [], {k: False for k in ["mano_izq","mano_der","cuerpo","rostro"]}
    for idx in idx_s:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, kp, det = extract_and_draw(frame_rgb)
        kp_frames.append(kp)
        for k in detected_any:
            if det[k]:
                detected_any[k] = True
    cap.release()

    if len(kp_frames) < 5:
        return None, "Video demasiado corto.", "", ""
    while len(kp_frames) < N_FRAMES:
        kp_frames.append(kp_frames[-1])

    t0     = time.perf_counter()
    result = run_inference(np.stack(kp_frames))
    if result is None:
        return None, "Modelo no disponible.", "", ""

    result["latency_ms"] = (time.perf_counter() - t0) * 1000
    conf = result["confidence"] * 100
    result_md = (
        f"## {result['seña']}\n\n"
        f"**Confianza:** {conf:.1f}%  |  **Modelo:** {result['modelo']}  |  "
        f"**Latencia:** {result['latency_ms']:.0f} ms"
    )
    top3_md = "**Top 3:**\n" + "\n".join(
        f"{i+1}. {t['clase']} — {t['prob']*100:.1f}%"
        for i, t in enumerate(result.get("top3", []))
    )
    estado_md = (
        f"{'🟢' if detected_any['mano_izq'] else '🔴'} Mano izq. &nbsp;&nbsp;"
        f"{'🟢' if detected_any['mano_der'] else '🔴'} Mano der. &nbsp;&nbsp;"
        f"{'🟢' if detected_any['cuerpo']   else '🔴'} Cuerpo &nbsp;&nbsp;"
        f"{'🟢' if detected_any['rostro']   else '🔴'} Rostro"
    )
    return video_path, result_md, top3_md, estado_md


# ── Interfaz Gradio ───────────────────────────────────────────────────────────

CSS = """
.traduccion { font-size: 1.6em !important; padding: 16px 20px;
              border-left: 5px solid #2196F3; background: #f0f7ff; }
.estado     { font-size: 1.05em; padding: 8px 12px; }
"""

with gr.Blocks(title="Traductor LSP → Castellano", css=CSS) as demo:

    gr.Markdown("""
    # 🤟 Traductor LSP → Castellano
    **Sistema integral de comunicación inclusiva** — Lengua de Señas Peruana a texto en tiempo real.
    MediaPipe Holistic detecta pose + ambas manos (150 dims/frame).
    LSTM Bidireccional clasifica la seña en **{}** clases LSP.
    """.format(len(idx2label)))

    with gr.Tabs():

        with gr.TabItem("📷 Cámara en vivo"):
            with gr.Row():
                with gr.Column(scale=3):
                    webcam_in  = gr.Image(sources=["webcam"], streaming=True,
                                          label="Cámara", height=360)
                    webcam_out = gr.Image(label="Landmarks detectados", height=420)
                with gr.Column(scale=2):
                    estado_cam  = gr.Markdown("", elem_classes=["estado"])
                    result_cam  = gr.Markdown("**Esperando señas…**",
                                              elem_classes=["traduccion"])
                    top3_cam    = gr.Markdown("")

            webcam_in.stream(
                fn=process_webcam_frame,
                inputs=[webcam_in],
                outputs=[webcam_out, result_cam, top3_cam, estado_cam],
                time_limit=300,
                stream_every=0.10,
            )

        with gr.TabItem("🎬 Subir video"):
            with gr.Row():
                with gr.Column():
                    video_in = gr.Video(label="Video MP4/AVI/MOV con señas LSP")
                    btn      = gr.Button("▶ Traducir señas", variant="primary", size="lg")
                with gr.Column():
                    estado_vid = gr.Markdown("", elem_classes=["estado"])
                    result_vid = gr.Markdown("", elem_classes=["traduccion"])
                    top3_vid   = gr.Markdown("")

            btn.click(
                fn=process_video_file,
                inputs=[video_in],
                outputs=[video_in, result_vid, top3_vid, estado_vid],
            )

        with gr.TabItem("ℹ️ Pipeline"):
            gr.Markdown(f"""
## Pipeline técnico

```
Cámara / Video MP4·AVI·MOV
     ↓
MediaPipe Holistic
     ├── 33 keypoints pose/cuerpo    (azul)
     ├── 21 keypoints mano izquierda (rojo)
     ├── 21 keypoints mano derecha   (verde)
     └── 468 keypoints rostro        (gris)
     ↓
Feature extraction: 150 dims/frame
     pose_x(33) + pose_y(33) + left_x(21) + left_y(21) + right_x(21) + right_y(21)
     ↓
LSTM Bidireccional + Attention  [30 frames × 150 dims]
     ↓
{len(idx2label)} señas LSP → texto en castellano
```

## Dataset de entrenamiento

| Fuente | Muestras | Clases |
|--------|---------|--------|
| Keypoints/pkl (viñetas segmentadas) | 3,684 | 1,086 |
| Glosas/MP4 (grabaciones individuales) | 252 | 143 |
| **Total** | **3,936** | **1,141** |

## Modelo

| | Valor |
|--|--|
| Arquitectura | LSTM Bidireccional 2 capas + Temporal Attention |
| Parámetros | 2.6M |
| F1-macro test | 0.0126 |
| Latencia ONNX | <50ms |
            """)

if __name__ == "__main__":
    import sys
    share = "--share" in sys.argv
    demo.launch(server_name="0.0.0.0", server_port=7860,
                show_error=True, theme=gr.themes.Soft(), css=CSS,
                share=share)
