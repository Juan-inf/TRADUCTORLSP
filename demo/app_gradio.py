"""
Demo Traductor LSP — MediaPipe Holistic + STGCN + CNN-LSTM
Detecta manos, dedos, cuerpo y rostro en tiempo real y traduce señas LSP.
"""

import json, time, warnings
import numpy as np
import cv2
import torch
import torch.nn.functional as F
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

# ── Configuración ─────────────────────────────────────────────────────────────

LABEL2IDX_PATH  = "data/label2idx.json"
CLASE_TEXTO_PATH = "data/clase_texto.json"
ONNX_PATH       = "checkpoints/cnn_lstm_best.onnx"
STGCN_CKPT      = "checkpoints/ab_var2_lr5e4.pt"

N_FRAMES  = 30
IMG_SIZE  = (112, 112)
N_KP      = 75   # 42 manos + 33 pose
DEVICE    = "cpu"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32)

# Buffers globales
pixel_buffer = deque(maxlen=N_FRAMES)
kp_buffer    = deque(maxlen=N_FRAMES)

# ── MediaPipe ─────────────────────────────────────────────────────────────────

mp_holistic      = mp.solutions.holistic
mp_drawing       = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4,
)

# Colores para dibujar
COLOR_LHAND = (255,  80,  80)   # rojo — mano izquierda
COLOR_RHAND = ( 80, 200,  80)   # verde — mano derecha
COLOR_POSE  = ( 80, 150, 255)   # azul — cuerpo
COLOR_FACE  = (200, 200, 200)   # gris — rostro

# ── Modelos ───────────────────────────────────────────────────────────────────

onnx_session = None
stgcn_model  = None
idx2label    = {}
clase_texto  = {}


def load_models():
    global onnx_session, stgcn_model, idx2label, clase_texto

    if Path(CLASE_TEXTO_PATH).exists():
        with open(CLASE_TEXTO_PATH, encoding="utf-8") as f:
            clase_texto = json.load(f)

    if not Path(LABEL2IDX_PATH).exists():
        print("label2idx.json no encontrado")
        return

    with open(LABEL2IDX_PATH) as f:
        l2i = json.load(f)
    idx2label = {int(v): k for k, v in l2i.items()}

    # CNN-LSTM ONNX (píxeles)
    if Path(ONNX_PATH).exists():
        try:
            import onnxruntime as ort
            avail = ort.get_available_providers()
            provs = [p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider")
                     if p in avail]
            onnx_session = ort.InferenceSession(ONNX_PATH, providers=provs)
            print(f"CNN-LSTM ONNX listo — {len(idx2label)} clases")
        except Exception as e:
            print(f"ONNX error: {e}")

    # STGCN (landmarks)
    if Path(STGCN_CKPT).exists():
        try:
            from src.models import STGCN
            m = STGCN(n_classes=len(idx2label), n_nodes=N_KP,
                      in_channels=3, hidden_channels=64, num_layers=4)
            ck = torch.load(STGCN_CKPT, map_location=DEVICE)
            m.load_state_dict(ck["model_state"])
            m.eval()
            stgcn_model = m
            print(f"STGCN listo — val_acc={ck.get('val_acc', '?'):.3f}")
        except Exception as e:
            print(f"STGCN error: {e}")


load_models()

# ── Extracción y dibujo de landmarks ─────────────────────────────────────────

def extract_and_draw(frame_rgb: np.ndarray):
    """
    Corre MediaPipe sobre frame_rgb.
    Devuelve:  vis (BGR con landmarks dibujados), kp [75,3], estado detección.
    """
    results = holistic.process(frame_rgb)
    vis = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    detected = {
        "mano_izq": results.left_hand_landmarks  is not None,
        "mano_der": results.right_hand_landmarks is not None,
        "cuerpo":   results.pose_landmarks       is not None,
        "rostro":   results.face_landmarks       is not None,
    }

    # ── Rostro (malla de contorno) ──
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            vis, results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=COLOR_FACE, thickness=1, circle_radius=1),
        )

    # ── Pose (esqueleto corporal) ──
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            vis, results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=COLOR_POSE, thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=COLOR_POSE, thickness=2),
        )

    # ── Mano izquierda ──
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            vis, results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=COLOR_LHAND, thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=COLOR_LHAND, thickness=2),
        )

    # ── Mano derecha ──
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            vis, results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=COLOR_RHAND, thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=COLOR_RHAND, thickness=2),
        )

    # ── Keypoints 75 para STGCN ──
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
    # Normalización posicional: relativo a muñeca derecha
    wrist = kp[21:22, :]
    if wrist.any():
        kp = kp - wrist

    return vis, kp, detected


def draw_status_bar(vis: np.ndarray, detected: dict, result: dict | None) -> np.ndarray:
    """Dibuja panel inferior con estado de detección y predicción."""
    h, w = vis.shape[:2]
    bar_h = 110
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)

    # Estado de cada parte del cuerpo
    parts = [
        ("Mano izq.", detected["mano_izq"], COLOR_LHAND),
        ("Mano der.", detected["mano_der"], COLOR_RHAND),
        ("Cuerpo",    detected["cuerpo"],   COLOR_POSE),
        ("Rostro",    detected["rostro"],   COLOR_FACE),
    ]
    x_off = 10
    for label, is_det, color in parts:
        sym   = "✓" if is_det else "✗"
        col   = color if is_det else (80, 80, 80)
        txt   = f"{sym} {label}"
        cv2.putText(bar, txt, (x_off, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        x_off += 155

    # Predicción
    if result:
        texto = result.get("texto", "")
        conf  = result.get("confidence", 0.0) * 100
        model_tag = result.get("modelo", "")
        col   = (0, 220, 50) if conf >= 60 else (50, 120, 255) if conf >= 40 else (80, 80, 80)

        # Texto en 2 líneas si es largo
        words = texto.split()
        line1 = " ".join(words[:4])
        line2 = " ".join(words[4:]) if len(words) > 4 else ""
        cv2.putText(bar, line1, (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)
        if line2:
            cv2.putText(bar, line2, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)

        conf_text = f"{conf:.0f}%  [{model_tag}]  {result.get('latency_ms', 0):.0f}ms"
        cv2.putText(bar, conf_text, (w - 260, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        # Barra de confianza en la parte baja
        cv2.rectangle(bar, (0, bar_h - 8), (int(conf / 100 * w), bar_h), col, -1)
    else:
        n_buf = len(kp_buffer) if kp_buffer else 0
        pct   = int(n_buf / N_FRAMES * w)
        cv2.putText(bar, f"Acumulando frames... {n_buf}/{N_FRAMES}",
                    (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1)
        cv2.rectangle(bar, (0, bar_h - 8), (pct, bar_h), (0, 180, 255), -1)

    return np.vstack([vis, bar])


# ── Inferencia ────────────────────────────────────────────────────────────────

def clase_a_texto(clase_id: str) -> str:
    if clase_id in clase_texto:
        return clase_texto[clase_id]
    if clase_id.startswith("vineta_"):
        return f"Historia viñeta {clase_id.replace('vineta_', '')}"
    return clase_id


@torch.no_grad()
def run_stgcn(kp_seq: np.ndarray) -> dict:
    """STGCN sobre secuencia de landmarks [T, N, 3]."""
    x = torch.from_numpy(kp_seq).unsqueeze(0).float()   # [1, T, N, 3]
    logits = stgcn_model(x)
    probs  = F.softmax(logits, dim=-1).numpy()[0]
    idx    = int(probs.argmax())
    clase  = idx2label.get(idx, "desconocida")
    top3   = [{"clase": idx2label.get(int(i), str(i)),
               "texto": clase_a_texto(idx2label.get(int(i), str(i))),
               "prob":  float(probs[i])}
              for i in np.argsort(probs)[::-1][:3]]
    return {"clase": clase, "texto": clase_a_texto(clase),
            "confidence": float(probs[idx]), "top3": top3, "modelo": "STGCN"}


def run_cnnlstm(pixel_seq: np.ndarray) -> dict:
    """CNN-LSTM ONNX sobre secuencia de píxeles [T, H, W, C]."""
    arr    = np.transpose(pixel_seq, (3, 0, 1, 2))[np.newaxis].astype(np.float32)
    names  = [i.name for i in onnx_session.get_inputs()]
    iname  = next((n for n in names if n in ("pixels", "input", "x")), names[0])
    logits = onnx_session.run(None, {iname: arr})[0]
    probs  = np.exp(logits[0] - logits[0].max())
    probs /= probs.sum()
    idx    = int(probs.argmax())
    clase  = idx2label.get(idx, "desconocida")
    top3   = [{"clase": idx2label.get(int(i), str(i)),
               "texto": clase_a_texto(idx2label.get(int(i), str(i))),
               "prob":  float(probs[i])}
              for i in np.argsort(probs)[::-1][:3]]
    return {"clase": clase, "texto": clase_a_texto(clase),
            "confidence": float(probs[idx]), "top3": top3, "modelo": "CNN-LSTM"}


def run_inference() -> dict | None:
    """Elige el mejor modelo disponible y ejecuta inferencia."""
    if len(kp_buffer) < N_FRAMES and len(pixel_buffer) < N_FRAMES:
        return None

    t0 = time.perf_counter()
    result = None

    # STGCN si hay landmarks suficientes (prioritario: usa señas reales)
    if stgcn_model is not None and len(kp_buffer) >= N_FRAMES:
        kp_seq = np.stack(list(kp_buffer))   # [T, N, 3]
        result = run_stgcn(kp_seq)

    # CNN-LSTM como fallback / segundo modelo
    if onnx_session is not None and len(pixel_buffer) >= N_FRAMES:
        pix_seq = np.stack(list(pixel_buffer))  # [T, H, W, C]
        r_cnn   = run_cnnlstm(pix_seq)
        # Usar CNN-LSTM si no hay resultado STGCN o si tiene mayor confianza
        if result is None or r_cnn["confidence"] > result["confidence"] + 0.15:
            result = r_cnn

    if result:
        result["latency_ms"] = (time.perf_counter() - t0) * 1000

    return result


# ── Handlers Gradio ───────────────────────────────────────────────────────────

last_result = None


def process_webcam_frame(frame):
    global last_result

    if frame is None:
        return None, "Sin señal de cámara", "", ""

    # Garantizar RGB 3 canales (Gradio puede enviar RGBA)
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    elif frame.shape[2] == 4:
        frame = frame[:, :, :3]

    # ── Extracción de landmarks y dibujo ──
    vis, kp, detected = extract_and_draw(frame)

    # ── Acumular buffers ──
    kp_buffer.append(kp)

    # Píxeles normalizados para CNN-LSTM
    px = cv2.resize(frame, IMG_SIZE, interpolation=cv2.INTER_LINEAR).astype(np.float32)
    px = (px / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    pixel_buffer.append(px)

    # ── Inferencia (cada N_FRAMES/2 frames) ──
    if len(kp_buffer) % (N_FRAMES // 2) == 0 and len(kp_buffer) >= N_FRAMES:
        r = run_inference()
        if r is not None:
            last_result = r

    # ── Componer frame final ──
    vis_out = draw_status_bar(vis, detected, last_result)

    # ── Textos para el panel derecho ──
    if last_result:
        r    = last_result
        conf = r["confidence"] * 100
        result_md = (
            f"## {r['texto']}\n\n"
            f"**Confianza:** {conf:.1f}%  |  "
            f"**Modelo:** {r.get('modelo', '?')}  |  "
            f"**Latencia:** {r.get('latency_ms', 0):.0f} ms"
        )
        top3_md = "**Top 3 predicciones:**\n" + "\n".join(
            f"{i+1}. {t['texto']} — {t['prob']*100:.1f}%"
            for i, t in enumerate(r.get("top3", []))
        )
    else:
        n_buf = len(kp_buffer)
        result_md = f"**Acumulando señas... {n_buf}/{N_FRAMES} frames**"
        top3_md   = ""

    # Estado de partes del cuerpo
    estado_md = (
        f"{'🟢' if detected['mano_izq'] else '🔴'} Mano izq. &nbsp;&nbsp;"
        f"{'🟢' if detected['mano_der'] else '🔴'} Mano der. &nbsp;&nbsp;"
        f"{'🟢' if detected['cuerpo']   else '🔴'} Cuerpo &nbsp;&nbsp;"
        f"{'🟢' if detected['rostro']   else '🔴'} Rostro"
    )

    return vis_out, result_md, top3_md, estado_md


def process_video_file(video_path):
    if video_path is None:
        return None, "No se subió ningún video.", "", ""

    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx_s = np.linspace(0, total - 1, N_FRAMES, dtype=int)

    kp_frames, px_frames, detected_any = [], [], {k: False for k in ["mano_izq","mano_der","cuerpo","rostro"]}
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
        px = cv2.resize(frame_rgb, IMG_SIZE).astype(np.float32)
        px_frames.append((px / 255.0 - IMAGENET_MEAN) / IMAGENET_STD)
    cap.release()

    if len(kp_frames) < 5:
        return None, "Video demasiado corto.", "", ""

    # Pad a N_FRAMES si hace falta
    while len(kp_frames) < N_FRAMES:
        kp_frames.append(kp_frames[-1])
        px_frames.append(px_frames[-1])

    t0  = time.perf_counter()
    result = None
    if stgcn_model is not None:
        result = run_stgcn(np.stack(kp_frames))
    if onnx_session is not None:
        r_cnn = run_cnnlstm(np.stack(px_frames))
        if result is None or r_cnn["confidence"] > result["confidence"] + 0.15:
            result = r_cnn
    if result:
        result["latency_ms"] = (time.perf_counter() - t0) * 1000

    if not result:
        return None, "No se pudo predecir.", "", ""

    conf      = result["confidence"] * 100
    result_md = (
        f"## {result['texto']}\n\n"
        f"**Confianza:** {conf:.1f}%  |  **Modelo:** {result['modelo']}  |  "
        f"**Latencia:** {result['latency_ms']:.0f} ms"
    )
    top3_md = "**Top 3:**\n" + "\n".join(
        f"{i+1}. {t['texto']} — {t['prob']*100:.1f}%"
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
.traduccion { font-size: 1.5em !important; padding: 14px 18px;
              border-left: 4px solid #2196F3; }
.estado     { font-size: 1.1em; padding: 8px 12px; }
"""

with gr.Blocks(title="Traductor LSP") as demo:

    gr.Markdown("""
    # 🤟 Traductor LSP → Castellano
    **MediaPipe Holistic** detecta manos, dedos, cuerpo y rostro en tiempo real.
    **STGCN + CNN-LSTM** reconoce la seña y muestra el texto en castellano.
    """)

    with gr.Tabs():

        # ── Tab Cámara ────────────────────────────────────────────────
        with gr.TabItem("📷 Cámara en vivo"):
            with gr.Row():
                with gr.Column(scale=3):
                    webcam_in  = gr.Image(sources=["webcam"], streaming=True,
                                          label="Cámara", height=360)
                    webcam_out = gr.Image(label="Landmarks detectados", height=420)
                with gr.Column(scale=2):
                    estado_cam  = gr.Markdown("", elem_classes=["estado"])
                    result_cam  = gr.Markdown("**Esperando señas...**",
                                              elem_classes=["traduccion"])
                    top3_cam    = gr.Markdown("")

            webcam_in.stream(
                fn=process_webcam_frame,
                inputs=[webcam_in],
                outputs=[webcam_out, result_cam, top3_cam, estado_cam],
                time_limit=120,
                stream_every=0.12,
            )

        # ── Tab Video ─────────────────────────────────────────────────
        with gr.TabItem("🎬 Subir video"):
            with gr.Row():
                with gr.Column():
                    video_in  = gr.Video(label="Video MP4 con señas LSP")
                    btn       = gr.Button("▶ Analizar señas", variant="primary", size="lg")
                with gr.Column():
                    estado_vid = gr.Markdown("", elem_classes=["estado"])
                    result_vid = gr.Markdown("", elem_classes=["traduccion"])
                    top3_vid   = gr.Markdown("")

            btn.click(
                fn=process_video_file,
                inputs=[video_in],
                outputs=[video_in, result_vid, top3_vid, estado_vid],
            )

        # ── Tab Info ──────────────────────────────────────────────────
        with gr.TabItem("ℹ️ Cómo funciona"):
            gr.Markdown("""
## Pipeline de detección y traducción

```
Cámara / Video
     ↓
MediaPipe Holistic
     ├── 21 keypoints mano izquierda  (rojo)
     ├── 21 keypoints mano derecha    (verde)
     ├── 33 keypoints pose/cuerpo     (azul)
     └── 468 keypoints rostro         (gris)
     ↓
Modelo 1 — ST-GCN        → sobre 75 keypoints [T=30, N=75, C=3]
Modelo 2 — CNN-LSTM ONNX → sobre píxeles      [T=30, 112, 112, 3]
     ↓
Se usa el modelo con mayor confianza
     ↓
Texto en castellano
```

## Partes del cuerpo detectadas

| Parte | Color | Puntos |
|-------|-------|--------|
| Mano izquierda | 🔴 Rojo | 21 keypoints (dedos + palma) |
| Mano derecha | 🟢 Verde | 21 keypoints (dedos + palma) |
| Cuerpo/Pose | 🔵 Azul | 33 keypoints (hombros, codos, muñecas, caderas...) |
| Rostro | ⚪ Gris | 468 keypoints (ojos, labios, cejas, nariz) |

## Modelos

| Modelo | Input | Acc test | Uso |
|--------|-------|---------|-----|
| **ST-GCN** | Landmarks [T,N,C] | 52.5% val | Principal (usa señas reales) |
| **CNN-LSTM** | Píxeles [T,H,W,C] | 83.0% | Fallback / confirmación |
            """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        theme=gr.themes.Soft(),
        css=CSS,
    )
