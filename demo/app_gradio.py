"""
Demo Traductor LSP — MediaPipe Holistic + RF (1086 señas LSP → texto español)
Detecta manos/cuerpo en tiempo real y traduce señas individuales a español.
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

# ── Configuración ─────────────────────────────────────────────────────────────

RF_CKPT         = "checkpoints/rf_signs.pkl"   # modelo principal (1086 señas LSP)
SIGN_LABEL_PATH = "data/sign_label2idx.json"

N_FRAMES  = 30
IMG_SIZE  = (112, 112)
N_KP      = 75   # layout: [0:21]=left_hand, [21:42]=right_hand, [42:75]=pose
DEVICE    = "cpu"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32)

# Buffers globales
pixel_buffer = deque(maxlen=N_FRAMES)
kp_buffer    = deque(maxlen=N_FRAMES)   # cada elemento: (75, 3)

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

rf_model   = None   # Pipeline sklearn: StandardScaler + RandomForest
idx2label  = {}     # int → nombre seña en español


def load_models():
    global rf_model, idx2label

    if not Path(RF_CKPT).exists():
        print(f"[ERROR] No se encontró {RF_CKPT}. Ejecuta: python scripts/train_sign_model.py")
        return

    try:
        with open(RF_CKPT, "rb") as f:
            data = pickle.load(f)
        rf_model  = data["pipeline"]
        idx2label = data["idx2label"]
        print(f"Modelo RF listo — {data['n_classes']} señas LSP | F1-train={data['f1_train']:.3f}")
    except Exception as e:
        print(f"Error cargando RF: {e}")


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

def kp_buffer_to_feature(kp_seq: np.ndarray) -> np.ndarray:
    """
    Convierte secuencia de keypoints [T, 75, 3] al vector de 108 features
    que usa el modelo RF (media temporal de pose x/y + right_hand x/y).
    Layout kp: [0:21]=left_hand, [21:42]=right_hand, [42:75]=pose
    """
    pose       = kp_seq[:, 42:75, :2]   # (T, 33, 2)
    right_hand = kp_seq[:, 21:42, :2]   # (T, 21, 2)

    # Media temporal
    pose_mean = pose.mean(axis=0)            # (33, 2)
    rh_mean   = right_hand.mean(axis=0)      # (21, 2)

    # Concatenar igual que extract_features(): pose_x, pose_y, rh_x, rh_y
    feat = np.concatenate([
        pose_mean[:, 0],    # pose x  (33)
        pose_mean[:, 1],    # pose y  (33)
        rh_mean[:, 0],      # right_hand x  (21)
        rh_mean[:, 1],      # right_hand y  (21)
    ]).astype(np.float32)   # (108,)
    return feat


def run_rf_inference(kp_seq: np.ndarray) -> dict:
    """RF sobre vector de 108 features → nombre de seña en español."""
    feat  = kp_buffer_to_feature(kp_seq).reshape(1, -1)
    proba = rf_model.predict_proba(feat)[0]
    idx   = int(proba.argmax())
    conf  = float(proba[idx])
    seña  = idx2label.get(idx, "?")

    top3 = [{"clase": idx2label.get(int(i), str(i)), "prob": float(proba[i])}
            for i in np.argsort(proba)[::-1][:3]]
    return {"clase": seña, "texto": seña, "confidence": conf,
            "top3": top3, "modelo": "RF-LSP"}


def run_inference() -> dict | None:
    """Ejecuta inferencia RF cuando el buffer tiene suficientes frames."""
    if rf_model is None or len(kp_buffer) < N_FRAMES:
        return None
    t0     = time.perf_counter()
    kp_seq = np.stack(list(kp_buffer))   # (30, 75, 3)
    result = run_rf_inference(kp_seq)
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

    if rf_model is None:
        return None, "Modelo no cargado. Ejecuta scripts/train_sign_model.py", "", ""

    t0     = time.perf_counter()
    result = run_rf_inference(np.stack(kp_frames))
    result["latency_ms"] = (time.perf_counter() - t0) * 1000

    conf      = result["confidence"] * 100
    result_md = (
        f"## {result['texto']}\n\n"
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
.traduccion { font-size: 1.5em !important; padding: 14px 18px;
              border-left: 4px solid #2196F3; }
.estado     { font-size: 1.1em; padding: 8px 12px; }
"""

with gr.Blocks(title="Traductor LSP") as demo:

    gr.Markdown("""
    # 🤟 Traductor LSP → Castellano
    **MediaPipe Holistic** detecta manos y cuerpo en tiempo real.
    **RandomForest** (1086 señas LSP) reconoce la seña y muestra el texto en castellano.
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
Extracción de features (108 dims)
     pose x/y (33×2=66) + mano derecha x/y (21×2=42)
     Media temporal sobre 30 frames
     ↓
RandomForest — 1086 señas LSP
     ↓
Nombre de la seña en español
```

## Partes del cuerpo detectadas

| Parte | Color | Puntos |
|-------|-------|--------|
| Mano izquierda | 🔴 Rojo | 21 keypoints (dedos + palma) |
| Mano derecha | 🟢 Verde | 21 keypoints (dedos + palma) |
| Cuerpo/Pose | 🔵 Azul | 33 keypoints (hombros, codos, muñecas, caderas...) |
| Rostro | ⚪ Gris | 468 keypoints (ojos, labios, cejas, nariz) |

## Modelo principal

| Modelo | Input | Clases | Uso |
|--------|-------|--------|-----|
| **RF-LSP** | 108 features (pose+mano) | **1086 señas LSP** | Reconocimiento seña→texto |

## Nota sobre precisión
El modelo fue entrenado con ~3.4 muestras/clase en promedio.
Con más datos de entrenamiento por seña, la precisión mejora significativamente.
            """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        theme=gr.themes.Soft(),
        css=CSS,
    )
