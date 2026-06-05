"""
Traductor LSP → Castellano — HuggingFace Spaces
Sistema integral de comunicación inclusiva: Lengua de Señas Peruana a texto.
"""

import json, time, warnings
import numpy as np
import cv2
from pathlib import Path
from collections import deque
warnings.filterwarnings("ignore")

import gradio as gr
import mediapipe as mp
import onnxruntime as ort

# ── Configuración ─────────────────────────────────────────────────────────────

N_FRAMES    = 30
N_DIMS      = 150
N_KP        = 75
CONF_UMBRAL = 0.25

kp_buffer  = deque(maxlen=N_FRAMES)
historial  = deque(maxlen=10)

# ── MediaPipe ─────────────────────────────────────────────────────────────────

mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode=False, model_complexity=1,
    min_detection_confidence=0.4, min_tracking_confidence=0.4,
)

COLOR_LHAND = (255, 80, 80)
COLOR_RHAND = (80, 200, 80)
COLOR_POSE  = (80, 150, 255)
COLOR_FACE  = (200, 200, 200)

# ── Carga de modelo ───────────────────────────────────────────────────────────

ROOT = Path(__file__).parent

with open(ROOT / "lstm_label2idx.json", encoding="utf-8") as f:
    l2i = json.load(f)
idx2label = {int(v): k for k, v in l2i.items()}

session = ort.InferenceSession(str(ROOT / "lstm_signs.onnx"),
                                providers=["CPUExecutionProvider"])
print(f"Modelo listo — {len(idx2label)} señas LSP")

# ── Utilidades ────────────────────────────────────────────────────────────────

def extract_and_draw(frame_rgb):
    results = holistic.process(frame_rgb)
    vis = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    detected = {
        "mano_izq": results.left_hand_landmarks  is not None,
        "mano_der": results.right_hand_landmarks is not None,
        "cuerpo":   results.pose_landmarks       is not None,
        "rostro":   results.face_landmarks       is not None,
    }
    if results.face_landmarks:
        mp_drawing.draw_landmarks(vis, results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS, landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=COLOR_FACE, thickness=1, circle_radius=1))
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(vis, results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=COLOR_POSE, thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=COLOR_POSE, thickness=2))
    for lm_list, color in [(results.left_hand_landmarks, COLOR_LHAND),
                            (results.right_hand_landmarks, COLOR_RHAND)]:
        if lm_list:
            mp_drawing.draw_landmarks(vis, lm_list, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=color, thickness=2))

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


def kp_seq_to_features(kp_seq):
    pose  = kp_seq[:, 42:75, :2]
    left  = kp_seq[:,  0:21, :2]
    right = kp_seq[:, 21:42, :2]
    return np.concatenate([
        pose[:, :, 0], pose[:, :, 1],
        left[:, :, 0], left[:, :, 1],
        right[:, :, 0], right[:, :, 1],
    ], axis=1).astype(np.float32)


def run_inference(kp_seq):
    feat   = kp_seq_to_features(kp_seq)[np.newaxis]
    logits = session.run(None, {"sequence": feat})[0][0]
    probs  = np.exp(logits - logits.max())
    probs /= probs.sum()
    idx    = int(probs.argmax())
    top3   = [{"clase": idx2label.get(int(i), str(i)), "prob": float(probs[i])}
              for i in np.argsort(probs)[::-1][:3]]
    return {"seña": idx2label.get(idx, "?"), "confidence": float(probs[idx]),
            "top3": top3}


def draw_bar(vis, detected, result):
    h, w = vis.shape[:2]
    bar  = np.zeros((110, w, 3), dtype=np.uint8)
    parts = [("Mano izq.", detected["mano_izq"], COLOR_LHAND),
             ("Mano der.", detected["mano_der"], COLOR_RHAND),
             ("Cuerpo",    detected["cuerpo"],   COLOR_POSE),
             ("Rostro",    detected["rostro"],   COLOR_FACE)]
    x = 10
    for label, ok, color in parts:
        cv2.putText(bar, f"{'✓' if ok else '✗'} {label}",
                    (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color if ok else (80, 80, 80), 2)
        x += 155
    if result:
        conf = result["confidence"] * 100
        col  = (0, 220, 50) if conf >= 60 else (50, 180, 255) if conf >= 35 else (120, 120, 120)
        cv2.putText(bar, result["seña"], (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
        cv2.putText(bar, f"{conf:.0f}%", (w - 80, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
        cv2.rectangle(bar, (0, 102), (int(conf / 100 * w), 110), col, -1)
    else:
        n = len(kp_buffer)
        cv2.putText(bar, f"Acumulando… {n}/{N_FRAMES}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1)
        cv2.rectangle(bar, (0, 102), (int(n / N_FRAMES * w), 110), (0, 180, 255), -1)
    return np.vstack([vis, bar])


# ── Handlers ──────────────────────────────────────────────────────────────────

last_result = None


def webcam_frame(frame):
    global last_result
    if frame is None:
        return None, "Sin señal de cámara", ""
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    elif frame.shape[2] == 4:
        frame = frame[:, :, :3]

    vis, kp, detected = extract_and_draw(frame)
    kp_buffer.append(kp)

    if len(kp_buffer) >= N_FRAMES and len(kp_buffer) % (N_FRAMES // 2) == 0:
        r = run_inference(np.stack(list(kp_buffer)))
        if r["confidence"] >= CONF_UMBRAL:
            last_result = r
            if not historial or historial[-1] != r["seña"]:
                historial.append(r["seña"])

    vis_out = draw_bar(vis, detected, last_result)
    if last_result:
        r    = last_result
        txt  = f"## {r['seña']}\n\nConfianza: **{r['confidence']*100:.0f}%**"
        top3 = "\n".join(f"{i+1}. {t['clase']} {t['prob']*100:.1f}%"
                          for i, t in enumerate(r["top3"]))
        hist = "**Historial:** " + " › ".join(list(historial)[-8:])
        return vis_out, txt, top3 + "\n\n" + hist
    return vis_out, f"**Acumulando… {len(kp_buffer)}/{N_FRAMES}**", ""


def video_file(video_path):
    if not video_path:
        return None, "Sin video", ""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx_s = np.linspace(0, total - 1, N_FRAMES, dtype=int)
    frames_kp = []
    for idx in idx_s:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, fr = cap.read()
        if not ret:
            continue
        _, kp, _ = extract_and_draw(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        frames_kp.append(kp)
    cap.release()
    if len(frames_kp) < 5:
        return None, "Video demasiado corto", ""
    while len(frames_kp) < N_FRAMES:
        frames_kp.append(frames_kp[-1])
    r = run_inference(np.stack(frames_kp))
    txt  = f"## {r['seña']}\n\nConfianza: **{r['confidence']*100:.0f}%**"
    top3 = "\n".join(f"{i+1}. {t['clase']} {t['prob']*100:.1f}%"
                      for i, t in enumerate(r["top3"]))
    return video_path, txt, top3


# ── UI ────────────────────────────────────────────────────────────────────────

CSS = ".traduccion{font-size:1.5em!important;padding:14px 18px;border-left:5px solid #2196F3;background:#f0f7ff}"

with gr.Blocks(title="Traductor LSP", css=CSS) as demo:
    gr.Markdown(f"""
    # 🤟 Traductor LSP → Castellano
    Sistema de comunicación inclusiva · {len(idx2label)} señas peruanas · LSTM Bidireccional
    """)
    with gr.Tabs():
        with gr.TabItem("📷 Cámara en vivo"):
            with gr.Row():
                with gr.Column(scale=3):
                    cam_in  = gr.Image(sources=["webcam"], streaming=True, height=360)
                    cam_out = gr.Image(height=420)
                with gr.Column(scale=2):
                    cam_txt  = gr.Markdown("**Esperando señas…**", elem_classes=["traduccion"])
                    cam_top3 = gr.Markdown("")
            cam_in.stream(fn=webcam_frame, inputs=[cam_in],
                          outputs=[cam_out, cam_txt, cam_top3],
                          time_limit=300, stream_every=0.10)

        with gr.TabItem("🎬 Subir video"):
            with gr.Row():
                with gr.Column():
                    vid_in = gr.Video(label="MP4 / AVI / MOV")
                    btn    = gr.Button("▶ Traducir", variant="primary", size="lg")
                with gr.Column():
                    vid_txt  = gr.Markdown("", elem_classes=["traduccion"])
                    vid_top3 = gr.Markdown("")
            btn.click(fn=video_file, inputs=[vid_in],
                      outputs=[vid_in, vid_txt, vid_top3])

if __name__ == "__main__":
    demo.launch()
