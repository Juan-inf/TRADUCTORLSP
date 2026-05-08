"""
Demo HuggingFace Spaces — Traductor LSP en tiempo real.
Usar con: python app_gradio.py
Desplegar en HuggingFace Spaces subiendo este archivo como app.py
"""

import json
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import gradio as gr
except ImportError:
    raise ImportError("pip install gradio>=4.0")

# ── Configuración ────────────────────────────────────────────────────────────

CHECKPOINT = "checkpoints/lsp_model.onnx"        # preferir ONNX
LABEL2IDX  = "data/label2idx.json"
N_FRAMES   = 30
IMG_SIZE   = (224, 224)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32)

# Estado del buffer por sesión
frame_buffer = []
last_prediction = {"seña": "—", "confidence": 0.0}

# ── Carga del modelo ─────────────────────────────────────────────────────────

predictor = None

def load_predictor():
    global predictor
    if not Path(LABEL2IDX).exists():
        print("label2idx.json no encontrado — usando predictor demo")
        return

    if Path(CHECKPOINT).exists():
        try:
            import onnxruntime as ort
            providers = ['CPUExecutionProvider']
            session = ort.InferenceSession(CHECKPOINT, providers=providers)

            with open(LABEL2IDX) as f:
                l2i = json.load(f)
            idx2label = {int(v): k for k, v in l2i.items()}

            predictor = {'session': session, 'idx2label': idx2label, 'type': 'onnx'}
            print(f"Modelo ONNX listo. Clases: {len(idx2label)}")
        except Exception as e:
            print(f"Error cargando ONNX: {e}")

load_predictor()

# ── Funciones auxiliares ─────────────────────────────────────────────────────

def preprocess_frame(frame_rgb: np.ndarray) -> np.ndarray:
    frame = cv2.resize(frame_rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    frame = frame.astype(np.float32) / 255.0
    frame = (frame - IMAGENET_MEAN) / IMAGENET_STD
    return frame


def predict_from_buffer(frames: list) -> dict:
    if predictor is None:
        # Demo sin modelo: retornar clase aleatoria para testing
        demo_classes = ["HOLA", "GRACIAS", "BIEN", "AGUA", "AYUDA"]
        import random
        return {"seña": random.choice(demo_classes), "confidence": round(random.uniform(0.65, 0.95), 2)}

    arr = np.stack(frames[-N_FRAMES:])                 # [T, H, W, C]
    arr = np.transpose(arr, (3, 0, 1, 2))              # [C, T, H, W]
    pixels = arr[np.newaxis].astype(np.float32)        # [1, C, T, H, W]

    t0 = time.perf_counter()

    if predictor['type'] == 'onnx':
        try:
            input_name = predictor['session'].get_inputs()[0].name
            logits = predictor['session'].run(None, {input_name: pixels})[0]  # [1, N]
            probs = np.exp(logits[0] - logits[0].max())
            probs /= probs.sum()
        except Exception as e:
            return {"seña": f"Error: {e}", "confidence": 0.0}
    else:
        return {"seña": "Modelo no cargado", "confidence": 0.0}

    lat = (time.perf_counter() - t0) * 1000
    idx = int(probs.argmax())
    confidence = float(probs[idx])
    seña = predictor['idx2label'].get(idx, f"clase_{idx}")

    top3 = sorted(zip(probs, predictor['idx2label'].values()), reverse=True)[:3]

    return {
        "seña": seña,
        "confidence": confidence,
        "latency_ms": lat,
        "top3": [(s, float(p)) for p, s in top3],
    }


# ── Handlers de Gradio ───────────────────────────────────────────────────────

def process_webcam_frame(frame):
    """Llamado por Gradio cada vez que hay un nuevo frame de cámara."""
    global frame_buffer

    if frame is None:
        return None, "Sin señal de cámara", ""

    # Preprocesar
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.shape[2] == 3 else frame
    frame_norm = preprocess_frame(frame_rgb)
    frame_buffer.append(frame_norm)

    if len(frame_buffer) > N_FRAMES * 2:
        frame_buffer = frame_buffer[-N_FRAMES:]

    # Predecir cuando hay suficientes frames
    result_text = "Acumulando frames..."
    top3_text = ""

    if len(frame_buffer) >= N_FRAMES:
        result = predict_from_buffer(frame_buffer)
        seña = result.get("seña", "—")
        conf = result.get("confidence", 0.0) * 100
        lat  = result.get("latency_ms", 0.0)

        result_text = f"**{seña}** ({conf:.1f}%)"
        top3 = result.get("top3", [])
        top3_text = "\n".join([f"{i+1}. {s}: {p*100:.1f}%" for i, (s, p) in enumerate(top3)])

    # Anotar frame
    h, w = frame.shape[:2]
    vis = frame.copy()
    if len(frame_buffer) >= N_FRAMES:
        color = (0, 220, 50) if conf >= 70 else (50, 100, 255)
        cv2.putText(vis, seña, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.putText(vis, f"{conf:.0f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    else:
        pct = int(len(frame_buffer) / N_FRAMES * w)
        cv2.rectangle(vis, (0, h - 8), (pct, h), (0, 180, 255), -1)

    return vis, result_text, top3_text


def process_video_file(video_path):
    """Procesa un video subido por el usuario."""
    global frame_buffer
    frame_buffer = []

    if video_path is None:
        return None, "No se subió ningún video", ""

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, N_FRAMES, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(preprocess_frame(frame_rgb))
    cap.release()

    if len(frames) < 5:
        return None, "Video demasiado corto", ""

    result = predict_from_buffer(frames)
    seña = result.get("seña", "—")
    conf = result.get("confidence", 0.0) * 100
    lat  = result.get("latency_ms", 0.0)

    result_text = f"**{seña}** (confianza: {conf:.1f}%, latencia: {lat:.0f}ms)"
    top3 = result.get("top3", [])
    top3_text = "**Top 3 predicciones:**\n" + "\n".join(
        [f"{i+1}. {s}: {p*100:.1f}%" for i, (s, p) in enumerate(top3)]
    )

    return video_path, result_text, top3_text


# ── Interfaz Gradio ──────────────────────────────────────────────────────────

with gr.Blocks(
    title="Traductor LSP — Lengua de Señas Peruana",
    theme=gr.themes.Soft(),
    css=".result-box { font-size: 2em !important; text-align: center; padding: 20px; }",
) as demo:

    gr.Markdown("""
    # 🤟 Traductor de Lengua de Señas Peruana (LSP)
    ### Sistema de reconocimiento en tiempo real con Deep Learning

    **Modo cámara**: Realiza una seña frente a la cámara durante 1-2 segundos.
    **Modo video**: Sube un video MP4 con una seña para analizarla.

    > Modelo: CNN-LSTM + ST-GCN Fusion | Dataset: LSP Peru
    """)

    with gr.Tabs():
        # ── Tab 1: Cámara web ─────────────────────────────────────────
        with gr.TabItem("📷 Cámara en Tiempo Real"):
            with gr.Row():
                with gr.Column(scale=2):
                    webcam_input = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        label="Cámara",
                        height=400,
                    )
                with gr.Column(scale=1):
                    webcam_output = gr.Image(label="Vista procesada", height=400)
                    result_display = gr.Markdown(
                        "**Esperando señas...**",
                        elem_classes=["result-box"],
                    )
                    top3_display = gr.Markdown("")

            webcam_input.stream(
                fn=process_webcam_frame,
                inputs=[webcam_input],
                outputs=[webcam_output, result_display, top3_display],
                time_limit=30,
                stream_every=0.1,
            )

        # ── Tab 2: Video ──────────────────────────────────────────────
        with gr.TabItem("🎬 Video"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Subir video MP4")
                    analyze_btn = gr.Button("Analizar seña", variant="primary", size="lg")

                with gr.Column():
                    video_preview = gr.Video(label="Video analizado")
                    video_result  = gr.Markdown("", elem_classes=["result-box"])
                    video_top3    = gr.Markdown("")

            analyze_btn.click(
                fn=process_video_file,
                inputs=[video_input],
                outputs=[video_preview, video_result, video_top3],
            )

        # ── Tab 3: Info ───────────────────────────────────────────────
        with gr.TabItem("ℹ️ Información"):
            gr.Markdown("""
            ## Arquitectura del Sistema

            ### Pipeline de reconocimiento
            ```
            Video MP4 / Frame de Cámara
                    ↓
            Rama A: ResNet50 → BiLSTM (características visuales)
            Rama B: MediaPipe Holistic → ST-GCN (landmarks de manos)
                    ↓
            Fusión Multimodal (concatenación + MLP)
                    ↓
            Softmax → Clase LSP → Texto Español
            ```

            ### Tecnologías
            - **PyTorch 2.x** — Framework de entrenamiento
            - **MediaPipe Holistic** — Extracción de 75 keypoints (manos + pose)
            - **ResNet50 + BiLSTM** — Características espaciotemporales
            - **ST-GCN** — Grafo espacio-temporal sobre esqueleto
            - **ONNX Runtime** — Inferencia optimizada en producción
            - **FastAPI** — API REST para integración institucional

            ### Métricas objetivo
            | Métrica | Objetivo |
            |---------|---------|
            | Accuracy Top-1 | >85% |
            | F1-macro | >80% |
            | Latencia | <200ms |

            ### Dataset
            Dataset LSP (Lengua de Señas Peruana) — Videos MP4 por seña.

            ### Despliegue
            Sistema desplegable en instituciones educativas y de salud del Perú.
            Compatible con CPU (laptops escolares) a través de ONNX Runtime.
            """)

    gr.Markdown("""
    ---
    *Sistema desarrollado para accesibilidad en Perú.
    Soporta despliegue en HuggingFace Spaces, FastAPI y edge devices.*
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,    # genera URL pública temporal
        show_error=True,
    )
