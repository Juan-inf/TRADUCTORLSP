"""
API REST FastAPI para inferencia LSP desde cámara web o video.

Endpoints:
  POST /predict/video       — sube un video MP4 y obtiene la seña
  POST /predict/frame       — sube un frame JPEG/PNG y acumula buffer
  GET  /predict/stream      — WebSocket para inferencia en tiempo real
  GET  /health              — estado del servicio
  GET  /classes             — lista de clases LSP disponibles
"""

import io
import json
import time
import base64
import asyncio
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import torch
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.predictor import LSPPredictor, ONNXPredictor

# ── Configuración ────────────────────────────────────────────────────────────

CONFIG = {
    "checkpoint_path": "checkpoints/lsp_best.pt",
    "onnx_path":       "checkpoints/lsp_model.onnx",
    "label2idx_path":  "data/label2idx.json",
    "n_frames":        30,
    "img_size":        (224, 224),
    "device":          "cuda" if torch.cuda.is_available() else "cpu",
    "confidence_threshold": 0.65,
    "mode":            "both",
}

app = FastAPI(
    title="Traductor LSP",
    description="API de Lengua de Señas Peruana — Deep Learning en tiempo real",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Estado global del predictor ──────────────────────────────────────────────

predictor: Optional[LSPPredictor] = None
idx2label: dict = {}


@app.on_event("startup")
async def load_model():
    global predictor, idx2label

    if not Path(CONFIG["label2idx_path"]).exists():
        print("ADVERTENCIA: label2idx.json no encontrado. Ejecutar EDA primero.")
        return

    with open(CONFIG["label2idx_path"]) as f:
        label2idx = json.load(f)
    idx2label = {int(v): k for k, v in label2idx.items()}

    if Path(CONFIG["onnx_path"]).exists():
        print("Cargando modelo ONNX (optimizado)...")
        predictor = ONNXPredictor(
            onnx_path=CONFIG["onnx_path"],
            label2idx_path=CONFIG["label2idx_path"],
            n_frames=CONFIG["n_frames"],
            img_size=CONFIG["img_size"],
        )
        print(f"Modelo ONNX listo. Clases: {len(idx2label)}")
    else:
        print("Modelo ONNX no encontrado. Verificar exportación.")


# ── Modelos Pydantic ─────────────────────────────────────────────────────────

class PredictionResult(BaseModel):
    seña:       str
    confidence: float
    latency_ms: float
    top3:       List[dict]


class HealthResponse(BaseModel):
    status:     str
    model_ready: bool
    device:     str
    n_classes:  int


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return {
        "status":      "ok",
        "model_ready": predictor is not None,
        "device":      CONFIG["device"],
        "n_classes":   len(idx2label),
    }


@app.get("/classes")
async def list_classes():
    return {"classes": sorted(idx2label.values()), "total": len(idx2label)}


@app.post("/predict/video", response_model=PredictionResult)
async def predict_video(file: UploadFile = File(...)):
    """Recibe un video MP4 y retorna la seña detectada."""
    if predictor is None:
        raise HTTPException(503, "Modelo no cargado")

    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.webm')):
        raise HTTPException(400, "Formato de video no soportado")

    t0 = time.perf_counter()

    # Leer video desde bytes
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    tmp_path = f"/tmp/lsp_upload_{int(time.time())}.mp4"
    with open(tmp_path, 'wb') as f:
        f.write(contents)

    # Extraer frames
    frames = _extract_frames_from_file(tmp_path, CONFIG["n_frames"], CONFIG["img_size"])
    Path(tmp_path).unlink(missing_ok=True)

    if frames is None:
        raise HTTPException(422, "No se pudo procesar el video")

    # Inferencia
    result = _run_inference(frames)
    result['latency_ms'] = (time.perf_counter() - t0) * 1000

    return result


@app.post("/predict/frame")
async def predict_frame(
    file: UploadFile = File(...),
    session_id: str = "default",
):
    """
    Recibe un frame individual (JPEG/PNG).
    Acumula en buffer por session_id y retorna predicción cuando hay suficientes.
    """
    if predictor is None:
        raise HTTPException(503, "Modelo no cargado")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(422, "No se pudo decodificar la imagen")

    # Por simplicidad, este endpoint procesa un frame a la vez
    # En producción, usar el WebSocket para streaming real
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_norm = _preprocess_single_frame(frame_rgb, CONFIG["img_size"])

    return {"status": "frame_received", "note": "Usar /predict/stream para tiempo real"}


@app.websocket("/predict/stream")
async def websocket_predict(websocket: WebSocket):
    """
    WebSocket para inferencia en tiempo real.

    Cliente envía frames como base64 JPEG.
    Servidor responde con predicciones JSON.

    Protocolo:
      Client → {"frame": "<base64>", "include_landmarks": true}
      Server → {"seña": "HOLA", "confidence": 0.92, "latency_ms": 145}
    """
    await websocket.accept()
    frame_buffer = []
    kp_buffer = []

    try:
        while True:
            data = await websocket.receive_json()
            t0 = time.perf_counter()

            # Decodificar frame
            frame_b64 = data.get("frame", "")
            frame_bytes = base64.b64decode(frame_b64)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame_bgr is None:
                await websocket.send_json({"error": "frame inválido"})
                continue

            # Preprocesar y agregar al buffer
            frame_norm = _preprocess_single_frame(
                cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                CONFIG["img_size"],
            )
            frame_buffer.append(frame_norm)

            # Mantener solo los últimos n_frames
            if len(frame_buffer) > CONFIG["n_frames"]:
                frame_buffer.pop(0)

            # Predecir cuando el buffer está lleno
            if len(frame_buffer) == CONFIG["n_frames"]:
                pixels = np.stack(frame_buffer)                    # [T, H, W, C]
                pixels = np.transpose(pixels, (3, 0, 1, 2))       # [C, T, H, W]
                pixels = pixels[np.newaxis]                        # [1, C, T, H, W]

                result = _run_inference(pixels)
                result['latency_ms'] = (time.perf_counter() - t0) * 1000

                await websocket.send_json(result)
            else:
                await websocket.send_json({
                    "status": "buffering",
                    "frames_collected": len(frame_buffer),
                    "frames_needed": CONFIG["n_frames"],
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


# ── Funciones auxiliares ─────────────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32)


def _preprocess_single_frame(
    frame_rgb: np.ndarray,
    img_size: tuple,
) -> np.ndarray:
    frame = cv2.resize(frame_rgb, img_size, interpolation=cv2.INTER_LINEAR)
    frame = frame.astype(np.float32) / 255.0
    frame = (frame - IMAGENET_MEAN) / IMAGENET_STD
    return frame


def _extract_frames_from_file(
    video_path: str,
    n_frames: int,
    img_size: tuple,
) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, n_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(_preprocess_single_frame(frame_rgb, img_size))

    cap.release()

    if not frames:
        return None

    # Pad si hace falta
    while len(frames) < n_frames:
        frames.append(frames[-1])

    arr = np.stack(frames[:n_frames])             # [T, H, W, C]
    arr = np.transpose(arr, (3, 0, 1, 2))         # [C, T, H, W]
    return arr[np.newaxis]                         # [1, C, T, H, W]


def _run_inference(pixels: np.ndarray) -> dict:
    """Ejecuta inferencia ONNX o PyTorch sobre batch de pixels."""
    global predictor, idx2label

    if isinstance(predictor, ONNXPredictor):
        result = predictor.predict(pixels)
    else:
        # PyTorch
        with torch.no_grad():
            t = torch.from_numpy(pixels).float().to(CONFIG["device"])
            logits = predictor.model(t)
            import torch.nn.functional as F
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        idx = int(probs.argmax())
        result = {
            'seña': idx2label.get(idx, str(idx)),
            'confidence': float(probs[idx]),
            'probs': probs,
        }

    # Top-3
    probs = result.pop('probs', None)
    if probs is not None:
        top3_idx = np.argsort(probs)[::-1][:3]
        result['top3'] = [
            {'seña': idx2label.get(int(i), str(i)), 'confidence': float(probs[i])}
            for i in top3_idx
        ]
    else:
        result['top3'] = []

    return result


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=1)
