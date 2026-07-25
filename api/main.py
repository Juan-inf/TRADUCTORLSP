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
import unicodedata
import numpy as np
import cv2
import mediapipe as mp
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import torch
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.predictor import ONNXPredictor
from src.features.landmarks import (
    results_to_kp, kp_seq_to_features, normalize_sample, resample_to_n_frames,
    aplicar_respaldo_manos_y_pose,
)
from src.features.segmentacion import SegmentadorPausas

# ── Configuración ────────────────────────────────────────────────────────────

CONFIG = {
    "onnx_path":       "checkpoints/bilstm_s27.onnx",
    "label2idx_path":  "data/s27_label2idx.json",
    # Ensemble (2026-07-19) — checkpoints/bilstm_s29.onnx es un segundo modelo
    # entrenado con más datos reales sobre el mismo vocabulario de 96 clases.
    # Solo, da un F1 parecido a v4 (0.42 vs 0.44), pero es el único que pasa
    # HE3 completo (ΔF1/PSI/KS) con márgenes amplios. Validado en un holdout
    # limpio (fuera del entrenamiento de AMBOS modelos, 342 muestras, 72
    # clases, ver scripts/validar_ensemble_v4_s29.py): promediar sus
    # probabilidades da F1=0.4424 — mejor que cualquiera de los dos solos, y
    # hereda la generalización de S29. Si el checkpoint no existe, la API
    # sigue funcionando solo con v4 (sin ensemble).
    "onnx_path_ensemble":      "checkpoints/bilstm_s29.onnx",
    "label2idx_path_ensemble": "data/s29_label2idx.json",
    "n_frames":        30,
    "device":          "cuda" if torch.cuda.is_available() else "cpu",
    # Calibrado con 15 clips reales (2026-07-17) en demo/app_gradio.py: con 0.30
    # se ocultaban aciertos reales (ej. "AHORA" correcto con 21% de confianza,
    # descartado). Portado aquí para que /predict/stream filtre igual que la
    # demo (Riesgo R9 — antes este valor no se usaba para nada).
    "confidence_threshold": 0.20,
    # R7 — límite de tamaño de upload para /predict/video. 150MB cubre videos
    # reales de varios minutos (el video de prueba del repo pesa ~25MB); evita
    # que un archivo enorme agote memoria/tiempo del servidor.
    "max_upload_mb": 150,
}

holistic = mp.solutions.holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4,
)

# Respaldo de manos compartido entre /predict/stream y /predict/video — mismo
# mecanismo que demo/app_gradio.py (ver src.features.landmarks.
# aplicar_respaldo_manos_y_pose y ENTREGABLE_PLAN_DE_DESPLIEGUE_S13.md R14),
# para que la API se comporte igual que la demo ante encuadres de mano sola
# sin cuerpo (p.ej. abecedario). Instancia persistente, no se recrea por frame.
hands_fallback = mp.solutions.hands.Hands(
    static_image_mode=False, model_complexity=1,
    min_detection_confidence=0.3, max_num_hands=2,
)

# ── Estado global del predictor ──────────────────────────────────────────────

predictor:            Optional[ONNXPredictor] = None
predictor_ensemble:   Optional[ONNXPredictor] = None
idx2label:            dict = {}
idx2label_ensemble:   dict = {}
clase_texto:          dict = {}   # clase_id → texto legible en castellano
_ens_idx_map:         dict = {}   # idx_s29 → idx_v4 (mismo nombre de clase), precalculado


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, predictor_ensemble, idx2label, idx2label_ensemble, clase_texto
    # Mapeo clase → texto castellano
    clase_texto_path = Path("data/clase_texto.json")
    if clase_texto_path.exists():
        with open(clase_texto_path, encoding="utf-8") as f:
            clase_texto = json.load(f)

    if not Path(CONFIG["label2idx_path"]).exists():
        print("ADVERTENCIA: label2idx.json no encontrado.")
    else:
        with open(CONFIG["label2idx_path"]) as f:
            label2idx = json.load(f)
        idx2label = {int(v): k for k, v in label2idx.items()}

        if Path(CONFIG["onnx_path"]).exists():
            print("Cargando modelo ONNX (v4)...")
            predictor = ONNXPredictor(
                onnx_path=CONFIG["onnx_path"],
                label2idx_path=CONFIG["label2idx_path"],
                n_frames=CONFIG["n_frames"],
            )
            print(f"Modelo ONNX listo — {len(idx2label)} LSP - Vocabulario-palabras")
        else:
            print(f"ONNX no encontrado: {CONFIG['onnx_path']}")

        # Segundo modelo del ensemble (opcional — si falta, sigue con v4 solo)
        onnx_ens_path = Path(CONFIG["onnx_path_ensemble"])
        l2i_ens_path = Path(CONFIG["label2idx_path_ensemble"])
        if onnx_ens_path.exists() and l2i_ens_path.exists():
            print("Cargando modelo ONNX (S29, ensemble)...")
            with open(l2i_ens_path, encoding="utf-8") as f:
                label2idx_ens = json.load(f)
            idx2label_ensemble = {int(v): k for k, v in label2idx_ens.items()}
            predictor_ensemble = ONNXPredictor(
                onnx_path=str(onnx_ens_path),
                label2idx_path=str(l2i_ens_path),
                n_frames=CONFIG["n_frames"],
            )
            # NFC — v4 guarda tildes en forma decompuesta (NFD), S29 en forma
            # precompuesta (NFC); sin normalizar, 8/96 clases con tilde no
            # alinean y su probabilidad en el ensemble se reduce a la mitad
            # en vez de promediarse correctamente. Encontrado en vivo (2026-07-19).
            nombre_a_idx_v4 = {unicodedata.normalize("NFC", n): i for i, n in idx2label.items()}
            global _ens_idx_map
            _ens_idx_map = {
                idx_s29: nombre_a_idx_v4[unicodedata.normalize("NFC", nombre)]
                for idx_s29, nombre in idx2label_ensemble.items()
                if unicodedata.normalize("NFC", nombre) in nombre_a_idx_v4
            }
            print(f"Modelo ONNX (ensemble) listo — {len(idx2label_ensemble)} clases "
                  f"({len(_ens_idx_map)} alineadas con v4)")
        else:
            print(f"Ensemble no disponible ({onnx_ens_path} no encontrado) — sirviendo solo con v4.")
    yield


app = FastAPI(
    title="Traductor LSP",
    description="API de Lengua de Señas Peruana — Deep Learning en tiempo real",
    version="1.0.0",
    lifespan=lifespan,
)

# R7 — antes abierto a cualquier origen (allow_origins=["*"]). Restringido a
# los orígenes locales reales desde los que se sirve el sistema hoy (demo
# Gradio, desarrollo local). Si se expone fuera de una red controlada, agregar
# el dominio real aquí explícitamente — no volver a "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:7860", "http://127.0.0.1:7860",   # demo/app_gradio.py
        "http://localhost:8000", "http://127.0.0.1:8000",   # esta misma API
        "http://localhost:3000", "http://127.0.0.1:3000",   # frontend local, si se agrega
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Modelos Pydantic ─────────────────────────────────────────────────────────

class PredictionResult(BaseModel):
    clase:           str
    texto_castellano: str
    confidence:      float
    latency_ms:      float
    top3:            List[dict]


class HealthResponse(BaseModel):
    status:     str
    model_ready: bool
    ensemble_ready: bool
    device:     str
    n_classes:  int


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return {
        "status":      "ok",
        "model_ready": predictor is not None,
        "ensemble_ready": predictor_ensemble is not None,
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

    # Leer video desde bytes, en chunks — corta apenas se supera el límite en
    # vez de esperar a tener todo el archivo en memoria (R7).
    max_bytes = CONFIG["max_upload_mb"] * 1024 * 1024
    chunks = bytearray()
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        chunks.extend(chunk)
        if len(chunks) > max_bytes:
            raise HTTPException(413, f"Archivo excede el límite de {CONFIG['max_upload_mb']}MB")
    contents = bytes(chunks)

    nparr = np.frombuffer(contents, np.uint8)
    tmp_path = f"/tmp/lsp_upload_{int(time.time())}.mp4"
    with open(tmp_path, 'wb') as f:
        f.write(contents)

    # Extraer secuencia de landmarks
    seq = _extract_landmark_sequence_from_file(tmp_path, CONFIG["n_frames"])
    Path(tmp_path).unlink(missing_ok=True)

    if seq is None:
        raise HTTPException(422, "No se pudo procesar el video")

    # Inferencia
    result = _run_inference(seq)
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

    # Por simplicidad, este endpoint solo confirma recepción de un frame.
    # El modelo necesita una secuencia de 30 frames — usar /predict/stream.
    return {"status": "frame_received", "note": "Usar /predict/stream para tiempo real"}


@app.websocket("/predict/stream")
async def websocket_predict(websocket: WebSocket):
    """
    WebSocket para inferencia en tiempo real.

    Cliente envía frames como base64 JPEG.
    Servidor responde con predicciones JSON.

    Protocolo:
      Client → {"frame": "<base64>", "include_landmarks": true}
      Server → {"clase": "HOLA", "confidence": 0.92, "latency_ms": 145, ...}

    A diferencia de una versión anterior (ventana fija de 30 frames sin
    filtrar), esto usa el mismo criterio que demo/app_gradio.py (Riesgo R9,
    resuelto): segmentación por pausas para decidir CUÁNDO clasificar (no
    cada N frames fijos), filtro de umbral de confianza, y exclusión de
    clases narrativas (HISTORIAS_VINETAS_*) que no tiene sentido detectar
    en una ventana de streaming corta.
    """
    await websocket.accept()
    segmentador = SegmentadorPausas()

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

            # Extraer landmarks del frame y pasarlos al segmentador por pausas
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            results = aplicar_respaldo_manos_y_pose(results, frame_rgb, hands_fallback)
            kp = results_to_kp(results)

            segmento = segmentador.push(kp)
            if segmento is not None:
                kp_seq = resample_to_n_frames(segmento, CONFIG["n_frames"])
                feat   = kp_seq_to_features(kp_seq)                 # [T, 150]
                feat   = normalize_sample(feat)
                seq    = feat[np.newaxis]                            # [1, T, 150]

                result = _run_inference(seq)
                result['latency_ms'] = (time.perf_counter() - t0) * 1000

                if (result["confidence"] >= CONFIG["confidence_threshold"]
                        and not _es_clase_narrativa(result["clase"])):
                    await websocket.send_json(result)
                else:
                    await websocket.send_json({
                        "status":            "below_threshold",
                        "clase_descartada":  result["clase"],
                        "confidence":        result["confidence"],
                    })
            else:
                await websocket.send_json({
                    "status": "buffering",
                    "frames_collected": len(segmentador),
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

def _extract_landmark_sequence_from_file(
    video_path: str,
    n_frames: int,
) -> Optional[np.ndarray]:
    """Muestrea n_frames uniformemente del video, extrae landmarks MediaPipe
    por frame, y arma la secuencia normalizada [1, n_frames, 150] lista para
    el modelo — mismo contrato que scripts/train_s27.py."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, n_frames, dtype=int)

    kps = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            results = aplicar_respaldo_manos_y_pose(results, frame_rgb, hands_fallback)
            kps.append(results_to_kp(results))

    cap.release()

    if not kps:
        return None

    while len(kps) < n_frames:
        kps.append(kps[-1])

    kp_seq = np.stack(kps[:n_frames])              # [T, 75, 3]
    feat   = kp_seq_to_features(kp_seq)             # [T, 150]
    feat   = normalize_sample(feat)
    return feat[np.newaxis]                          # [1, T, 150]


def _clase_a_texto(clase_id: str) -> str:
    return clase_texto.get(clase_id, clase_id)


def _es_clase_narrativa(clase: str) -> bool:
    """HISTORIAS_VINETAS_N es la clase 'este clip ENTERO de 1-9 minutos es la
    narrativa N' — no una seña puntual. No tiene sentido que aparezca como
    detección dentro de un segmento corto de streaming: ninguna persona puede
    'firmar' un video ajeno en unos segundos. Mismo criterio que
    demo/app_gradio.py (Riesgo R9)."""
    return clase.startswith("HISTORIAS_VINETAS_")


def _run_inference(sequence: np.ndarray) -> dict:
    """Ejecuta inferencia ONNX sobre la secuencia de landmarks y devuelve texto castellano.

    Si el segundo modelo del ensemble está cargado, promedia las
    probabilidades de v4 y S29 (alineadas por NOMBRE de clase, no por índice
    — cada modelo tiene su propio orden interno de las 96 clases). Validado
    en scripts/validar_ensemble_v4_s29.py sobre un holdout limpio para
    ambos modelos: F1=0.4424, mejor que cualquiera de los dos solos."""
    global predictor, predictor_ensemble, idx2label, idx2label_ensemble

    result = predictor.predict(sequence)
    probs_v4 = result.pop('probs', None)
    result.pop('seña', None)

    if probs_v4 is None:
        result['clase'] = ''
        result['texto_castellano'] = ''
        result['top3'] = []
        return result

    if predictor_ensemble is not None:
        result_ens = predictor_ensemble.predict(sequence)
        probs_s29 = result_ens.get('probs')
        # Remapear las probs de S29 al orden de índices de v4 (precalculado en lifespan)
        probs_s29_alineadas = np.zeros_like(probs_v4)
        for idx_s29, idx_v4 in _ens_idx_map.items():
            probs_s29_alineadas[idx_v4] = probs_s29[idx_s29]
        probs = (probs_v4 + probs_s29_alineadas) / 2
    else:
        probs = probs_v4

    idx = int(probs.argmax())
    clase_id = idx2label.get(idx, str(idx))
    result['clase']            = clase_id
    result['confidence']       = float(probs[idx])
    result['texto_castellano'] = _clase_a_texto(clase_id)

    top3_idx = np.argsort(probs)[::-1][:3]
    result['top3'] = [
        {
            'clase':           idx2label.get(int(i), str(i)),
            'texto_castellano': _clase_a_texto(idx2label.get(int(i), str(i))),
            'confidence':      float(probs[i]),
        }
        for i in top3_idx
    ]

    return result


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=1)
