"""
Traductor LSP → Castellano  — Sprint 27
MediaPipe Holistic + BiLSTM Bidireccional S27 (96 señas LSP, F1=0.4349, Top-5=63.6%)
Tiempo real desde cámara web o video pregrabado.
"""

import json, time, pickle, warnings, tempfile
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

from src.features.landmarks import (
    results_to_kp as _results_to_kp,
    kp_seq_to_features as kp_seq_to_lstm_features,
    normalize_sample,
    resample_to_n_frames,
)
from src.features.segmentacion import SegmentadorPausas

# ── Configuración ─────────────────────────────────────────────────────────────

LSTM_ONNX      = "checkpoints/bilstm_s27.onnx"
RF_CKPT        = "checkpoints/rf_signs.pkl"
LABEL_PATH     = "data/s27_label2idx.json"
CLASE_TEXTO_PATH = "data/clase_texto.json"

N_FRAMES    = 30
N_DIMS      = 150   # pose(33×2) + left_hand(21×2) + right_hand(21×2)
# Calibrado con 15 clips reales (2026-07-17): con 0.30 se ocultaban aciertos
# reales (ej. "AHORA" correcto con 21% de confianza, descartado). El modelo
# tiene 96 clases muy parecidas entre sí — la confianza cruda tiende a
# repartirse, no a concentrarse, incluso cuando acierta. 0.20 capturó el 100%
# de los aciertos de la muestra sin dejar pasar mucho más ruido.
CONF_UMBRAL = 0.20
STRIDE      = N_FRAMES // 2   # ventana deslizante 50% overlap

# ── Buffers globales (webcam) ─────────────────────────────────────────────────

kp_buffer     = deque(maxlen=N_FRAMES)  # solo para el contador "acumulando..." en pantalla
historial     = deque(maxlen=10)
segmentador_cam = SegmentadorPausas()

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
clase_texto  = {}   # clase → descripción legible (solo clases sin traducción 1:1, ej. viñetas)


def load_models():
    global lstm_session, rf_model, idx2label, clase_texto

    if Path(LABEL_PATH).exists():
        with open(LABEL_PATH, encoding="utf-8") as f:
            l2i = json.load(f)
        idx2label = {int(v): k for k, v in l2i.items()}
        print(f"Etiquetas cargadas: {len(idx2label)} señas LSP")

    if Path(CLASE_TEXTO_PATH).exists():
        with open(CLASE_TEXTO_PATH, encoding="utf-8") as f:
            clase_texto = json.load(f)

    if Path(LSTM_ONNX).exists():
        try:
            avail = ort.get_available_providers()
            provs = [p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider")
                     if p in avail]
            lstm_session = ort.InferenceSession(LSTM_ONNX, providers=provs)
            inp = lstm_session.get_inputs()[0]
            print(f"BiLSTM S27 ONNX listo — input={inp.name} {inp.shape} — {len(idx2label)} etiquetas")
        except Exception as e:
            print(f"BiLSTM S27 error: {e}")

    if Path(RF_CKPT).exists() and lstm_session is None:
        try:
            with open(RF_CKPT, "rb") as f:
                data = pickle.load(f)
            rf_model = data["pipeline"]
            if not idx2label:
                idx2label.update(data["idx2label"])
            print(f"RF fallback listo — {data['n_classes']} clases")
        except Exception as e:
            print(f"RF error: {e}")


load_models()

# ── Helpers MediaPipe ─────────────────────────────────────────────────────────

def _draw_landmarks(vis_bgr: np.ndarray, results) -> None:
    """Dibuja todos los landmarks detectados sobre el frame BGR (in-place)."""
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            vis_bgr, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=COLOR_FACE, thickness=1, circle_radius=1))
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            vis_bgr, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=COLOR_POSE, thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=COLOR_POSE, thickness=2))
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            vis_bgr, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=COLOR_LHAND, thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=COLOR_LHAND, thickness=2))
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            vis_bgr, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=COLOR_RHAND, thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=COLOR_RHAND, thickness=2))


def _make_detected(results) -> dict:
    return {
        "mano_izq": results.left_hand_landmarks  is not None,
        "mano_der": results.right_hand_landmarks is not None,
        "cuerpo":   results.pose_landmarks       is not None,
        "rostro":   results.face_landmarks       is not None,
    }


# ── Extracción + dibujo (webcam) ──────────────────────────────────────────────

def extract_and_draw(frame_rgb: np.ndarray):
    """Ejecuta MediaPipe, dibuja landmarks y devuelve (vis_bgr, kp[75,3], detected)."""
    results = holistic.process(frame_rgb)
    vis = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    _draw_landmarks(vis, results)
    kp = _results_to_kp(results)
    return vis, kp, _make_detected(results)


# ── Feature extraction ────────────────────────────────────────────────────────
# kp_seq_to_lstm_features y normalize_sample vienen de src/features/landmarks.py
# (compartido con api/main.py, debe coincidir exactamente con el entrenamiento).

def kp_seq_to_rf_features(kp_seq: np.ndarray) -> np.ndarray:
    """kp_seq [T,75,3] → [108] para RF fallback."""
    pose  = kp_seq[:, 42:75, :2].mean(0)
    right = kp_seq[:, 21:42, :2].mean(0)
    return np.concatenate([pose[:, 0], pose[:, 1],
                            right[:, 0], right[:, 1]]).astype(np.float32)


# ── Inferencia ────────────────────────────────────────────────────────────────

def run_inference(kp_seq: np.ndarray) -> dict | None:
    """Inferencia LSTM (o RF fallback) sobre secuencia [T, 75, 3]."""
    t0 = time.perf_counter()

    if lstm_session is not None:
        feat   = normalize_sample(kp_seq_to_lstm_features(kp_seq))[np.newaxis]   # (1, 30, 150)
        logits = lstm_session.run(None, {"sequence": feat})[0][0]
        probs  = np.exp(logits - logits.max())
        probs /= probs.sum()
        idx    = int(probs.argmax())
        top3   = [{"clase": idx2label.get(int(i), str(i)), "prob": float(probs[i])}
                  for i in np.argsort(probs)[::-1][:5]]
        seña   = idx2label.get(idx, "?")
        conf   = float(probs[idx])
        modelo = "BiLSTM-S27"

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


# ── Construcción de texto ─────────────────────────────────────────────────────

def _es_clase_narrativa(seña: str) -> bool:
    """HISTORIAS_VINETAS_N es la clase 'este clip ENTERO de 1-9 minutos es la
    narrativa N' — no una seña puntual. No tiene sentido que aparezca como
    detección dentro de una ventana de 30 frames (~1s) de streaming continuo:
    ninguna persona puede 'firmar' un video ajeno en 1 segundo. Se excluye de
    la detección en vivo; el modelo sigue midiéndose con F1 sobre el clip
    completo (esa parte no cambia)."""
    return seña.startswith("HISTORIAS_VINETAS_")


def _texto_legible(seña: str) -> str:
    """Texto para mostrar en la transcripción. La mayoría de clases YA son
    palabras en castellano (BIEN, DIEZ, IGUAL...) y basta con capitalizarlas.
    Las clases HISTORIAS_VINETAS_N son clips narrativos completos, no una
    palabra — no tienen traducción 1:1, así que se muestran como marcador
    corto en vez de mangled ('Historias_vinetas_2' via .capitalize())."""
    if seña.startswith("HISTORIAS_VINETAS_"):
        n = seña.rsplit("_", 1)[-1]
        return f"[Viñeta {n}]"
    if seña in clase_texto:
        return clase_texto[seña]
    return seña.capitalize()


def build_translation_text(parts: list) -> str:
    """Construye texto desde señas acumuladas.

    Letras individuales se concatenan (deletreo manual LSP).
    Palabras/frases completas se separan con espacios.
    """
    if not parts:
        return ""
    words = []
    letter_group = []
    for part in parts:
        seña = part["seña"]
        if len(seña) == 1 and seña.isalpha():
            letter_group.append(seña.upper())
        else:
            if letter_group:
                words.append("".join(letter_group))
                letter_group = []
            words.append(_texto_legible(seña))
    if letter_group:
        words.append("".join(letter_group))
    return " ".join(words)


def build_confidence_display(parts: list) -> str:
    """Panel de confianza por seña reconocida."""
    if not parts:
        return ""
    lines = ["### Señas reconocidas\n"]
    for p in parts[-20:]:
        conf = p["confidence"] * 100
        filled = int(conf / 10)
        bar = "█" * filled + "░" * (10 - filled)
        lines.append(f"**{p['seña']}** — {conf:.0f}% `{bar}`")
    return "\n\n".join(lines)


def build_estado_md(detected: dict) -> str:
    """Estado de detección de partes del cuerpo."""
    has_body  = detected.get("cuerpo",   False)
    has_face  = detected.get("rostro",   False)
    has_lhand = detected.get("mano_izq", False)
    has_rhand = detected.get("mano_der", False)
    has_hands = has_lhand or has_rhand
    items = [
        ("Persona",       has_body or has_hands),
        ("Rostro/Cabeza", has_face),
        ("Tronco/Hombros",has_body),
        ("Brazos/Codos",  has_body),
        ("Mano izq.",     has_lhand),
        ("Mano der.",     has_rhand),
        ("Dedos",         has_hands),
    ]
    return " &nbsp;&nbsp; ".join(
        f"{'🟢' if ok else '🔴'} {label}" for label, ok in items
    )


def _build_result_md(translation_text: str, parts: list) -> str:
    """Markdown del resultado principal de traducción."""
    if not parts:
        return "**Esperando señas…**"
    last = parts[-1]
    conf = last["confidence"] * 100
    col  = "green" if conf >= 60 else "orange" if conf >= 35 else "gray"
    md   = []
    if translation_text:
        md.append(f"## TRADUCCIÓN EN TIEMPO REAL\n\n### {translation_text}")
    detalle = clase_texto.get(last["seña"])
    md.append(
        f"\n**Última seña:** `{last['seña']}`"
        + (f" — {detalle}" if detalle else "")
        + f"  <span style='color:{col}'>**{conf:.0f}%**</span>  |  "
        f"**Modelo:** {last.get('modelo','N/A')}"
    )
    return "\n\n".join(md)


# ── Panel de estado visual (webcam overlay) ───────────────────────────────────

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
        sym = "+" if ok else "-"
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
        cv2.putText(bar, f"Acumulando... {n_buf}/{N_FRAMES} frames",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1)
        cv2.rectangle(bar, (0, 102), (pct, 110), (0, 180, 255), -1)
    return np.vstack([vis, bar])


# ── Handler: cámara web ───────────────────────────────────────────────────────

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
    kp_buffer.append(kp)  # solo para el contador visual "acumulando..."

    segmento = segmentador_cam.push(kp)
    if segmento is not None:
        kp_seq = resample_to_n_frames(segmento, N_FRAMES)
        r = run_inference(kp_seq)
        if r is not None and r["confidence"] >= CONF_UMBRAL and not _es_clase_narrativa(r["seña"]):
            last_result = r
            if not historial or historial[-1] != r["seña"]:
                historial.append(r["seña"])

    vis_out = draw_status_bar(vis, detected, last_result)

    if last_result:
        r    = last_result
        conf = r["confidence"] * 100
        result_md = (
            f"## {_texto_legible(r['seña'])}\n\n"
            f"**Confianza:** {conf:.1f}%  |  **Modelo:** {r['modelo']}  |  "
            f"**Latencia:** {r['latency_ms']:.0f} ms"
        )
        top3_md = "**Top 5:**\n" + "\n".join(
            f"{i+1}. {t['clase']} — {t['prob']*100:.1f}%"
            for i, t in enumerate(r.get("top3", []))
        )
        hist_md = "**Historial:** " + " › ".join(_texto_legible(s) for s in list(historial)[-8:])
    else:
        result_md = "**Esperando seña… hacé la seña y pausá un instante al terminar**"
        top3_md   = ""
        hist_md   = ""

    estado_md = build_estado_md(detected)
    return vis_out, result_md, top3_md + "\n\n" + hist_md, estado_md


# ── Handler: video (streaming con ventana deslizante) ────────────────────────

def process_video_streaming(video_path):
    """
    Generator: procesa el video frame a frame con ventana deslizante de N_FRAMES.
    Hace yield de resultados progresivos mientras avanza el video.

    Outputs: (frame_rgb, result_md, confidence_md, estado_md, progress_md, full_text)
    """
    if video_path is None:
        yield None, "No se subió ningún video.", "", "", "", ""
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield None, "⚠️ Error: no se pudo abrir el video. Verificar formato MP4/AVI/MOV.", "", "", "", ""
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if total_frames == 0:
        yield None, "⚠️ Error: video sin frames detectados.", "", "", "", ""
        cap.release()
        return

    # Modo tracking para frames consecutivos (mejor que static_image_mode=True)
    # model_complexity=0 se probó para acelerar, pero degradaba la detección de
    # la mano derecha (3/10 → 1/10 frames en prueba real) y dejaba de mostrar
    # traducción — revertido a 1. La velocidad se gana solo con frame_stride.
    holistic_vid = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    segmentador       = SegmentadorPausas()
    translation_parts = []   # [{seña, confidence, modelo}]
    frame_count       = 0    # frames leídos del video
    frames_procesados = 0    # frames pasados al segmentador (tras submuestreo)
    # Submuestreo para procesar ~10 fps efectivos (antes 15) — más velocidad de
    # principio a fin de la demo, a costa de un poco de resolución temporal.
    frame_stride      = max(1, int(fps / 10))
    detected_any      = {k: False for k in ["mano_izq", "mano_der", "cuerpo", "rostro"]}
    last_vis_rgb      = None

    yield None, "⏳ Iniciando procesamiento del video...", "", "", "0%", ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Saltar frames para velocidad (mantiene sincronía temporal)
        if frame_count % frame_stride != 0:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = holistic_vid.process(frame_rgb)

        # Actualizar detecciones acumuladas
        detected = _make_detected(results)
        for k in detected_any:
            if detected[k]:
                detected_any[k] = True

        # Dibujar landmarks sobre frame original (BGR)
        vis = frame.copy()
        _draw_landmarks(vis, results)

        # Añadir barra de progreso sobre el frame
        h, w = vis.shape[:2]
        pct_bar = int(frame_count / max(total_frames, 1) * w)
        cv2.rectangle(vis, (0, h - 6), (pct_bar, h), (0, 200, 255), -1)

        last_vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

        # Extraer keypoints y pasarlos al segmentador por pausas
        kp = _results_to_kp(results)
        frames_procesados += 1

        # Clasificar cuando el segmentador detecta el fin de una seña (pausa
        # de las manos), no cada N frames fijos — ver src/features/segmentacion.py
        new_sign = False
        segmento = segmentador.push(kp)
        if segmento is not None:
            kp_seq = resample_to_n_frames(segmento, N_FRAMES)
            r = run_inference(kp_seq)
            if r and r["confidence"] >= CONF_UMBRAL and not _es_clase_narrativa(r["seña"]):
                translation_parts.append({
                    "seña":       r["seña"],
                    "confidence": r["confidence"],
                    "modelo":     r.get("modelo", "LSTM"),
                })
                new_sign = True

        # Yield al detectar seña nueva o cada STRIDE frames (actualización de progreso)
        if new_sign or (frames_procesados % STRIDE == 0):
            progress_pct      = min(frame_count / max(total_frames, 1) * 100, 99)
            translation_text  = build_translation_text(translation_parts)
            result_md         = _build_result_md(translation_text, translation_parts)
            confidence_md     = build_confidence_display(translation_parts)
            estado_md         = build_estado_md(detected_any)
            progress_md       = (
                f"⏳ {progress_pct:.0f}% — frame {frame_count}/{total_frames} "
                f"| Señas detectadas: {len(translation_parts)}"
            )
            yield (last_vis_rgb, result_md, confidence_md, estado_md, progress_md, translation_text)

    holistic_vid.close()
    cap.release()

    # El video puede terminar a mitad de una seña (sin pausa final que la
    # cierre) — típico en clips ya recortados de una sola seña. El segmentador
    # entrega lo que quedó acumulado si alcanza el mínimo de frames.
    segmento_final = segmentador.flush()
    if segmento_final is not None:
        kp_seq = resample_to_n_frames(segmento_final, N_FRAMES)
        r = run_inference(kp_seq)
        if r and r["confidence"] >= CONF_UMBRAL and not _es_clase_narrativa(r["seña"]):
            translation_parts.append({
                "seña":       r["seña"],
                "confidence": r["confidence"],
                "modelo":     r.get("modelo", "LSTM"),
            })

    # Resultado final
    translation_text = build_translation_text(translation_parts)
    n_señas = len(translation_parts)

    if n_señas == 0:
        result_md = (
            "⚠️ No se reconocieron señas LSP en este video.\n\n"
            "**Posibles causas:**\n"
            "- El video no contiene señas visibles o la persona no aparece claramente\n"
            "- La confianza del modelo no superó el umbral del 30%\n"
            "- El video es muy corto (necesita al menos 2 segundos)\n\n"
            f"**Umbral de confianza:** {CONF_UMBRAL*100:.0f}%  |  "
            f"**Frames procesados:** {frame_count}"
        )
    else:
        result_md = _build_result_md(translation_text, translation_parts)

    confidence_md = build_confidence_display(translation_parts)
    estado_md     = build_estado_md(detected_any)
    progress_md   = (
        f"✅ Procesamiento completado — "
        f"{n_señas} señas detectadas en {frame_count} frames"
    )

    yield (last_vis_rgb, result_md, confidence_md, estado_md, progress_md, translation_text)


# ── Handler: exportar TXT ─────────────────────────────────────────────────────

def export_translation_txt(text: str):
    """Genera archivo TXT con la traducción para descarga."""
    if not text or not text.strip():
        return None
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", prefix="traduccion_lsp_",
        delete=False, encoding="utf-8"
    )
    tmp.write("TRADUCCION LSP → CASTELLANO\n")
    tmp.write("=" * 40 + "\n\n")
    tmp.write(text.strip())
    tmp.write("\n\n" + "=" * 40 + "\n")
    tmp.write(f"Generado por: Traductor LSP  |  {time.strftime('%Y-%m-%d %H:%M')}\n")
    tmp.close()
    return tmp.name


# ── Handler: imagen estática ──────────────────────────────────────────────────

def process_image(image):
    """Imagen estática → landmarks + inferencia (keypoints replicados N_FRAMES veces)."""
    if image is None:
        return None, "No se subió ninguna imagen.", "", ""

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 4:
        image = image[:, :, :3]

    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).clip(0, 255)
        image = image.astype(np.uint8)

    h, w = image.shape[:2]
    if h < 64 or w < 64:
        return None, "Imagen demasiado pequeña.", "", ""
    if max(h, w) > 1920:
        scale = 1920 / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    frame_rgb = np.ascontiguousarray(image)

    holistic_img = mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.1,
        min_tracking_confidence=0.1,
    )
    results = holistic_img.process(frame_rgb)
    holistic_img.close()

    detected = _make_detected(results)
    vis = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    _draw_landmarks(vis, results)
    kp = _results_to_kp(results)

    if not any(detected.values()):
        vis_rgb   = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        estado_md = build_estado_md(detected)
        return vis_rgb, "**No se detectó ninguna persona en la imagen.**", "", estado_md

    kp_seq = np.stack([kp] * N_FRAMES)
    result = run_inference(kp_seq)

    vis_rgb   = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    estado_md = build_estado_md(detected)

    if result is None:
        return vis_rgb, "Modelo no disponible.", "", estado_md

    conf = result["confidence"] * 100
    result_md = (
        f"## {result['seña']}\n\n"
        f"**Confianza:** {conf:.1f}%  |  **Modelo:** {result['modelo']}  |  "
        f"**Latencia:** {result['latency_ms']:.0f} ms"
    )
    top3_md = "**Top 5:**\n" + "\n".join(
        f"{i+1}. {t['clase']} — {t['prob']*100:.1f}%"
        for i, t in enumerate(result.get("top3", []))
    )
    return vis_rgb, result_md, top3_md, estado_md


# ── Interfaz Gradio ───────────────────────────────────────────────────────────

CSS = """
.traduccion    { font-size: 1.6em !important; padding: 16px 20px;
                 border-left: 5px solid #2196F3; background: #f0f7ff; }
.estado        { font-size: 1.0em; padding: 8px 12px; }
.transcripcion { font-size: 1.2em; padding: 12px 16px;
                 border: 2px solid #4CAF50; background: #f0fff0;
                 font-family: monospace; min-height: 80px; }
"""

# TTS 100% navegador (Web Speech API) — sin backend, sin costo.
# %s se reemplaza por el selector CSS del elemento con el texto a leer.
TTS_JS = """
() => {
  const el = document.querySelector('%s');
  const text = el ? (el.innerText || el.textContent) : '';
  if (!text || !text.trim()) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text.trim());
  u.lang = 'es-PE';
  window.speechSynthesis.speak(u);
}
"""

TTS_JS_TEXTAREA = """
() => {
  const el = document.querySelector('%s');
  const text = el ? el.value : '';
  if (!text || !text.trim()) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text.trim());
  u.lang = 'es-PE';
  window.speechSynthesis.speak(u);
}
"""

with gr.Blocks(title="Traductor LSP → Castellano", css=CSS) as demo:

    gr.Markdown(f"""
# 🤟 Traductor LSP → Castellano — Sprint 27
**Sistema integral de comunicación inclusiva** — Lengua de Señas Peruana a texto en tiempo real.
MediaPipe Holistic detecta pose + ambas manos + rostro (150 dims/frame).
**BiLSTM S27** clasifica la seña en **{len(idx2label)}** clases LSP | F1=0.4349 | Top-5=63.6%

> ⚠️ **Alcance actual:** reconoce **una seña aislada a la vez** (con una breve pausa entre cada una) —
> igual que fue entrenado y medido. **No** está diseñado para traducir narración continua palabra por
> palabra (p. ej. una historia contada de corrido); eso requiere un modelo de reconocimiento continuo
> que todavía no existe en el proyecto.
    """)

    with gr.Tabs():

        # ── Tab 1: Cámara en vivo ─────────────────────────────────────────
        with gr.TabItem("📷 Cámara en vivo"):
            with gr.Row():
                with gr.Column(scale=3):
                    webcam_in  = gr.Image(sources=["webcam"], streaming=True,
                                          label="Cámara", height=360)
                    webcam_out = gr.Image(label="Landmarks detectados", height=420)
                with gr.Column(scale=2):
                    estado_cam = gr.Markdown("", elem_classes=["estado"])
                    result_cam = gr.Markdown("**Esperando señas…**",
                                             elem_id="result_cam",
                                             elem_classes=["traduccion"])
                    tts_btn_cam = gr.Button("🔊 Leer en voz alta", size="sm")
                    top3_cam   = gr.Markdown("")

            tts_btn_cam.click(fn=None, inputs=None, outputs=None, js=TTS_JS % "#result_cam")

            webcam_in.stream(
                fn=process_webcam_frame,
                inputs=[webcam_in],
                outputs=[webcam_out, result_cam, top3_cam, estado_cam],
                time_limit=300,
                stream_every=0.10,
            )

        # ── Tab 2: Subir video ────────────────────────────────────────────
        with gr.TabItem("🎬 Subir video"):
            gr.Markdown(
                "> El sistema detecta el **fin de cada seña por la pausa de las manos** "
                "(no una ventana fija) y clasifica cada segmento por separado — igual que "
                "el modelo fue entrenado. Cada seña reconocida se añade a la transcripción.\n\n"
                "> ⚠️ **Funciona con señante que hace pausas breves entre señas.** "
                "Videos de narración fluida sin pausas (una historia completa firmada de corrido) "
                "**no van a segmentarse ni traducirse bien** — eso es reconocimiento continuo "
                "gloss-por-gloss, un problema de investigación aparte, no resuelto en este sistema."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    video_in = gr.Video(label="Video MP4 / AVI / MOV con señas LSP")
                    btn_vid  = gr.Button("▶ Traducir señas en tiempo real",
                                         variant="primary", size="lg")

                with gr.Column(scale=3):
                    vid_frame_out = gr.Image(label="Frame actual con landmarks detectados",
                                             height=280)
                    estado_vid    = gr.Markdown("", elem_classes=["estado"])
                    progress_vid  = gr.Markdown("")
                    result_vid    = gr.Markdown("", elem_classes=["traduccion"])

            with gr.Row():
                confidence_vid = gr.Markdown("")

            gr.Markdown("### 📋 Transcripción completa en tiempo real")
            with gr.Row():
                export_text_vid = gr.Textbox(
                    label="TRADUCCIÓN EN TIEMPO REAL",
                    placeholder="La traducción aparecerá aquí mientras se procesa el video…",
                    lines=4,
                    max_lines=12,
                    elem_id="export_text_vid",
                    elem_classes=["transcripcion"],
                    interactive=False,
                )
            with gr.Row():
                btn_export = gr.Button("💾 Exportar a TXT", variant="secondary", size="sm")
                btn_copy   = gr.Button("📋 Copiar texto", variant="secondary", size="sm")
                tts_btn_vid = gr.Button("🔊 Leer transcripción", variant="secondary", size="sm")
                file_out   = gr.File(label="Descargar traduccion_lsp.txt")

            tts_btn_vid.click(fn=None, inputs=None, outputs=None,
                              js=TTS_JS_TEXTAREA % "#export_text_vid textarea")

            btn_vid.click(
                fn=process_video_streaming,
                inputs=[video_in],
                outputs=[vid_frame_out, result_vid, confidence_vid,
                         estado_vid, progress_vid, export_text_vid],
            )

            btn_export.click(
                fn=export_translation_txt,
                inputs=[export_text_vid],
                outputs=[file_out],
            )

        # ── Tab 3: Imagen estática ────────────────────────────────────────
        with gr.TabItem("🖼️ Subir imagen"):
            with gr.Row():
                with gr.Column():
                    image_in = gr.Image(label="Imagen JPG/PNG con seña LSP",
                                        type="numpy", height=360)
                    btn_img  = gr.Button("🔍 Detectar y traducir",
                                         variant="primary", size="lg")
                with gr.Column():
                    estado_img = gr.Markdown("", elem_classes=["estado"])
                    result_img = gr.Markdown("", elem_classes=["traduccion"])
                    top3_img   = gr.Markdown("")
            image_out = gr.Image(label="Landmarks detectados", height=360)

            btn_img.click(
                fn=process_image,
                inputs=[image_in],
                outputs=[image_out, result_img, top3_img, estado_img],
            )

        # ── Tab 4: Pipeline ───────────────────────────────────────────────
        with gr.TabItem("ℹ️ Pipeline"):
            gr.Markdown(f"""
## Pipeline técnico

```
Cámara / Video MP4·AVI·MOV
     ↓
MediaPipe Holistic (frame a frame, modo tracking)
     ├── 33 keypoints pose/cuerpo    (azul)   → hombros, brazos, codos, muñecas, tronco
     ├── 21 keypoints mano izquierda (rojo)   → mano + 5 dedos
     ├── 21 keypoints mano derecha   (verde)  → mano + 5 dedos
     └── 468 keypoints rostro        (gris)   → cabeza, expresión facial
     ↓
Feature extraction: 150 dims/frame
     pose_x(33) + pose_y(33) + left_x(21) + left_y(21) + right_x(21) + right_y(21)
     ↓
Buffer deslizante [{N_FRAMES} frames × 150 dims], stride={STRIDE} frames (overlap 50%)
     ↓
BiLSTM S27: proj(150→128) → LayerNorm → BiLSTM(128,256) → TemporalAttention → head(512→{len(idx2label)})
     ↓
Construcción de texto:
     Letras individuales → se concatenan (H+O+L+A → "HOLA")
     Palabras/frases → se unen con espacios
     ↓
Transcripción en tiempo real + Top-5 candidatos + Confianza + Exportar TXT
```

## Ventana deslizante

| Parámetro | Valor |
|-----------|-------|
| Tamaño ventana | {N_FRAMES} frames |
| Stride (overlap 50%) | {STRIDE} frames |
| FPS efectivos procesados | ~15 fps |
| Umbral de confianza | {CONF_UMBRAL*100:.0f}% |

## Dataset S17 de entrenamiento (fix cross-source dgi156↔vineta + sub-grupos ampliados)

| Fuente | Muestras |
|--------|---------|
| vineta (Historias viñetas) | 3,684 |
| dgi156 (múltiples señantes) | 3,642 |
| abecedario (10 sub-grupos) | 3,600 |
| AEC (intérprete TV, 10 sub-grupos) | 1,102 |
| vocabulario_lsp_p | 80 |
| glosa | 42 |
| **Total (≥15 muestras/clase)** | **12,150** en **96 clases** |

## Modelo BiLSTM S27

| Parámetro | Valor |
|-----------|-------|
| Arquitectura | proj(150→128) → LayerNorm → BiLSTM(128,256,1capa) → TemporalAttention |
| Parámetros | ~966K |
| hidden=256, dropout=0.20, lr=1.71e-3 | |
| F1-macro test | **0.4349** (mejor punto histórico del proyecto) |
| Top-3 accuracy | **56.1%** |
| Top-5 accuracy | **63.6%** |
| ΔF1 (HE3, generalización) | **0.0509** (umbral ≤0.15) |
| Latencia ONNX | **0.72 ms** |
            """)


if __name__ == "__main__":
    import sys
    share = "--share" in sys.argv
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        theme=gr.themes.Soft(),
        css=CSS,
        share=share,
    )
