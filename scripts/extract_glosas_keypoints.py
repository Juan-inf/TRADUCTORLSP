"""
extract_glosas_keypoints.py — Extrae keypoints MediaPipe de los 526 MP4 de Glosas.
Usa los archivos EAF para recortar exactamente los frames de la seña (sin pre/post-roll).
Salida: data/Keypoints/glosas_pkl/<CLASE>/<NOMBRE>.pkl
"""

import cv2, pickle, pathlib, xml.etree.ElementTree as ET, warnings
import numpy as np
import mediapipe as mp
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT       = pathlib.Path(__file__).parent.parent
GLOSAS_DIR = ROOT / "data" / "Glosas"
OUT_DIR    = ROOT / "data" / "Keypoints" / "glosas_pkl"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_FRAMES   = 30    # secuencia objetivo (resample/pad)

# ── MediaPipe ─────────────────────────────────────────────────────────────────

mp_holistic = mp.solutions.holistic

def parse_eaf_timing(eaf_path):
    """Lee el EAF y devuelve (start_ms, end_ms) de la anotación GLOSA principal."""
    try:
        tree = ET.parse(eaf_path)
        root = tree.getroot()

        # Mapa TIME_SLOT_ID → TIME_VALUE
        ts_map = {}
        for ts in root.findall(".//TIME_SLOT"):
            ts_map[ts.get("TIME_SLOT_ID")] = int(ts.get("TIME_VALUE", 0))

        # Buscar primera anotación con tiempo válido
        for ann in root.findall(".//ALIGNABLE_ANNOTATION"):
            ts1 = ann.get("TIME_SLOT_REF1", "")
            ts2 = ann.get("TIME_SLOT_REF2", "")
            if ts1 in ts_map and ts2 in ts_map:
                start = ts_map[ts1]
                end   = ts_map[ts2]
                if end > start:
                    return start, end
    except Exception:
        pass
    return None, None  # fallback: usar video completo


def extract_keypoints_from_video(mp4_path, start_ms=None, end_ms=None):
    """
    Extrae keypoints MediaPipe Holistic del segmento [start_ms, end_ms].
    Devuelve lista de dicts {pose, left_hand, right_hand} o None si falla.
    """
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return None

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Índices de frames del segmento
    if start_ms is not None and end_ms is not None:
        f_start = max(0, int(start_ms / 1000 * fps))
        f_end   = min(total - 1, int(end_ms   / 1000 * fps))
    else:
        f_start, f_end = 0, total - 1

    if f_end <= f_start:
        f_start, f_end = 0, total - 1

    # Resamplear a N_FRAMES índices dentro del segmento
    n_seg   = f_end - f_start + 1
    indices = set(int(i) for i in np.linspace(f_start, f_end,
                                               min(N_FRAMES, n_seg), dtype=int))

    frames_kp = []
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as holistic:
        fi = 0
        while cap.isOpened() and fi <= f_end:
            ret, frame = cap.read()
            if not ret:
                break
            if fi in indices:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = holistic.process(rgb)

                def lm_to_xy(lm_list, n):
                    if lm_list:
                        return ([l.x for l in lm_list.landmark],
                                [l.y for l in lm_list.landmark])
                    return [0.0] * n, [0.0] * n

                px, py = lm_to_xy(res.pose_landmarks,       33)
                lx, ly = lm_to_xy(res.left_hand_landmarks,  21)
                rx, ry = lm_to_xy(res.right_hand_landmarks, 21)

                frames_kp.append({
                    "pose":       {"x": px, "y": py},
                    "left_hand":  {"x": lx, "y": ly},
                    "right_hand": {"x": rx, "y": ry},
                })
            fi += 1

    cap.release()

    if not frames_kp:
        return None

    # Pad hasta N_FRAMES si hay menos
    while len(frames_kp) < N_FRAMES:
        frames_kp.append(frames_kp[-1])

    return frames_kp[:N_FRAMES]


# ── Procesar todos los MP4 ────────────────────────────────────────────────────

mp4_files = sorted(GLOSAS_DIR.rglob("*.mp4"))
print(f"MP4 encontrados: {len(mp4_files)}")

ok, skip, err = 0, 0, 0
for mp4_path in tqdm(mp4_files, desc="Extrayendo Glosas"):
    clase = mp4_path.parent.name
    stem  = mp4_path.stem

    # Saltar ORACION (frases completas, no señas individuales)
    if "ORACION" in stem.upper():
        skip += 1
        continue

    out_pkl = OUT_DIR / clase / f"{stem}.pkl"
    if out_pkl.exists():
        ok += 1
        continue

    out_pkl.parent.mkdir(parents=True, exist_ok=True)

    # Leer timing del EAF
    eaf_path = mp4_path.with_suffix(".eaf")
    start_ms, end_ms = parse_eaf_timing(eaf_path) if eaf_path.exists() else (None, None)

    frames_kp = extract_keypoints_from_video(mp4_path, start_ms, end_ms)
    if frames_kp is None:
        err += 1
        continue

    with open(out_pkl, "wb") as f:
        pickle.dump(frames_kp, f)
    ok += 1

total_glosas = sum(1 for _ in (OUT_DIR).rglob("*.pkl"))
print(f"\n✅ OK={ok}  SKIP(oraciones)={skip}  ERR={err}")
print(f"PKL guardados en {OUT_DIR}: {total_glosas}")
