"""
extract_abecedario_keypoints.py — Extrae keypoints MediaPipe de los 3600 JPG del Abecedario.
Señas estáticas del alfabeto dactilológico LSP (a–y, sin j/z que son dinámicas).
Salida: data/Keypoints/abecedario_pkl/<LETRA>/<LETRA>_NNN.pkl
"""

import cv2, pickle, pathlib, warnings
import numpy as np
import mediapipe as mp
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT    = pathlib.Path(__file__).parent.parent
ABC_DIR = ROOT / "data" / "Abecedario"
OUT_DIR = ROOT / "data" / "Keypoints" / "abecedario_pkl"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_FRAMES = 30   # repetir el mismo frame N veces → señas estáticas

mp_hands = mp.solutions.hands
mp_pose  = mp.solutions.pose


def extract_from_image(img_path):
    """
    Extrae landmarks de mano + pose (estimada) de una imagen JPG.
    Devuelve lista de N_FRAMES frames idénticos en formato PKL.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    frame_data = {
        "pose":       {"x": [0.0] * 33, "y": [0.0] * 33},
        "left_hand":  {"x": [0.0] * 21, "y": [0.0] * 21},
        "right_hand": {"x": [0.0] * 21, "y": [0.0] * 21},
    }

    # Detectar manos
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.3,
    ) as hands:
        res = hands.process(img_rgb)
        if res.multi_hand_landmarks and res.multi_handedness:
            for lm_list, handedness in zip(res.multi_hand_landmarks,
                                            res.multi_handedness):
                label = handedness.classification[0].label  # "Left" o "Right"
                xs = [lm.x for lm in lm_list.landmark]
                ys = [lm.y for lm in lm_list.landmark]
                if label == "Right":
                    frame_data["right_hand"] = {"x": xs, "y": ys}
                else:
                    frame_data["left_hand"]  = {"x": xs, "y": ys}

    # Señas estáticas: repetir N_FRAMES veces
    return [frame_data] * N_FRAMES


# ── Procesar todas las imágenes ───────────────────────────────────────────────

jpg_files = sorted(ABC_DIR.rglob("*.jpg"))
print(f"JPG encontrados: {len(jpg_files)}")

ok, err = 0, 0
for jpg_path in tqdm(jpg_files, desc="Abecedario"):
    letra = jpg_path.parent.name.upper()
    stem  = jpg_path.stem.replace(" ", "_")

    out_pkl = OUT_DIR / letra / f"{letra}_{stem}.pkl"
    if out_pkl.exists():
        ok += 1
        continue

    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    frames = extract_from_image(jpg_path)
    if frames is None:
        err += 1
        continue

    with open(out_pkl, "wb") as f:
        pickle.dump(frames, f)
    ok += 1

total = sum(1 for _ in OUT_DIR.rglob("*.pkl"))
print(f"\n✅ OK={ok}  ERR={err}")
print(f"PKL guardados en {OUT_DIR}: {total}")
