"""build_dataset_s34_eaf.py

Construye un dataset NUEVO a partir de un recurso hasta ahora sin usar:
los 274 archivos data/Glosas/*/*_ORACION_*.eaf (formato ELAN) — oraciones
completas de LSP con la glosa de CADA seña anotada a mano con timestamp en
milisegundos (tier "GLOSA"), más una traducción natural al castellano
(tier "Traducción") para 119 de ellas. A diferencia del SRT usado en
build_dataset_s31_continuo.py (transcripción del audio hablado), esta
anotación es lingüística: hecha directamente sobre la seña, no inferida
del castellano oral.

scripts/extract_glosas_keypoints.py ya parsea estos .eaf pero SOLO para
recortar el inicio/fin de clips de seña individual, y descarta
explícitamente los "_ORACION_" ("frases completas, no señas individuales").
Este script hace lo contrario: usa exactamente esos 274 archivos,
extrayendo keypoints MediaPipe Holistic por cada segmento de glosa dentro
de la oración completa (multi-seña), igual que build_dataset_s31_continuo
hace con SRT x landmarks precomputados — pero aquí la extracción de
keypoints es directa desde el mp4 (no hay landmarks precomputados para
estos videos).

Split: a nivel de VIDEO/ORACIÓN completa (no de segmento) — se reservan
oraciones completas nunca vistas en entrenamiento para evaluación WER real
(scripts/evaluar_wer_s34.py), reconstruyendo la secuencia de glosas
predichas sobre la oración entera y comparándola con la anotación GLOSA
real.

Uso:
    python3 scripts/build_dataset_s34_eaf.py [--min-muestras N]
"""
import argparse
import collections
import json
import pathlib
import unicodedata
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import mediapipe as mp

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
GLOSAS_DIR = DATA / "Glosas"

import sys
sys.path.insert(0, str(ROOT))
from src.features.landmarks import kp_seq_to_features, normalize_sample, resample_to_n_frames

SEED = 42
N_FRAMES = 30
N_ORACIONES_WER_HOLDOUT = 10  # oraciones completas reservadas solo para evaluación WER

mp_holistic = mp.solutions.holistic


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def parse_eaf_glosa(eaf_path: pathlib.Path):
    """Devuelve [(start_ms, end_ms, gloss), ...] ordenado por tiempo, tier GLOSA."""
    tree = ET.parse(eaf_path)
    root = tree.getroot()
    ts_map = {ts.get("TIME_SLOT_ID"): int(ts.get("TIME_VALUE", 0))
              for ts in root.findall(".//TIME_SLOT")}
    out = []
    for tier in root.findall(".//TIER"):
        if tier.get("TIER_ID") != "GLOSA":
            continue
        for ann in tier.findall(".//ALIGNABLE_ANNOTATION"):
            ts1, ts2 = ann.get("TIME_SLOT_REF1"), ann.get("TIME_SLOT_REF2")
            val = ann.find("ANNOTATION_VALUE")
            if ts1 not in ts_map or ts2 not in ts_map or val is None or not val.text:
                continue
            start, end = ts_map[ts1], ts_map[ts2]
            if end <= start:
                continue
            gloss = nfc(val.text.strip().upper())
            if gloss:
                out.append((start, end, gloss))
    return sorted(out, key=lambda t: t[0])


def extraer_keypoints_segmento(cap, fps, start_ms, end_ms, holistic):
    """Extrae kp [T,75,3] (left,right,pose) de los frames en [start_ms,end_ms]."""
    f_start = max(0, int(start_ms / 1000 * fps))
    f_end = int(end_ms / 1000 * fps)
    if f_end <= f_start:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
    frames_kp = []
    for fi in range(f_start, f_end + 1):
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = holistic.process(rgb)
        kp = np.zeros((75, 3), dtype=np.float32)
        if res.left_hand_landmarks:
            for i, lm in enumerate(res.left_hand_landmarks.landmark):
                kp[i] = [lm.x, lm.y, lm.z]
        if res.right_hand_landmarks:
            for i, lm in enumerate(res.right_hand_landmarks.landmark):
                kp[21 + i] = [lm.x, lm.y, lm.z]
        if res.pose_landmarks:
            for i, lm in enumerate(res.pose_landmarks.landmark):
                kp[42 + i] = [lm.x, lm.y, lm.z]
        frames_kp.append(kp)
    if not frames_kp:
        return None
    return resample_to_n_frames(frames_kp, N_FRAMES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-muestras", type=int, default=5)
    args = ap.parse_args()

    print("=" * 65)
    print("build_dataset_s34_eaf — oraciones ELAN (glosa x milisegundo) x MediaPipe")
    print("=" * 65)

    eaf_files = sorted(GLOSAS_DIR.rglob("*_ORACION_*.eaf"))
    pares = []
    for eaf in eaf_files:
        mp4 = eaf.with_suffix(".mp4")
        if mp4.exists():
            pares.append((eaf, mp4))
    print(f"Archivos ORACION con .eaf + .mp4: {len(pares)}")

    rng = np.random.RandomState(SEED)
    idx_holdout = rng.choice(len(pares), size=min(N_ORACIONES_WER_HOLDOUT, len(pares)), replace=False)
    holdout_set = set(idx_holdout.tolist())
    print(f"Oraciones reservadas 100% para evaluación WER (nunca en train): {len(holdout_set)}")

    holdout_info = []
    Xs, Ls, Gs = [], [], []
    sin_video, sin_glosa = 0, 0

    for i, (eaf, mp4) in enumerate(pares):
        entradas = parse_eaf_glosa(eaf)
        if not entradas:
            sin_glosa += 1
            continue

        if i in holdout_set:
            holdout_info.append({
                "eaf": str(eaf.relative_to(ROOT)),
                "mp4": str(mp4.relative_to(ROOT)),
                "glosas": [[s, e, g] for s, e, g in entradas],
            })
            continue

        cap = cv2.VideoCapture(str(mp4))
        if not cap.isOpened():
            sin_video += 1
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        with mp_holistic.Holistic(static_image_mode=False, model_complexity=1,
                                   min_detection_confidence=0.3,
                                   min_tracking_confidence=0.3) as holistic:
            for start_ms, end_ms, gloss in entradas:
                kp_seq = extraer_keypoints_segmento(cap, fps, start_ms, end_ms, holistic)
                if kp_seq is None:
                    continue
                feat = normalize_sample(kp_seq_to_features(kp_seq))
                Xs.append(feat)
                Ls.append(gloss)
                Gs.append(20000 + i)
        cap.release()
        if (i + 1) % 40 == 0:
            print(f"  ... procesadas {i + 1}/{len(pares)} oraciones, {len(Xs)} segmentos hasta ahora")

    print(f"\nOraciones sin video legible: {sin_video}")
    print(f"Oraciones sin glosas parseadas: {sin_glosa}")
    print(f"Segmentos de glosa extraídos (train pool): {len(Xs)}  |  glosas únicas (antes de filtro): {len(set(Ls))}")

    X_all = np.stack(Xs).astype(np.float32)
    L_all = np.array(Ls)
    G_all = np.array(Gs, dtype=np.int32)

    cnts = collections.Counter(L_all.tolist())
    mask = np.array([cnts[l] >= args.min_muestras for l in L_all])
    X_all, L_all, G_all = X_all[mask], L_all[mask], G_all[mask]
    print(f"Después de min{args.min_muestras}: {len(X_all)} muestras, {len(set(L_all.tolist()))} clases")

    print("\nDistribución (top 20 por frecuencia):")
    for gloss, n in collections.Counter(L_all.tolist()).most_common(20):
        print(f"  {gloss:<20}: {n}")

    classes = sorted(set(L_all.tolist()))
    label2idx = {c: i for i, c in enumerate(classes)}
    idx2label = {i: c for c, i in label2idx.items()}
    y_all = np.array([label2idx[l] for l in L_all.tolist()], dtype=np.int64)

    np.savez_compressed(DATA / "dataset_s34.npz", X=X_all, y=y_all, groups=G_all)
    with open(DATA / "s34_label2idx.json", "w", encoding="utf-8") as f:
        json.dump(label2idx, f, ensure_ascii=False, indent=2)
    with open(DATA / "s34_idx2label.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in idx2label.items()}, f, ensure_ascii=False, indent=2)
    with open(DATA / "s34_oraciones_wer_holdout.json", "w", encoding="utf-8") as f:
        json.dump({"holdout": holdout_info}, f, ensure_ascii=False, indent=2)

    print(f"\n✅ dataset_s34.npz  ({len(X_all)} muestras, {len(classes)} clases)")
    print(f"✅ s34_label2idx.json / s34_idx2label.json")
    print(f"✅ s34_oraciones_wer_holdout.json — {len(holdout_info)} oraciones reservadas para WER")


if __name__ == "__main__":
    main()
