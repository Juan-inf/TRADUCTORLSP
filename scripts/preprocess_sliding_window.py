"""
Preprocesamiento con Sliding Window para dataset LSP continuo.
NO guarda pixels (demasiado grande ~125GB). Genera:
  - data/manifest_segments.csv: metadata de segmentos (inicio/fin frames)
  - data/landmarks/: archivos .npy de keypoints (pequeños: ~27KB/segmento)
  - Splits train/val/test a nivel de video (evita data leakage)

Uso: python scripts/preprocess_sliding_window.py [--max_videos N] [--no_landmarks]
     .venv310/bin/python scripts/preprocess_sliding_window.py
"""
import argparse, cv2, json, os, re, sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

DATA_DIR   = Path("data")
LAND_DIR   = DATA_DIR / "landmarks"
N_FRAMES   = 30
STRIDE     = 15
N_KP       = 75  # 42 manos + 33 pose
IMG_SIZE   = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def extract_landmarks_video(video_path: str, n_frames: int = N_FRAMES,
                             stride: int = STRIDE):
    """Extrae landmarks de todos los segmentos de un video."""
    try:
        import mediapipe as mp
        mp_holistic = mp.solutions.holistic
    except ImportError:
        return []

    holistic = mp_holistic.Holistic(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.4, min_tracking_confidence=0.4
    )
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_kp = []
    ok, frame = cap.read()
    while ok:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = holistic.process(rgb)
        kp = np.zeros((N_KP, 3), dtype=np.float32)
        if res.left_hand_landmarks:
            for i, lm in enumerate(res.left_hand_landmarks.landmark):
                kp[i] = [lm.x, lm.y, lm.z]
        if res.right_hand_landmarks:
            for i, lm in enumerate(res.right_hand_landmarks.landmark):
                kp[21 + i] = [lm.x, lm.y, lm.z]
        if res.pose_landmarks:
            for i, lm in enumerate(res.pose_landmarks.landmark):
                kp[42 + i] = [lm.x, lm.y, lm.z]
        all_kp.append(kp)
        ok, frame = cap.read()

    cap.release()
    holistic.close()

    if len(all_kp) < n_frames:
        return []

    segments = []
    for start in range(0, len(all_kp) - n_frames + 1, stride):
        seg = np.stack(all_kp[start:start + n_frames])  # [T, N, 3]
        ref = seg[:, 21:22, :]
        seg = seg - ref  # normalizar a muñeca derecha
        segments.append((start, seg))
    return segments


def build_segment_manifest(df_valid, n_frames=N_FRAMES, stride=STRIDE):
    """Construye manifest de segmentos sin cargar frames."""
    rows = []
    for _, row in df_valid.iterrows():
        total = int(row['n_frames'])
        for start in range(0, total - n_frames + 1, stride):
            rows.append({
                'video_path':  row['ruta'],
                'clase':       row['clase'],
                'num_vineta':  int(row['num_vineta']),
                'start_frame': start,
                'end_frame':   start + n_frames,
                'kp_path':     '',
            })
    return pd.DataFrame(rows)


def make_video_splits(df_valid, train_r=0.70, val_r=0.15, seed=42):
    """Splits a nivel de VIDEO (no mezclar frames del mismo video)."""
    nums = df_valid['num_vineta'].tolist()
    train_nums, tmp = train_test_split(nums, test_size=1-train_r, random_state=seed)
    val_nums, test_nums = train_test_split(tmp, test_size=0.5, random_state=seed)
    return set(train_nums), set(val_nums), set(test_nums)


def assign_temporal_splits(df_segs, train_r=0.70, val_r=0.15):
    """
    Splits TEMPORALES dentro de cada video (primeros 70% → train, etc.).
    Garantiza que cada clase aparezca en los tres splits.
    Tradeoff: autocorrelación temporal entre train y test, pero es la
    única opción viable cuando hay 1 video por clase.
    """
    splits = []
    for _, group in df_segs.groupby('num_vineta', sort=True):
        n = len(group)
        i_train = int(n * train_r)
        i_val   = int(n * (train_r + val_r))
        s = ['train'] * i_train + ['val'] * (i_val - i_train) + ['test'] * (n - i_val)
        splits.extend(s)
    return splits


def main(args):
    LAND_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_DIR / "manifest.csv")
    df_valid = df[df['valido']].sort_values('num_vineta').reset_index(drop=True)

    if args.max_videos:
        df_valid = df_valid.head(args.max_videos)
        print(f"Modo debug: {args.max_videos} videos")

    # ── 1. Construir manifest de segmentos (solo metadata) ────────────────
    print("\n1. Construyendo manifest de segmentos...")
    df_segs = build_segment_manifest(df_valid)
    total_segs = len(df_segs)
    print(f"   Segmentos totales: {total_segs:,}")
    print(f"   Distribución por clase:")
    print(df_segs['clase'].value_counts().to_string())

    # ── 2. Splits ─────────────────────────────────────────────────────────
    if args.temporal_splits:
        print("\n2. Generando splits TEMPORALES por video (train/val/test dentro de cada video)...")
        print("   Garantiza que cada clase aparezca en los tres splits.")
        df_segs['split'] = assign_temporal_splits(df_segs)
    else:
        print("\n2. Generando splits a nivel de video...")
        train_v, val_v, test_v = make_video_splits(df_valid)
        df_segs['split'] = df_segs['num_vineta'].apply(
            lambda n: 'train' if n in train_v else ('val' if n in val_v else 'test')
        )
    counts = df_segs['split'].value_counts()
    n_per_split = df_segs.groupby('split')['num_vineta'].nunique()
    print(f"   Train: {counts.get('train',0):,} segs ({n_per_split.get('train',0)} videos con esa etiqueta)")
    print(f"   Val:   {counts.get('val',0):,} segs ({n_per_split.get('val',0)} videos con esa etiqueta)")
    print(f"   Test:  {counts.get('test',0):,} segs ({n_per_split.get('test',0)} videos con esa etiqueta)")

    # ── 3. Extraer landmarks (pequeños, ~27KB/segmento) ───────────────────
    if not args.no_landmarks:
        print("\n3. Extrayendo landmarks MediaPipe...")
        print("   Tiempo estimado: ~2-5 min/video en CPU")

        for _, row in tqdm(df_valid.iterrows(), total=len(df_valid), desc="Videos"):
            vp = row['ruta']
            clase = row['clase']
            num = int(row['num_vineta'])
            print(f"\n   Vineta {num} ({row['duracion_seg']:.0f}s, {int(row['n_frames'])} frames)...")

            seg_list = extract_landmarks_video(vp, N_FRAMES, args.stride)
            print(f"   → {len(seg_list)} segmentos landmarks")

            for start, kp_seg in seg_list:
                kp_path = LAND_DIR / f"{clase}_s{start:06d}_kp.npy"
                np.save(str(kp_path), kp_seg)
                # Actualizar manifest
                mask = (df_segs['clase'] == clase) & (df_segs['start_frame'] == start)
                df_segs.loc[mask, 'kp_path'] = str(kp_path)
    else:
        print("\n3. Landmarks omitidos (--no_landmarks)")

    # ── 4. Guardar manifest ───────────────────────────────────────────────
    out_path = DATA_DIR / "manifest_segments.csv"
    df_segs.to_csv(out_path, index=False)
    print(f"\n4. Manifest guardado: {out_path}")
    print(f"   Filas: {len(df_segs):,}")

    if not args.no_landmarks:
        n_kp = len(list(LAND_DIR.glob("*.npy")))
        sz = sum(f.stat().st_size for f in LAND_DIR.glob("*.npy"))
        print(f"   Landmarks: {n_kp} archivos, {sz/1e9:.2f} GB")

    print("\n" + "="*60)
    print("LISTO. Próximo paso:")
    print("  .venv310/bin/python scripts/run_training.py")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_videos',      type=int, default=None)
    parser.add_argument('--no_landmarks',    action='store_true')
    parser.add_argument('--stride',          type=int, default=STRIDE)
    parser.add_argument('--temporal_splits', action='store_true',
                        help='Splits temporales dentro de cada video (recomendado con 1 video/clase)')
    args = parser.parse_args()
    main(args)
