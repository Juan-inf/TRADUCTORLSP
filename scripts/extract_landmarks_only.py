"""
Extrae landmarks por segmento usando el manifest_segments.csv existente.
Más eficiente que procesar frame a frame: lee cada video UNA vez y extrae
todos los segmentos en un solo pase por VideoCapture.

Uso: .venv310/bin/python scripts/extract_landmarks_only.py
"""
import sys, json, cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR  = Path("data")
LAND_DIR  = DATA_DIR / "landmarks"
LAND_DIR.mkdir(parents=True, exist_ok=True)

N_FRAMES = 30
N_KP     = 75   # 42 manos + 33 pose

def extract_all_frames(video_path: str):
    """Extrae keypoints MediaPipe frame a frame para todo el video."""
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic

    holistic = mp_holistic.Holistic(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.4, min_tracking_confidence=0.4
    )
    cap = cv2.VideoCapture(video_path)
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
    return all_kp  # list of [N_KP, 3]


def build_segment(all_kp, start, end, n_frames=N_FRAMES):
    """Recorta y normaliza el segmento [start:end]."""
    seg_kp = all_kp[start:end]
    if len(seg_kp) == 0:
        return np.zeros((n_frames, N_KP, 3), dtype=np.float32)
    seg = np.stack(seg_kp)          # [T, N_KP, 3]
    ref = seg[:, 21:22, :]          # muñeca derecha
    seg = seg - ref                 # normalización posicional
    # pad o crop a n_frames
    T = len(seg)
    if T >= n_frames:
        seg = seg[:n_frames]
    else:
        pad = np.zeros((n_frames - T, N_KP, 3), dtype=np.float32)
        seg = np.concatenate([seg, pad], axis=0)
    return seg.astype(np.float32)


def main():
    df = pd.read_csv(DATA_DIR / "manifest_segments.csv")
    already_done = df['kp_path'].notna() & (df['kp_path'] != '')
    print(f"Segmentos totales: {len(df)}")
    print(f"Con kp_path: {already_done.sum()}")
    print(f"Sin kp_path: {(~already_done).sum()}")

    videos = df['video_path'].unique()
    print(f"Videos a procesar: {len(videos)}")

    kp_path_updates = {}

    for video_path in tqdm(videos, desc="Videos"):
        segs_for_video = df[df['video_path'] == video_path]
        # Saltar si todos los segmentos de este video ya tienen kp_path
        pending = segs_for_video[segs_for_video['kp_path'].isna() |
                                 (segs_for_video['kp_path'] == '')]
        if len(pending) == 0:
            continue

        try:
            all_kp = extract_all_frames(video_path)
        except Exception as e:
            print(f"  ERROR {video_path}: {e}")
            continue

        total_frames = len(all_kp)
        print(f"\n  {Path(video_path).name}: {total_frames} frames, "
              f"{len(pending)} segmentos pendientes")

        for _, row in pending.iterrows():
            start = int(row['start_frame'])
            end   = int(row['end_frame'])
            if start >= total_frames:
                continue
            end = min(end, total_frames)

            seg = build_segment(all_kp, start, end, N_FRAMES)

            clase = row['clase']
            kp_file = LAND_DIR / f"{clase}_s{start:06d}_kp.npy"
            np.save(str(kp_file), seg)
            kp_path_updates[row.name] = str(kp_file)

    # Actualizar manifest con kp_path
    for idx, kp_path in kp_path_updates.items():
        df.at[idx, 'kp_path'] = kp_path

    df.to_csv(DATA_DIR / "manifest_segments.csv", index=False)

    n_kp = len(list(LAND_DIR.glob("*.npy")))
    sz   = sum(f.stat().st_size for f in LAND_DIR.glob("*.npy"))
    print(f"\nListo. Landmarks: {n_kp} archivos, {sz/1e6:.1f} MB")
    print(f"Manifest actualizado: {DATA_DIR / 'manifest_segments.csv'}")


if __name__ == '__main__':
    main()
