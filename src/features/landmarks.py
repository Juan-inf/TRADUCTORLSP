"""Extracción de landmarks MediaPipe → features del modelo BiLSTM.

Compartido entre demo/app_gradio.py y api/main.py para que ambos alimenten
al modelo exactamente con el mismo contrato que scripts/train_s27.py.
"""

import numpy as np

N_KP   = 75    # [0:21]=left_hand, [21:42]=right_hand, [42:75]=pose
N_DIMS = 150   # pose_x/y(33) + left_x/y(21) + right_x/y(21)


def results_to_kp(results) -> np.ndarray:
    """MediaPipe Holistic results → array kp [75, 3]: [left(21), right(21), pose(33)]."""
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
    return kp


def kp_seq_to_features(kp_seq: np.ndarray) -> np.ndarray:
    """kp_seq [T, 75, 3] → [T, 150]: pose_x/y(33) + left_x/y(21) + right_x/y(21)."""
    pose  = kp_seq[:, 42:75, :2]
    left  = kp_seq[:,  0:21, :2]
    right = kp_seq[:, 21:42, :2]
    return np.concatenate([
        pose[:, :, 0], pose[:, :, 1],
        left[:, :, 0], left[:, :, 1],
        right[:, :, 0], right[:, :, 1],
    ], axis=1).astype(np.float32)


def normalize_sample(x: np.ndarray) -> np.ndarray:
    """Z-score escalar global sobre la secuencia completa — idéntico a
    scripts/train_s27.py:normalize_sample. Requerido por checkpoints con
    normalize_input=True (s27+); s13 no lo necesita pero no le hace daño."""
    mu, std = x.mean(), x.std()
    if std < 1e-8:
        return x
    return ((x - mu) / std).astype(np.float32)


def resample_to_n_frames(seq: list, n_frames: int = 30) -> np.ndarray:
    """Remuestrea una secuencia de longitud variable a exactamente n_frames,
    idéntico al criterio de scripts/build_dataset_s17.py:pkl_to_sequence —
    submuestreo uniforme (np.linspace) si es más larga, padding con el último
    frame si es más corta. Necesario para clasificar segmentos de duración
    variable (detectados por pausa) con un modelo entrenado a longitud fija."""
    seq = list(seq)
    if len(seq) > n_frames:
        idx = np.linspace(0, len(seq) - 1, n_frames, dtype=int)
        seq = [seq[i] for i in idx]
    while len(seq) < n_frames:
        seq.append(seq[-1])
    return np.array(seq[:n_frames], dtype=np.float32)
