"""Extracción de landmarks MediaPipe → features del modelo BiLSTM.

Compartido entre demo/app_gradio.py y api/main.py para que ambos alimenten
al modelo exactamente con el mismo contrato que scripts/train_s27.py.
"""

import numpy as np

N_KP   = 75    # [0:21]=left_hand, [21:42]=right_hand, [42:75]=pose
N_DIMS = 150   # pose_x/y(33) + left_x/y(21) + right_x/y(21)


def aplicar_respaldo_manos_y_pose(results, frame_rgb: np.ndarray, hands_solution,
                                   descartar_pose_sin_rostro: bool = False):
    """Corrige un resultado de MediaPipe Holistic con hasta dos respaldos —
    ver ENTREGABLE_PLAN_DE_DESPLIEGUE_S13.md R14:

    1. Si Holistic no encontró NINGUNA mano, se corre `hands_solution`
       (mediapipe.solutions.hands.Hands, no depende de pose) como respaldo.
       Holistic recorta la región de cada mano usando la muñeca que estima
       SU PROPIO modelo de pose; si esa pose es una adivinanza mala (p.ej.
       encuadre de mano sola sin cuerpo, como el dataset de entrenamiento
       del abecedario), el recorte falla y la mano nunca se detecta aunque
       ocupe casi todo el frame. `hands_solution` debe ser una instancia ya
       inicializada (se reutiliza entre llamadas, no se crea una por frame).
       Se aplica siempre — nunca reemplaza una detección real, solo actúa
       cuando Holistic ya falló en encontrar mano por su cuenta. Seguro en
       las 5 rutas (cámara, video, WebSocket, imagen).
    2. Si no hay rostro detectado, se descarta la pose que haya devuelto
       Holistic. Validado y seguro **solo en imagen estática** (una foto de
       mano sola sin cuerpo nunca tiene rostro, y su pose es siempre una
       adivinanza de baja confianza). **No** se activa por defecto — en
       cámara/video/WebSocket, una persona señando de verdad puede tapar
       momentáneamente su propio rostro con la mano al hacer una seña cerca
       de la cara (común en LSP), lo que haría fallar la detección de
       rostro en ese frame puntual aunque el cuerpo sea real y la pose
       también — descartarla ahí borraría información válida a mitad de una
       seña real y degradaría la clasificación. Pasar
       `descartar_pose_sin_rostro=True` solo desde `process_image()`."""
    if results.left_hand_landmarks is None and results.right_hand_landmarks is None:
        hands_result = hands_solution.process(frame_rgb)
        if hands_result.multi_hand_landmarks:
            for lm, handedness in zip(hands_result.multi_hand_landmarks,
                                       hands_result.multi_handedness):
                if handedness.classification[0].label == "Left":
                    results.left_hand_landmarks = lm
                else:
                    results.right_hand_landmarks = lm
    if descartar_pose_sin_rostro and results.face_landmarks is None:
        results.pose_landmarks = None
    return results


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
