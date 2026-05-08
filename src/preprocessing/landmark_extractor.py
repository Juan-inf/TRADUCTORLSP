"""
Extracción de landmarks con MediaPipe Holistic por cada frame de un video MP4.
Exporta secuencias [T, N_keypoints, 3] como .npy.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional
import json

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("ADVERTENCIA: mediapipe no disponible. Instalar: pip install mediapipe")


class LandmarkExtractor:
    """
    Aplica MediaPipe Holistic frame a frame sobre un video MP4.

    Keypoints extraídos:
    - Mano izquierda:  21 puntos × 3 (x, y, z)
    - Mano derecha:    21 puntos × 3
    - Pose (cuerpo):   33 puntos × 3 (solo si use_pose=True)
    - Cara:           468 puntos × 3 (solo si use_face=True, pesado)

    Normalización posicional: coordenadas relativas a muñeca derecha (punto 0 mano derecha),
    lo que permite invarianza a posición absoluta en el frame.
    """

    # Índices anatómicos para el grafo ST-GCN (manos)
    HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),       # pulgar
        (0,5),(5,6),(6,7),(7,8),       # índice
        (0,9),(9,10),(10,11),(11,12),  # medio
        (0,13),(13,14),(14,15),(15,16),# anular
        (0,17),(17,18),(18,19),(19,20),# meñique
        (5,9),(9,13),(13,17),          # metacarpo
    ]

    POSE_UPPER_BODY = list(range(11, 25))  # hombros, codos, muñecas, caderas

    def __init__(
        self,
        use_pose: bool = True,
        use_face: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        normalize_wrist: bool = True,
        target_fps: float = 25.0,
    ):
        if not MP_AVAILABLE:
            raise ImportError("mediapipe requerido. pip install mediapipe")

        self.use_pose = use_pose
        self.use_face = use_face
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.normalize_wrist = normalize_wrist
        self.target_fps = target_fps

        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils

        # Dimensiones del vector de keypoints por frame
        self.n_hand_points = 21 * 2   # mano izq + mano der
        self.n_pose_points = 33 if use_pose else 0
        self.n_face_points = 468 if use_face else 0
        self.n_total = self.n_hand_points + self.n_pose_points + self.n_face_points

    # ── Extracción por frame ─────────────────────────────────────────────

    def _extract_holistic(self, frame_rgb: np.ndarray, holistic) -> np.ndarray:
        """
        Corre MediaPipe Holistic sobre un frame RGB.

        Retorna
        -------
        kp : np.ndarray  shape [N, 3]  — coordenadas (x, y, z) normalizadas [0,1]
        """
        results = holistic.process(frame_rgb)

        parts = []

        # Mano izquierda
        if results.left_hand_landmarks:
            lm = results.left_hand_landmarks.landmark
            left = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        else:
            left = np.zeros((21, 3), dtype=np.float32)
        parts.append(left)

        # Mano derecha
        if results.right_hand_landmarks:
            lm = results.right_hand_landmarks.landmark
            right = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        else:
            right = np.zeros((21, 3), dtype=np.float32)
        parts.append(right)

        # Pose (cuerpo superior)
        if self.use_pose:
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                pose = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
            else:
                pose = np.zeros((33, 3), dtype=np.float32)
            parts.append(pose)

        # Cara (costoso, desactivado por defecto)
        if self.use_face:
            if results.face_landmarks:
                lm = results.face_landmarks.landmark
                face = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
            else:
                face = np.zeros((468, 3), dtype=np.float32)
            parts.append(face)

        kp = np.concatenate(parts, axis=0)  # [N, 3]
        return kp

    def _normalize_keypoints(self, seq: np.ndarray) -> np.ndarray:
        """
        Normaliza coordenadas respecto a la muñeca derecha (punto 21, índice 21).
        Invarianza posicional: la seña no depende de dónde esté la mano en el frame.
        """
        # Punto 21 = primer keypoint de la mano derecha
        wrist = seq[:, 21:22, :]   # [T, 1, 3]  muñeca derecha
        seq_norm = seq - wrist     # resta posición absoluta
        return seq_norm

    # ── Lectura de video y extracción completa ───────────────────────────

    def extract(self, video_path: str, n_frames: int = 30,
                start_frame: int = 0, end_frame: Optional[int] = None) -> np.ndarray:
        """
        Extrae landmarks de un video MP4 (o segmento si se pasan start/end_frame).

        Retorna
        -------
        sequence : np.ndarray  shape [n_frames, N_keypoints, 3]
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"No se pudo abrir: {video_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or self.target_fps
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        seg_end = min(end_frame, total) if end_frame is not None else total
        seg_end = max(seg_end, start_frame + 1)
        seg_len = seg_end - start_frame

        n_out = max(1, int((seg_len / max(src_fps, 1e-6)) * self.target_fps))
        indices = (np.linspace(0, seg_len - 1, n_out, dtype=int) + start_frame)
        indices = np.clip(indices, start_frame, seg_end - 1)

        frames = {}
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()

        if not frames:
            return np.zeros((n_frames, self.n_total, 3), dtype=np.float32)

        # Extraer landmarks con MediaPipe Holistic
        keypoints_seq = []
        with self.mp_holistic.Holistic(
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            model_complexity=1,
        ) as holistic:
            for idx in sorted(frames.keys()):
                kp = self._extract_holistic(frames[idx], holistic)
                keypoints_seq.append(kp)

        seq = np.stack(keypoints_seq)  # [T_raw, N, 3]

        # Normalización posicional
        if self.normalize_wrist:
            seq = self._normalize_keypoints(seq)

        # Pad o crop a n_frames fijo
        seq = self._temporal_fix(seq, n_frames)
        return seq   # [n_frames, N, 3]

    def _temporal_fix(self, seq: np.ndarray, n_frames: int) -> np.ndarray:
        T = len(seq)
        if T >= n_frames:
            start = (T - n_frames) // 2
            return seq[start: start + n_frames]
        pad = np.zeros((n_frames - T, seq.shape[1], seq.shape[2]), dtype=np.float32)
        return np.concatenate([seq, pad], axis=0)

    # ── Batch processing ─────────────────────────────────────────────────

    def batch_extract(
        self,
        manifest_csv: str,
        output_dir: str,
        n_frames: int = 30,
        overwrite: bool = False,
    ) -> None:
        """
        Extrae landmarks para todos los videos del manifest.
        Guarda un .npy por video con shape [n_frames, N_kp, 3].
        """
        import pandas as pd
        from tqdm import tqdm

        df = pd.read_csv(manifest_csv)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        errors = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extrayendo landmarks"):
            out_path = out_dir / f"{Path(row['ruta']).stem}_kp.npy"
            if not overwrite and out_path.exists():
                continue
            try:
                seq = self.extract(row['ruta'], n_frames=n_frames)
                np.save(str(out_path), seq)
            except Exception as e:
                errors.append({'video': row['ruta'], 'error': str(e)})

        if errors:
            with open(out_dir / 'landmark_errors.json', 'w') as f:
                json.dump(errors, f, indent=2)
            print(f"Errores: {len(errors)}")
        print(f"Landmarks exportados en: {out_dir}")

    # ── Visualización ────────────────────────────────────────────────────

    def visualize_frame(self, frame_rgb: np.ndarray, kp: np.ndarray) -> np.ndarray:
        """Dibuja los keypoints sobre el frame para verificación visual."""
        frame_vis = frame_rgb.copy()
        h, w = frame_vis.shape[:2]

        # Puntos de manos
        for i in range(self.n_hand_points):
            x = int(kp[i, 0] * w)
            y = int(kp[i, 1] * h)
            color = (0, 255, 100) if i < 21 else (255, 100, 0)
            cv2.circle(frame_vis, (x, y), 3, color, -1)

        # Conexiones mano derecha
        for a, b in self.HAND_CONNECTIONS:
            xa = int(kp[21 + a, 0] * w)
            ya = int(kp[21 + a, 1] * h)
            xb = int(kp[21 + b, 0] * w)
            yb = int(kp[21 + b, 1] * h)
            cv2.line(frame_vis, (xa, ya), (xb, yb), (255, 200, 0), 1)

        return frame_vis
