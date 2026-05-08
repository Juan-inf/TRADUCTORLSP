"""
Predictor en tiempo real para Lengua de Señas Peruana.
Opera sobre stream de cámara web con latencia objetivo <200ms.
"""

import time
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from collections import deque
import threading


class LSPPredictor:
    """
    Predictor de señas LSP en tiempo real desde cámara o video.

    Arquitectura de sliding window:
    - Mantiene un buffer circular de N frames
    - Cada N/2 frames hace una predicción (overlap 50%)
    - Suavizado temporal con media móvil de predicciones recientes
    """

    def __init__(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        label2idx_path: str,
        mode: str = 'both',
        n_frames: int = 30,
        img_size: Tuple[int, int] = (224, 224),
        device: str = 'cuda',
        confidence_threshold: float = 0.7,
        smoothing_window: int = 3,
        imagenet_norm: bool = True,
    ):
        self.mode = mode
        self.n_frames = n_frames
        self.img_size = img_size
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window

        # Cargar modelo
        self.model = model.to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()

        # Label encoder
        with open(label2idx_path, 'r', encoding='utf-8') as f:
            label2idx = json.load(f)
        self.idx2label = {int(v): k for k, v in label2idx.items()}
        self.n_classes = len(label2idx)

        # Buffers de frames
        self.frame_buffer = deque(maxlen=n_frames)    # frames RGB redimensionados
        self.kp_buffer    = deque(maxlen=n_frames)    # landmarks por frame

        # Historial de predicciones para suavizado
        self.pred_history = deque(maxlen=smoothing_window)

        # Normalización ImageNet
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.imagenet_norm = imagenet_norm

        # MediaPipe (lazy init para evitar problemas en ambientes sin cámara)
        self._mp_holistic = None
        self._holistic_instance = None

        # Métricas de latencia
        self._latencies = deque(maxlen=100)

    def _init_mediapipe(self):
        if self._mp_holistic is None:
            import mediapipe as mp
            self._mp_holistic = mp.solutions.holistic
            self._holistic_instance = self._mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=0,  # 0 = más rápido, importante para <200ms
            )
            self._mp_drawing = mp.solutions.drawing_utils

    def _preprocess_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Convierte frame BGR de cámara a RGB normalizado [H, W, C] float32."""
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, self.img_size, interpolation=cv2.INTER_LINEAR)
        frame = frame.astype(np.float32) / 255.0
        if self.imagenet_norm:
            frame = (frame - self.mean) / self.std
        return frame

    def _extract_kp(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Extrae landmarks MediaPipe de un frame RGB original (pre-resize)."""
        results = self._holistic_instance.process(frame_rgb)

        parts = []
        # Mano izquierda
        if results.left_hand_landmarks:
            left = np.array([[p.x, p.y, p.z] for p in results.left_hand_landmarks.landmark], np.float32)
        else:
            left = np.zeros((21, 3), np.float32)
        parts.append(left)

        # Mano derecha
        if results.right_hand_landmarks:
            right = np.array([[p.x, p.y, p.z] for p in results.right_hand_landmarks.landmark], np.float32)
        else:
            right = np.zeros((21, 3), np.float32)
        parts.append(right)

        # Pose
        if results.pose_landmarks:
            pose = np.array([[p.x, p.y, p.z] for p in results.pose_landmarks.landmark], np.float32)
        else:
            pose = np.zeros((33, 3), np.float32)
        parts.append(pose)

        kp = np.concatenate(parts, axis=0)  # [75, 3]

        # Normalizar respecto a muñeca derecha
        wrist = kp[21:22, :]
        kp = kp - wrist

        return kp

    def _build_tensors(self) -> Dict[str, torch.Tensor]:
        """Construye tensores de entrada a partir de los buffers actuales."""
        tensors = {}

        if self.mode in ('pixels', 'both'):
            frames = list(self.frame_buffer)
            # Pad si hay menos de n_frames
            while len(frames) < self.n_frames:
                frames.append(frames[-1] if frames else np.zeros((*self.img_size, 3), np.float32))
            arr = np.stack(frames[-self.n_frames:])          # [T, H, W, C]
            arr = np.transpose(arr, (3, 0, 1, 2))            # [C, T, H, W]
            tensors['pixels'] = torch.from_numpy(arr).unsqueeze(0).to(self.device)

        if self.mode in ('landmarks', 'both'):
            kps = list(self.kp_buffer)
            while len(kps) < self.n_frames:
                kps.append(kps[-1] if kps else np.zeros((75, 3), np.float32))
            seq = np.stack(kps[-self.n_frames:])              # [T, N, 3]
            tensors['landmarks'] = torch.from_numpy(seq).unsqueeze(0).to(self.device)

        return tensors

    @torch.no_grad()
    def predict_buffer(self) -> Optional[Dict]:
        """Realiza una predicción con los frames actuales en el buffer."""
        if len(self.frame_buffer) < self.n_frames // 2:
            return None

        t0 = time.perf_counter()
        tensors = self._build_tensors()

        # Forward pass
        if self.mode == 'pixels':
            logits = self.model(tensors['pixels'])
        elif self.mode == 'landmarks':
            logits = self.model(tensors['landmarks'])
        else:
            logits = self.model(tensors['pixels'], tensors['landmarks'])

        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])

        latency_ms = (time.perf_counter() - t0) * 1000
        self._latencies.append(latency_ms)

        if confidence < self.confidence_threshold:
            return None

        # Suavizado temporal
        self.pred_history.append(pred_idx)
        # Voto mayoritario entre predicciones recientes
        from collections import Counter
        smoothed_idx = Counter(self.pred_history).most_common(1)[0][0]

        return {
            'seña':       self.idx2label.get(smoothed_idx, f'clase_{smoothed_idx}'),
            'idx':        smoothed_idx,
            'confidence': confidence,
            'latency_ms': latency_ms,
            'probs':      probs,
        }

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[Dict]:
        """
        Procesa un frame nuevo de la cámara.
        Actualiza buffers y retorna predicción si el buffer está lleno.
        """
        frame_rgb_original = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Preprocesar para el modelo de pixels
        frame_norm = self._preprocess_frame(frame_bgr)
        self.frame_buffer.append(frame_norm)

        # Extraer landmarks
        if self.mode in ('landmarks', 'both'):
            self._init_mediapipe()
            kp = self._extract_kp(frame_rgb_original)
            self.kp_buffer.append(kp)

        # Predecir cada n_frames/2 frames (stride 50%)
        stride = self.n_frames // 2
        if len(self.frame_buffer) % stride == 0 and len(self.frame_buffer) >= self.n_frames:
            return self.predict_buffer()

        return None

    # ── Visualización en vivo ────────────────────────────────────────────

    def draw_prediction(
        self,
        frame_bgr: np.ndarray,
        result: Optional[Dict],
        show_landmarks: bool = True,
    ) -> np.ndarray:
        """Dibuja predicción y landmarks sobre el frame para visualización."""
        vis = frame_bgr.copy()
        h, w = vis.shape[:2]

        # Landmarks MediaPipe
        if show_landmarks and self.mode in ('landmarks', 'both') and self._holistic_instance:
            import mediapipe as mp
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self._holistic_instance.process(frame_rgb)
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles

            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    vis, results.right_hand_landmarks,
                    mp.solutions.holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 120), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 200, 0), thickness=1),
                )
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    vis, results.left_hand_landmarks,
                    mp.solutions.holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(120, 0, 255), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 150, 255), thickness=1),
                )

        # Panel de predicción
        panel_h = 100
        panel = np.zeros((panel_h, w, 3), dtype=np.uint8)

        if result:
            seña = result['seña']
            conf = result['confidence'] * 100
            lat  = result['latency_ms']

            # Barra de confianza
            bar_w = int(conf / 100 * w)
            bar_color = (0, 220, 50) if conf >= 80 else (0, 180, 220) if conf >= 60 else (50, 50, 220)
            cv2.rectangle(panel, (0, 80), (bar_w, panel_h), bar_color, -1)

            cv2.putText(panel, f"{seña}", (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            cv2.putText(panel, f"Conf: {conf:.1f}%  |  Latencia: {lat:.0f}ms",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        else:
            cv2.putText(panel, "Realizando seña...", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160, 160, 160), 1)

        # Indicador de latencia media
        if self._latencies:
            avg_lat = np.mean(self._latencies)
            lat_color = (0, 255, 0) if avg_lat < 200 else (0, 140, 255)
            cv2.putText(vis, f"Lat avg: {avg_lat:.0f}ms", (w - 160, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, lat_color, 1)

        return np.vstack([vis, panel])

    # ── Ejecución desde cámara ───────────────────────────────────────────

    def run_camera(self, camera_index: int = 0, window_title: str = "LSP Traductor") -> None:
        """Bucle principal de inferencia desde cámara web."""
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self._init_mediapipe()
        last_result = None
        print(f"Cámara iniciada. Presionar 'q' para salir.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = self.process_frame(frame)
            if result is not None:
                last_result = result

            display = self.draw_prediction(frame, last_result)
            cv2.imshow(window_title, display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if self._latencies:
            print(f"\nLatencia media: {np.mean(self._latencies):.1f}ms")
            print(f"Latencia P95:   {np.percentile(self._latencies, 95):.1f}ms")


class ONNXPredictor:
    """
    Predictor optimizado usando el modelo ONNX exportado.
    Más rápido que PyTorch en CPU; ideal para despliegue en instituciones.
    """

    def __init__(
        self,
        onnx_path: str,
        label2idx_path: str,
        n_frames: int = 30,
        img_size: Tuple[int, int] = (224, 224),
    ):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("pip install onnxruntime")

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]

        with open(label2idx_path, 'r', encoding='utf-8') as f:
            label2idx = json.load(f)
        self.idx2label = {int(v): k for k, v in label2idx.items()}

        self.n_frames = n_frames
        self.img_size = img_size

    def predict(self, pixels: np.ndarray, landmarks: Optional[np.ndarray] = None) -> Dict:
        feeds = {}
        if 'pixels' in self.input_names:
            feeds['pixels'] = pixels.astype(np.float32)
        if 'landmarks' in self.input_names and landmarks is not None:
            feeds['landmarks'] = landmarks.astype(np.float32)

        logits = self.session.run(None, feeds)[0]   # [1, n_classes]
        probs = self._softmax(logits[0])
        idx = int(probs.argmax())
        return {'seña': self.idx2label.get(idx, str(idx)), 'confidence': float(probs[idx]), 'probs': probs}

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()
