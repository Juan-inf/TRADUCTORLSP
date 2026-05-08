"""
Preprocesamiento de videos MP4 para el dataset LSP.
Estandariza resolución, FPS y ventana temporal; exporta tensores .npy.
"""

import os
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple
import warnings

try:
    import decord
    from decord import VideoReader, cpu, gpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False


class VideoPreprocessor:
    """
    Lee un MP4 y devuelve un tensor numpy [T, H, W, C] normalizado.

    Parámetros
    ----------
    height, width   : resolución de salida (default 224×224)
    n_frames        : ventana temporal fija (default 30)
    target_fps      : FPS al que remuestrear (default 25)
    imagenet_norm   : aplicar normalización ImageNet (para backbones preentrenados)
    use_decord      : usar decord en lugar de OpenCV para leer (más rápido)
    """

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        height: int = 224,
        width: int = 224,
        n_frames: int = 30,
        target_fps: float = 25.0,
        imagenet_norm: bool = True,
        use_decord: bool = False,
    ):
        self.height = height
        self.width = width
        self.n_frames = n_frames
        self.target_fps = target_fps
        self.imagenet_norm = imagenet_norm
        self.use_decord = use_decord and DECORD_AVAILABLE

    # ── Lectura ───────────────────────────────────────────────────────────

    def _read_frames_opencv(self, path: str,
                             start_frame: int = 0,
                             end_frame: Optional[int] = None) -> Tuple[np.ndarray, float]:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"No se pudo abrir: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or self.target_fps
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        seg_end = min(end_frame, total) if end_frame is not None else total
        seg_end = max(seg_end, start_frame + 1)
        indices = self._resample_indices(seg_end - start_frame, fps, offset=start_frame)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        return np.stack(frames) if frames else np.zeros((1, self.height, self.width, 3), np.uint8), fps

    def _read_frames_decord(self, path: str,
                             start_frame: int = 0,
                             end_frame: Optional[int] = None) -> Tuple[np.ndarray, float]:
        vr = VideoReader(path, ctx=cpu(0))
        fps = vr.get_avg_fps() or self.target_fps
        total = len(vr)
        seg_end = min(end_frame, total) if end_frame is not None else total
        indices = self._resample_indices(seg_end - start_frame, fps, offset=start_frame)
        frames = vr.get_batch(indices.tolist()).asnumpy()
        return frames, fps

    def _resample_indices(self, segment_len: int, src_fps: float, offset: int = 0) -> np.ndarray:
        """Genera índices de frames para remuestrear de src_fps a target_fps."""
        duration = segment_len / max(src_fps, 1e-6)
        n_out = max(1, int(duration * self.target_fps))
        indices = np.linspace(0, segment_len - 1, n_out, dtype=int)
        indices = np.clip(indices, 0, segment_len - 1)
        return indices + offset

    # ── Transformaciones ─────────────────────────────────────────────────

    def _resize_frames(self, frames: np.ndarray) -> np.ndarray:
        resized = np.zeros((len(frames), self.height, self.width, 3), dtype=np.uint8)
        for i, frame in enumerate(frames):
            resized[i] = cv2.resize(
                frame,
                (self.width, self.height),
                interpolation=cv2.INTER_CUBIC
            )
        return resized

    def _temporal_pad_or_crop(self, frames: np.ndarray) -> np.ndarray:
        T = len(frames)
        if T >= self.n_frames:
            # Recorte central
            start = (T - self.n_frames) // 2
            return frames[start: start + self.n_frames]
        # Padding con el último frame
        pad = np.stack([frames[-1]] * (self.n_frames - T))
        return np.concatenate([frames, pad], axis=0)

    def _normalize(self, frames: np.ndarray) -> np.ndarray:
        arr = frames.astype(np.float32) / 255.0
        if self.imagenet_norm:
            arr = (arr - self.IMAGENET_MEAN) / self.IMAGENET_STD
        return arr

    # ── API pública ──────────────────────────────────────────────────────

    def process(self, video_path: str,
                start_frame: int = 0,
                end_frame: Optional[int] = None) -> np.ndarray:
        """
        Lee y preprocesa un video MP4 (o un segmento si se pasan start/end_frame).

        Retorna
        -------
        tensor : np.ndarray  shape [T, H, W, C] float32 normalizado
        """
        if self.use_decord:
            try:
                frames, _ = self._read_frames_decord(video_path, start_frame, end_frame)
            except Exception:
                frames, _ = self._read_frames_opencv(video_path, start_frame, end_frame)
        else:
            frames, _ = self._read_frames_opencv(video_path, start_frame, end_frame)

        frames = self._resize_frames(frames)
        frames = self._temporal_pad_or_crop(frames)
        frames = self._normalize(frames)
        return frames   # [T, H, W, C]

    def process_to_tensor(self, video_path: str):
        """Retorna tensor PyTorch [C, T, H, W] listo para modelos 3D-CNN."""
        import torch
        arr = self.process(video_path)                    # [T, H, W, C]
        arr = np.transpose(arr, (3, 0, 1, 2))            # [C, T, H, W]
        return torch.from_numpy(arr)

    def save_npy(self, video_path: str, out_path: str) -> None:
        arr = self.process(video_path)
        np.save(out_path, arr)

    # ── Procesamiento en batch ───────────────────────────────────────────

    def batch_process(
        self,
        manifest_csv: str,
        output_dir: str,
        overwrite: bool = False,
        n_workers: int = 4,
    ) -> None:
        """
        Procesa todos los videos del manifest y guarda tensores .npy.
        Usa ProcessPoolExecutor para paralelizar.
        """
        import pandas as pd
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from tqdm import tqdm

        df = pd.read_csv(manifest_csv)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        tasks = []
        for _, row in df.iterrows():
            out_path = out_dir / f"{Path(row['ruta']).stem}.npy"
            if not overwrite and out_path.exists():
                continue
            tasks.append((row['ruta'], str(out_path)))

        print(f"Videos a procesar: {len(tasks)}")

        errors = []
        with tqdm(total=len(tasks)) as pbar:
            for video_path, out_path in tasks:
                try:
                    self.save_npy(video_path, out_path)
                except Exception as e:
                    errors.append({'video': video_path, 'error': str(e)})
                pbar.update(1)

        if errors:
            import json
            with open(out_dir / 'preprocessing_errors.json', 'w') as f:
                json.dump(errors, f, indent=2)
            print(f"Errores: {len(errors)} (ver {out_dir}/preprocessing_errors.json)")
        print("Preprocesamiento completado.")
