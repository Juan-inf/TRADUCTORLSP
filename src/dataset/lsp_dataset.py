"""
LSPVideoDataset — PyTorch Dataset para videos MP4 de Lengua de Señas Peruana.

Soporta tres modos:
  'pixels'    : tensor de frames [C, T, H, W] (para 3D-CNN / VideoMAE)
  'landmarks' : secuencia de keypoints [T, N, 3] (para ST-GCN / LSTM)
  'both'      : ambos (para fusión multimodal)
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import random
import cv2

from ..preprocessing.video_preprocessor import VideoPreprocessor
from ..preprocessing.landmark_extractor import LandmarkExtractor


class LSPVideoDataset(Dataset):
    """
    Dataset para el pipeline LSP. Lee MP4 con OpenCV o cached .npy.

    Parámetros
    ----------
    manifest_csv  : ruta al CSV con columnas [ruta, clase, ...]
    label2idx     : dict clase→índice
    mode          : 'pixels' | 'landmarks' | 'both'
    split         : 'train' | 'val' | 'test'
    pixels_dir    : directorio de .npy precalculados (pixels). None = leer MP4 en vuelo.
    landmarks_dir : directorio de .npy precalculados (landmarks). None = extraer en vuelo.
    augment       : aplicar aumentación (solo en split='train')
    n_frames      : ventana temporal
    img_size      : resolución HxW
    """

    def __init__(
        self,
        manifest_csv: str,
        label2idx: Dict[str, int],
        mode: str = 'both',
        split: str = 'train',
        pixels_dir: Optional[str] = None,
        landmarks_dir: Optional[str] = None,
        augment: bool = True,
        n_frames: int = 30,
        img_size: Tuple[int, int] = (224, 224),
        imagenet_norm: bool = True,
    ):
        assert mode in ('pixels', 'landmarks', 'both'), "mode debe ser 'pixels', 'landmarks' o 'both'"
        assert split in ('train', 'val', 'test')

        self.df = pd.read_csv(manifest_csv)
        # Filtrar por split si la columna existe
        if 'split' in self.df.columns:
            self.df = self.df[self.df['split'] == split].reset_index(drop=True)

        self.label2idx = label2idx
        self.mode = mode
        self.split = split
        self.pixels_dir = Path(pixels_dir) if pixels_dir else None
        self.landmarks_dir = Path(landmarks_dir) if landmarks_dir else None
        self.augment = augment and (split == 'train')
        self.n_frames = n_frames
        self.img_size = img_size

        self.preprocessor = VideoPreprocessor(
            height=img_size[0], width=img_size[1],
            n_frames=n_frames, imagenet_norm=imagenet_norm,
        )
        self.landmark_extractor = LandmarkExtractor() \
            if mode in ('landmarks', 'both') else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        label = self.label2idx.get(row['clase'], 0)
        result = {'label': torch.tensor(label, dtype=torch.long), 'clase': row['clase']}

        if self.mode in ('pixels', 'both'):
            result['pixels'] = self._load_pixels(row)

        if self.mode in ('landmarks', 'both'):
            result['landmarks'] = self._load_landmarks(row)

        return result

    # ── Carga de pixels ──────────────────────────────────────────────────

    def _get_video_path(self, row: pd.Series) -> str:
        if 'video_path' in row.index:
            return row['video_path']
        return row['ruta']

    def _load_pixels(self, row: pd.Series) -> torch.Tensor:
        video_path = self._get_video_path(row)
        stem = Path(video_path).stem

        # Intentar cargar .npy precalculado
        if self.pixels_dir:
            npy_path = self.pixels_dir / f"{stem}.npy"
            if npy_path.exists():
                arr = np.load(str(npy_path))   # [T, H, W, C]
                if self.augment:
                    arr = self._augment_pixels(arr)
                return self._pixels_to_tensor(arr)

        # Leer segmento desde el video usando start_frame/end_frame si están disponibles
        start = int(row['start_frame']) if 'start_frame' in row.index else 0
        end   = int(row['end_frame'])   if 'end_frame'   in row.index else None
        arr = self.preprocessor.process(video_path, start, end)
        if self.augment:
            arr = self._augment_pixels(arr)
        return self._pixels_to_tensor(arr)

    def _pixels_to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """[T, H, W, C] float32 → [C, T, H, W] tensor."""
        t = torch.from_numpy(np.transpose(arr, (3, 0, 1, 2)))
        return t

    # ── Carga de landmarks ───────────────────────────────────────────────

    def _load_landmarks(self, row: pd.Series) -> torch.Tensor:
        # Prioridad 1: kp_path por segmento en el manifest (de preprocess_sliding_window.py)
        if 'kp_path' in row.index and pd.notna(row['kp_path']) and str(row['kp_path']).strip():
            kp_path = Path(str(row['kp_path']))
            if kp_path.exists():
                seq = np.load(str(kp_path)).astype(np.float32)
                if self.augment:
                    seq = self._augment_landmarks(seq)
                return torch.from_numpy(seq)

        # Prioridad 2: .npy en landmarks_dir por nombre de video
        video_path = self._get_video_path(row)
        if self.landmarks_dir:
            npy_path = self.landmarks_dir / f"{Path(video_path).stem}_kp.npy"
            if npy_path.exists():
                seq = np.load(str(npy_path)).astype(np.float32)
                if self.augment:
                    seq = self._augment_landmarks(seq)
                return torch.from_numpy(seq)

        # Prioridad 3: extraer on-the-fly para el segmento específico
        start = int(row['start_frame']) if 'start_frame' in row.index else 0
        end   = int(row['end_frame'])   if 'end_frame'   in row.index else None
        seq = self.landmark_extractor.extract(
            video_path, self.n_frames, start, end
        ).astype(np.float32)
        if self.augment:
            seq = self._augment_landmarks(seq)
        return torch.from_numpy(seq)

    # ── Aumentación de pixels ─────────────────────────────────────────────

    def _augment_pixels(self, arr: np.ndarray) -> np.ndarray:
        """arr: [T, H, W, C] float32 normalizado."""

        # Flip horizontal (50%)
        if random.random() < 0.5:
            arr = arr[:, :, ::-1, :].copy()

        # Variación de velocidad (±25%)
        if random.random() < 0.5:
            factor = random.choice([0.75, 1.25])
            T = len(arr)
            new_T = max(1, int(T * factor))
            new_indices = np.linspace(0, T - 1, new_T, dtype=int)
            arr = arr[new_indices]

        # Recorte temporal aleatorio
        T = len(arr)
        if T > self.n_frames:
            start = random.randint(0, T - self.n_frames)
            arr = arr[start: start + self.n_frames]

        # Pad si hace falta tras el crop
        if len(arr) < self.n_frames:
            pad = np.stack([arr[-1]] * (self.n_frames - len(arr)))
            arr = np.concatenate([arr, pad], axis=0)

        # Jitter de brillo/contraste (±10%) — aplicado sobre datos normalizados
        if random.random() < 0.5:
            factor = 1.0 + random.uniform(-0.1, 0.1)
            arr = (arr * factor).astype(np.float32)

        # Rotación leve ±10° (aplicar por frame, costoso pero correcto)
        if random.random() < 0.3:
            angle = random.uniform(-10, 10)
            H, W = arr.shape[1], arr.shape[2]
            M = cv2.getRotationMatrix2D((W // 2, H // 2), angle, 1.0)
            arr = np.stack([
                cv2.warpAffine(f, M, (W, H)).astype(np.float32)
                for f in arr
            ])

        return arr

    # ── Aumentación de landmarks ──────────────────────────────────────────

    def _augment_landmarks(self, seq: np.ndarray) -> np.ndarray:
        """seq: [T, N, 3] float32."""

        # Ruido gaussiano
        if random.random() < 0.5:
            seq = seq + np.random.normal(0, 0.01, seq.shape).astype(np.float32)

        # Flip: invertir eje X e intercambiar mano izq (0:21) ↔ derecha (21:42)
        if random.random() < 0.5:
            seq = seq.copy()
            seq[:, :, 0] = -seq[:, :, 0]
            left  = seq[:, :21, :].copy()
            right = seq[:, 21:42, :].copy()
            seq[:, :21, :]  = right
            seq[:, 21:42, :] = left

        # Variación de velocidad
        if random.random() < 0.5:
            T = len(seq)
            factor = random.choice([0.75, 1.25])
            new_T = max(1, int(T * factor))
            new_idx = np.linspace(0, T - 1, new_T, dtype=int)
            seq = seq[new_idx]

        # Pad/crop a n_frames
        T = len(seq)
        if T > self.n_frames:
            start = random.randint(0, T - self.n_frames)
            seq = seq[start: start + self.n_frames]
        elif T < self.n_frames:
            pad = np.zeros((self.n_frames - T, seq.shape[1], seq.shape[2]), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=0)

        return seq


# ── Splits estratificados ────────────────────────────────────────────────────

def create_splits(
    manifest_csv: str,
    output_csv: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Añade columna 'split' al manifest usando splits temporales por video.
    Funciona tanto con manifest.csv (nivel video) como manifest_segments.csv
    (nivel segmento). Con 1 video por clase, la estratificación cruzada es
    imposible; splits temporales garantizan que cada clase aparezca en los 3 splits.
    """
    df = pd.read_csv(manifest_csv)

    # Detectar si es manifest de segmentos (tiene start_frame) o de videos
    is_segments = 'start_frame' in df.columns
    group_col = 'num_vineta' if 'num_vineta' in df.columns else 'clase'

    splits = []
    for _, group in df.groupby(group_col, sort=True):
        n = len(group)
        i_train = int(n * train_ratio)
        i_val   = int(n * (train_ratio + val_ratio))
        s = (['train'] * i_train
             + ['val']   * (i_val - i_train)
             + ['test']  * (n - i_val))
        # Garantizar al menos 1 en val y test si el grupo es muy pequeño
        if n >= 3 and i_train >= n:
            s[-2] = 'val'
            s[-1] = 'test'
        splits.extend(s)

    df['split'] = splits
    df.to_csv(output_csv, index=False)

    counts = df['split'].value_counts()
    print(f"Splits: train={counts.get('train',0)}, val={counts.get('val',0)}, test={counts.get('test',0)}")
    return df


# ── DataLoader factory ───────────────────────────────────────────────────────

def get_dataloaders(
    manifest_csv: str,
    label2idx: Dict[str, int],
    mode: str = 'both',
    pixels_dir: Optional[str] = None,
    landmarks_dir: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    class_weights: Optional[Dict[int, float]] = None,
) -> Dict[str, DataLoader]:
    """
    Crea DataLoaders para train/val/test con WeightedRandomSampler en train.
    """
    loaders = {}
    for split in ('train', 'val', 'test'):
        dataset = LSPVideoDataset(
            manifest_csv=manifest_csv,
            label2idx=label2idx,
            mode=mode,
            split=split,
            pixels_dir=pixels_dir,
            landmarks_dir=landmarks_dir,
            augment=(split == 'train'),
        )

        sampler = None
        shuffle = False
        if split == 'train' and class_weights:
            labels = [label2idx[r['clase']] for _, r in dataset.df.iterrows()]
            weights = [class_weights.get(l, 1.0) for l in labels]
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        elif split == 'train':
            shuffle = True

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == 'train'),
        )

    return loaders
