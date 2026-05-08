"""
Fusión multimodal: combina embeddings de píxeles (CNN-LSTM/VideoMAE)
y landmarks (ST-GCN) para mejorar precisión en señas similares.
"""

import torch
import torch.nn as nn
from typing import Optional


class MultimodalFusion(nn.Module):
    """
    Fusión tardía de dos ramas:
      - Rama A (pixels):     embedding_a de dimensión dim_a
      - Rama B (landmarks):  embedding_b de dimensión dim_b

    Estrategias soportadas:
      'concat'      : concatenar y proyectar con MLP
      'attention'   : cross-attention entre ramas
      'weighted_sum': suma ponderada aprendible
    """

    def __init__(
        self,
        dim_a: int,
        dim_b: int,
        n_classes: int,
        fusion_strategy: str = 'concat',
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        assert fusion_strategy in ('concat', 'attention', 'weighted_sum')
        self.fusion_strategy = fusion_strategy
        self.n_classes = n_classes

        if fusion_strategy == 'concat':
            self.classifier = nn.Sequential(
                nn.Linear(dim_a + dim_b, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )

        elif fusion_strategy == 'attention':
            # Proyectar ambas ramas al mismo espacio
            self.proj_a = nn.Linear(dim_a, hidden_dim)
            self.proj_b = nn.Linear(dim_b, hidden_dim)
            self.attn   = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )

        elif fusion_strategy == 'weighted_sum':
            self.proj_a = nn.Linear(dim_a, hidden_dim)
            self.proj_b = nn.Linear(dim_b, hidden_dim)
            self.gate   = nn.Parameter(torch.tensor(0.5))
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        if self.fusion_strategy == 'concat':
            fused = torch.cat([emb_a, emb_b], dim=-1)
            return self.classifier(fused)

        elif self.fusion_strategy == 'attention':
            qa = self.proj_a(emb_a).unsqueeze(1)   # [B, 1, D]
            kb = self.proj_b(emb_b).unsqueeze(1)   # [B, 1, D]
            out, _ = self.attn(qa, kb, kb)
            return self.classifier(out.squeeze(1))

        elif self.fusion_strategy == 'weighted_sum':
            alpha = torch.sigmoid(self.gate)
            fused = alpha * self.proj_a(emb_a) + (1 - alpha) * self.proj_b(emb_b)
            return self.classifier(fused)


class LSPFusionModel(nn.Module):
    """
    Modelo completo de fusión multimodal para LSP.
    Encapsula ambas ramas y el módulo de fusión.

    Permite entrenamiento conjunto end-to-end o por separado de cada rama.
    """

    def __init__(
        self,
        pixel_backbone: nn.Module,
        landmark_backbone: nn.Module,
        dim_pixels: int,
        dim_landmarks: int,
        n_classes: int,
        fusion_strategy: str = 'concat',
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.pixel_backbone = pixel_backbone
        self.landmark_backbone = landmark_backbone
        self.fusion = MultimodalFusion(
            dim_pixels, dim_landmarks, n_classes,
            fusion_strategy, hidden_dim, dropout,
        )

    def forward(
        self,
        pixels: torch.Tensor,
        landmarks: torch.Tensor,
    ) -> torch.Tensor:
        emb_pixels    = self.pixel_backbone.get_embedding(pixels)
        emb_landmarks = self.landmark_backbone.get_embedding(landmarks)
        return self.fusion(emb_pixels, emb_landmarks)

    def on_epoch_start(self, epoch: int):
        for backbone in (self.pixel_backbone, self.landmark_backbone):
            if hasattr(backbone, 'on_epoch_start'):
                backbone.on_epoch_start(epoch)
