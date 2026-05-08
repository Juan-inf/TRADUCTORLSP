"""
Baseline: CNN-LSTM para reconocimiento de LSP.
ResNet50 extrae características por frame → LSTM modela la secuencia temporal.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class CNNLSTM(nn.Module):
    """
    Arquitectura CNN-LSTM:
      1. Backbone CNN (ResNet50 preentrenado) aplicado frame a frame.
      2. LSTM de 2 capas sobre la secuencia de embeddings por frame.
      3. Capa lineal de clasificación.

    Input : [B, C, T, H, W]
    Output: [B, n_classes]
    """

    def __init__(
        self,
        n_classes: int,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        pretrained: bool = True,
        freeze_backbone_epochs: int = 5,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self._freeze_epochs = freeze_backbone_epochs
        self._current_epoch = 0

        # Backbone CNN: ResNet50 sin la cabeza de clasificación
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = models.resnet50(weights=weights)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # output: [B, 2048, 1, 1]
        self.embed_dim = 2048

        # Proyección a espacio menor antes del LSTM
        self.frame_proj = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

        if pretrained:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.cnn.parameters():
            param.requires_grad = False

    def _unfreeze_backbone(self):
        for param in self.cnn.parameters():
            param.requires_grad = True

    def on_epoch_start(self, epoch: int):
        self._current_epoch = epoch
        if epoch == self._freeze_epochs:
            self._unfreeze_backbone()
            print(f"Época {epoch}: backbone CNN descongelado para fine-tuning completo.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T, H, W]
        """
        B, C, T, H, W = x.shape

        # Procesar todos los frames en un batch para eficiencia
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)   # [B*T, C, H, W]
        features = self.cnn(x).flatten(1)                         # [B*T, 2048]
        features = self.frame_proj(features)                      # [B*T, 512]
        features = features.view(B, T, 512)                       # [B, T, 512]

        # LSTM sobre la secuencia temporal
        lstm_out, _ = self.lstm(features)    # [B, T, hidden*2]
        # Global average pooling temporal
        out = lstm_out.mean(dim=1)           # [B, hidden*2]

        logits = self.classifier(out)        # [B, n_classes]
        return logits

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna embedding pre-clasificación para fusión multimodal."""
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        features = self.cnn(x).flatten(1)
        features = self.frame_proj(features).view(B, T, 512)
        lstm_out, _ = self.lstm(features)
        return lstm_out.mean(dim=1)   # [B, hidden*2]
