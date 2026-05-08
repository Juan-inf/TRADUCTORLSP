"""
VideoMAE fine-tuning para LSP.
Usa MCG-NJU/videomae-base preentrenado en Kinetics con adaptación al dataset LSP.
"""

import torch
import torch.nn as nn
from typing import Optional


def build_videomae_model(n_classes: int, num_frames: int = 16, pretrained: bool = True):
    """
    Carga VideoMAE base desde HuggingFace y reemplaza la cabeza de clasificación.

    Retorna el modelo listo para fine-tuning.
    """
    try:
        from transformers import VideoMAEForVideoClassification, VideoMAEConfig
    except ImportError:
        raise ImportError("Instalar: pip install transformers>=4.30")

    model_name = "MCG-NJU/videomae-base-finetuned-kinetics"

    if pretrained:
        model = VideoMAEForVideoClassification.from_pretrained(
            model_name,
            num_labels=n_classes,
            ignore_mismatched_sizes=True,
        )
    else:
        config = VideoMAEConfig(
            num_frames=num_frames,
            num_labels=n_classes,
        )
        model = VideoMAEForVideoClassification(config)

    return model


class VideoMAEWrapper(nn.Module):
    """
    Wrapper de VideoMAE con:
    - Congelado del backbone los primeros N epochs
    - Extracción de embedding para fusión multimodal
    - API unificada con el resto de modelos LSP
    """

    def __init__(
        self,
        n_classes: int,
        num_frames: int = 16,
        pretrained: bool = True,
        freeze_backbone_epochs: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.num_frames = num_frames
        self._freeze_epochs = freeze_backbone_epochs

        self.model = build_videomae_model(n_classes, num_frames, pretrained)

        # Añadir dropout antes del clasificador final
        hidden_size = self.model.config.hidden_size
        self.model.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes),
        )

        if pretrained:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False

    def _unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def on_epoch_start(self, epoch: int):
        if epoch == self._freeze_epochs:
            self._unfreeze_backbone()
            print(f"Época {epoch}: VideoMAE backbone descongelado.")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: [B, T, C, H, W]  (formato HuggingFace)

        Nota: nuestro DataLoader produce [B, C, T, H, W].
        Esta función reordena internamente.
        """
        if pixel_values.shape[1] != self.num_frames:
            # Reordenar de [B, C, T, H, W] a [B, T, C, H, W]
            pixel_values = pixel_values.permute(0, 2, 1, 3, 4)

        out = self.model(pixel_values=pixel_values)
        return out.logits

    def get_embedding(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.shape[1] != self.num_frames:
            pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        out = self.model.videomae(pixel_values=pixel_values)
        # CLS token (primer token de la secuencia)
        return out.last_hidden_state[:, 0, :]
