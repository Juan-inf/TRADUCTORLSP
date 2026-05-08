"""
Motor de entrenamiento unificado para todos los modelos LSP.
Soporta mixed precision, early stopping, W&B logging y checkpointing.
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from typing import Optional, Dict, Any
from collections import defaultdict

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .metrics import compute_metrics


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class LSPTrainer:
    """
    Entrenador genérico para modelos de reconocimiento LSP.

    Parámetros
    ----------
    model       : nn.Module con método forward() y opcionalmente on_epoch_start()
    mode        : 'pixels' | 'landmarks' | 'both' (debe coincidir con el DataLoader)
    device      : 'cuda' | 'cpu'
    output_dir  : directorio para checkpoints y métricas
    """

    def __init__(
        self,
        model: nn.Module,
        loaders: Dict[str, Any],
        mode: str = 'both',
        n_classes: int = None,
        device: str = 'cuda',
        output_dir: str = 'checkpoints',
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        patience: int = 10,
        class_weights: Optional[torch.Tensor] = None,
        mixed_precision: bool = True,
        wandb_project: Optional[str] = None,
        model_name: str = 'lsp_model',
    ):
        self.model = model.to(device)
        self.loaders = loaders
        self.mode = mode
        self.device = device
        self.epochs = epochs
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Loss
        weight_tensor = class_weights.to(device) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        # Optimizador y scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )

        self.scaler = GradScaler(enabled=mixed_precision and torch.cuda.is_available())
        self.mixed_precision = mixed_precision
        self.early_stopping = EarlyStopping(patience=patience)

        self.history = defaultdict(list)
        self.best_val_acc = 0.0

        # Weights & Biases
        self.use_wandb = WANDB_AVAILABLE and wandb_project is not None
        if self.use_wandb:
            wandb.init(project=wandb_project, name=model_name, reinit=True)
            wandb.watch(model, log='gradients', log_freq=50)

    # ── Paso de entrenamiento ────────────────────────────────────────────

    def _forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Selecciona las entradas correctas según el modo del modelo."""
        if self.mode == 'pixels':
            return self.model(batch['pixels'].to(self.device))
        elif self.mode == 'landmarks':
            return self.model(batch['landmarks'].to(self.device))
        else:  # 'both'
            return self.model(
                batch['pixels'].to(self.device),
                batch['landmarks'].to(self.device),
            )

    def _run_epoch(self, loader, train: bool = True) -> Dict[str, float]:
        self.model.train(train)
        total_loss = 0.0
        all_preds, all_labels = [], []

        for batch in loader:
            labels = batch['label'].to(self.device)

            if train:
                self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.mixed_precision):
                logits = self._forward(batch)
                loss = self.criterion(logits, labels)

            if train:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        n = len(loader.dataset)
        metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
        metrics['loss'] = total_loss / n
        return metrics

    # ── Loop principal ───────────────────────────────────────────────────

    def train(self) -> Dict[str, list]:
        print(f"\nIniciando entrenamiento — {self.model_name}")
        print(f"Device: {self.device} | Épocas: {self.epochs} | Modo: {self.mode}\n")

        best_ckpt = self.output_dir / f"{self.model_name}_best.pt"

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            if hasattr(self.model, 'on_epoch_start'):
                self.model.on_epoch_start(epoch)

            train_metrics = self._run_epoch(self.loaders['train'], train=True)
            val_metrics   = self._run_epoch(self.loaders['val'],   train=False)

            self.scheduler.step()
            elapsed = time.time() - t0

            # Registro
            for k, v in train_metrics.items():
                self.history[f'train_{k}'].append(v)
            for k, v in val_metrics.items():
                self.history[f'val_{k}'].append(v)

            # Checkpoint del mejor modelo
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                torch.save({
                    'epoch': epoch,
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'val_acc': self.best_val_acc,
                    'history': dict(self.history),
                }, best_ckpt)

            # Logging
            lr_now = self.scheduler.get_last_lr()[0]
            print(
                f"Época {epoch:3d}/{self.epochs} [{elapsed:.1f}s] "
                f"| train loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} "
                f"| val loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} "
                f"| f1={val_metrics['f1_macro']:.4f} | lr={lr_now:.2e}"
            )

            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'lr': lr_now,
                    **{f'train/{k}': v for k, v in train_metrics.items()},
                    **{f'val/{k}': v for k, v in val_metrics.items()},
                })

            if self.early_stopping.step(val_metrics['loss']):
                print(f"\nEarly stopping en época {epoch}. Mejor val acc: {self.best_val_acc:.4f}")
                break

        # Guardar historial
        with open(self.output_dir / f"{self.model_name}_history.json", 'w') as f:
            json.dump(dict(self.history), f, indent=2)

        if self.use_wandb:
            wandb.finish()

        print(f"\nEntrenamiento completado. Mejor val acc: {self.best_val_acc:.4f}")
        print(f"Checkpoint guardado en: {best_ckpt}")
        return dict(self.history)

    # ── Evaluación en test ───────────────────────────────────────────────

    def evaluate_test(self, checkpoint_path: Optional[str] = None) -> Dict[str, float]:
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state'])
            print(f"Checkpoint cargado: {checkpoint_path}")

        with torch.no_grad():
            test_metrics = self._run_epoch(self.loaders['test'], train=False)

        print("\n=== EVALUACIÓN EN TEST ===")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")
        return test_metrics

    # ── Exportación ONNX ─────────────────────────────────────────────────

    def export_onnx(
        self,
        checkpoint_path: str,
        output_path: str,
        opset_version: int = 17,
    ) -> None:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval().cpu()

        # Crear input dummy según modo
        dummy_inputs = {}
        if self.mode in ('pixels', 'both'):
            dummy_inputs['pixels'] = torch.randn(1, 3, 30, 224, 224)
        if self.mode in ('landmarks', 'both'):
            dummy_inputs['landmarks'] = torch.randn(1, 30, 75, 3)

        if self.mode == 'pixels':
            dummy = (dummy_inputs['pixels'],)
            input_names = ['pixels']
        elif self.mode == 'landmarks':
            dummy = (dummy_inputs['landmarks'],)
            input_names = ['landmarks']
        else:
            dummy = (dummy_inputs['pixels'], dummy_inputs['landmarks'])
            input_names = ['pixels', 'landmarks']

        torch.onnx.export(
            self.model,
            dummy,
            output_path,
            input_names=input_names,
            output_names=['logits'],
            dynamic_axes={name: {0: 'batch_size'} for name in input_names},
            opset_version=opset_version,
        )
        print(f"Modelo exportado a ONNX: {output_path}")
