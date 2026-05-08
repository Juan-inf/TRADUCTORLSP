"""
Métricas de evaluación para el sistema LSP.
"""

import numpy as np
from typing import Dict, Optional
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, top_k_accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
    n_classes: Optional[int] = None,
) -> Dict[str, float]:
    """
    Calcula métricas principales de clasificación.

    Retorna dict con: accuracy, f1_macro, f1_weighted, top5_acc (si scores disponibles)
    """
    metrics = {
        'accuracy':    float(accuracy_score(y_true, y_pred)),
        'f1_macro':    float(f1_score(y_true, y_pred, average='macro',    zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }

    if y_scores is not None and n_classes is not None:
        try:
            metrics['top5_acc'] = float(
                top_k_accuracy_score(y_true, y_scores, k=min(5, n_classes))
            )
        except Exception:
            pass

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    output_path: Optional[str] = None,
    normalize: bool = True,
    figsize: tuple = (14, 12),
) -> plt.Figure:
    """Genera y guarda la matriz de confusión normalizada."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt='.2f' if normalize else 'd',
        cmap='Blues', xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel('Predicción', fontsize=12)
    ax.set_ylabel('Verdadero', fontsize=12)
    ax.set_title('Matriz de Confusión — LSP', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_training_curves(
    history: dict,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Curvas de loss y accuracy durante el entrenamiento."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history.get('train_loss', []), label='Train', linewidth=2)
    axes[0].plot(history.get('val_loss',   []), label='Val',   linewidth=2, linestyle='--')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Curva de Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history.get('train_accuracy', []), label='Train', linewidth=2)
    axes[1].plot(history.get('val_accuracy',   []), label='Val',   linewidth=2, linestyle='--')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Curva de Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Entrenamiento LSP', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def compute_wer(references: list, hypotheses: list) -> float:
    """
    Word Error Rate para secuencias de señas.
    Cada elemento puede ser una cadena de palabras separadas por espacio.
    """
    total_words = 0
    total_errors = 0

    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.strip().split()
        hyp_words = hyp.strip().split()
        total_words += len(ref_words)

        # Distancia de edición (Levenshtein) entre secuencias de palabras
        d = _edit_distance(ref_words, hyp_words)
        total_errors += d

    return total_errors / max(total_words, 1)


def _edit_distance(seq1: list, seq2: list) -> int:
    m, n = len(seq1), len(seq2)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    dp[:, 0] = np.arange(m + 1)
    dp[0, :] = np.arange(n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i-1] == seq2[j-1] else 1
            dp[i, j] = min(dp[i-1, j] + 1, dp[i, j-1] + 1, dp[i-1, j-1] + cost)

    return int(dp[m, n])
