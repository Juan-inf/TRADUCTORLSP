"""evaluar_ensemble_s40.py — ¿El ensemble v4 + S40-finetune supera a v4 solo
en el vocabulario de 96 clases? Mismo principio que el ensemble de
producción (v4+S29) y el de narración continua (S33+S36): promediar
softmax de dos checkpoints con fortalezas distintas.

v4: F1=0.4426 (entrenado desde cero, 27+ sprints de tuning)
S40-finetune: F1=0.3957 (transferencia desde bilstm_s36.pt), pero mejor
generalización (ΔF1=0.0086 vs 0.0406 de v4).

Evalúa sobre el MISMO test holdout que reportó ambos números (dataset_s17,
StratifiedShuffleSplit seed=42, test=15%).

Uso:
    python3 scripts/evaluar_ensemble_s40.py
"""
import json
import pathlib
import unicodedata

import numpy as np
import onnxruntime as ort
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SEED = 42
MIN_SAMPLES = 15


def nfc(s):
    return unicodedata.normalize("NFC", s)


def normalize_sample(x):
    mu, std = x.mean(), x.std()
    if std < 1e-8:
        return x
    return ((x - mu) / std).astype(np.float32)


def normalize_batch(X):
    return np.stack([normalize_sample(X[i]) for i in range(len(X))])


def softmax(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main():
    data = np.load(DATA / "dataset_s17.npz")
    X_raw, y_raw = data["X"], data["y"]
    with open(DATA / "s17_label2idx.json", encoding="utf-8") as f:
        label2idx = json.load(f)

    counts = Counter(y_raw.tolist())
    keep = np.array([counts[int(v)] >= MIN_SAMPLES for v in y_raw])
    X_raw, y_raw = X_raw[keep], y_raw[keep]
    X = normalize_batch(X_raw)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    _, te_idx = next(sss.split(X, y_raw))
    X_te, y_te = X[te_idx], y_raw[te_idx]
    print(f"Test holdout: {len(X_te)} muestras (mismo split que v4=0.4426 y S40=0.3957)")

    sess_v4 = ort.InferenceSession(str(ROOT / "checkpoints/bilstm_s27.onnx"), providers=["CPUExecutionProvider"])
    sess_s40 = ort.InferenceSession(str(ROOT / "checkpoints/bilstm_s40_finetune.onnx"), providers=["CPUExecutionProvider"])

    # dataset_s17 y s27/s40 comparten el mismo label2idx (mismo vocabulario de 96 clases)
    l2i_s27 = json.load(open(DATA / "s27_label2idx.json", encoding="utf-8"))
    assert label2idx == l2i_s27, "labels de s17/s27 no coinciden"

    logits_v4 = sess_v4.run(None, {"sequence": X_te.astype(np.float32)})[0]
    logits_s40 = sess_s40.run(None, {"sequence": X_te.astype(np.float32)})[0]

    probs_v4 = softmax(logits_v4)
    probs_s40 = softmax(logits_s40)

    preds_v4 = probs_v4.argmax(axis=1)
    preds_s40 = probs_s40.argmax(axis=1)
    f1_v4 = f1_score(y_te, preds_v4, average="macro", zero_division=0)
    f1_s40 = f1_score(y_te, preds_s40, average="macro", zero_division=0)
    print(f"F1 v4 solo          : {f1_v4:.4f}")
    print(f"F1 S40-finetune solo: {f1_s40:.4f}")

    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        probs_ens = w * probs_v4 + (1 - w) * probs_s40
        preds_ens = probs_ens.argmax(axis=1)
        f1_ens = f1_score(y_te, preds_ens, average="macro", zero_division=0)
        print(f"Ensemble (w_v4={w:.1f}, w_s40={1-w:.1f}): F1={f1_ens:.4f}")


if __name__ == "__main__":
    main()
