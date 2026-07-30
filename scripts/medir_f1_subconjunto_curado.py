"""medir_f1_subconjunto_curado.py — Mide el F1-macro REAL del checkpoint
activo (bilstm_s27.onnx / v4) restringido a subconjuntos de clases con
suficientes muestras de entrenamiento, sobre el MISMO test holdout usado
para reportar F1=0.4426 (mismo dataset_s17.npz, mismo filtro min15, mismo
StratifiedShuffleSplit(test_size=0.15, seed=42) que train_s27.py).

No es un modelo nuevo ni un reentrenamiento: es una medición honesta de
"¿qué F1 real tiene el modelo YA entrenado si el alcance se acota a las
clases mejor representadas?" — opción #5 de las discutidas (recorte de
alcance), la de menor riesgo técnico y verificable en minutos.

Uso:
    python3 scripts/medir_f1_subconjunto_curado.py
"""
import json
import pathlib
import collections

import numpy as np
import onnxruntime as ort
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SEED = 42
MIN_SAMPLES = 15


def normalize_sample(x):
    mu, std = x.mean(), x.std()
    if std < 1e-8:
        return x
    return ((x - mu) / std).astype(np.float32)


def normalize_batch(X):
    return np.stack([normalize_sample(X[i]) for i in range(len(X))])


def main():
    data = np.load(DATA / "dataset_s17.npz")
    X_raw, y_raw = data["X"], data["y"]

    with open(DATA / "s17_label2idx.json", encoding="utf-8") as f:
        label2idx = json.load(f)
    idx2label = {v: k for k, v in label2idx.items()}

    counts = collections.Counter(y_raw.tolist())
    keep = np.array([counts[int(v)] >= MIN_SAMPLES for v in y_raw])
    X_raw, y_raw = X_raw[keep], y_raw[keep]
    print(f"Dataset S17 tras min{MIN_SAMPLES}: {len(X_raw)} muestras, {len(set(y_raw.tolist()))} clases")

    X = normalize_batch(X_raw)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    tv_idx, te_idx = next(sss.split(X, y_raw))
    X_tv, y_tv = X[tv_idx], y_raw[tv_idx]
    X_te, y_te = X[te_idx], y_raw[te_idx]
    print(f"Train+Val: {len(X_tv)}  |  Test holdout: {len(X_te)} (mismo split que reportó F1=0.4426)")

    train_counts = collections.Counter(y_tv.tolist())

    # Cargar checkpoint activo (v4) y correr inferencia sobre el test set
    onnx_path = ROOT / "checkpoints" / "bilstm_s27.onnx"
    l2i_s27 = json.load(open(DATA / "s27_label2idx.json", encoding="utf-8"))
    i2l_s27 = {v: k for k, v in l2i_s27.items()}
    # dataset_s17 usa las mismas clases/orden que s27 (mismo label2idx) — verificar
    assert label2idx == l2i_s27, "label2idx de s17 y s27 no coinciden — el split no es reproducible tal cual"

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    logits = sess.run(None, {"sequence": X_te.astype(np.float32)})[0]
    preds = logits.argmax(axis=1)

    f1_full = f1_score(y_te, preds, average="macro", zero_division=0)
    print(f"\nF1-macro sobre las {len(set(y_te.tolist()))} clases completas (verificación): {f1_full:.4f}")
    print("(el valor reportado en la tesis es 0.4426 con normalize_input embebido en el checkpoint; "
          "aquí se aplica normalize_sample manualmente antes de exportar a ONNX — puede haber una "
          "pequeña diferencia de precisión numérica, no de metodología)")

    # F1 por clase
    f1_per_class = f1_score(y_te, preds, average=None, zero_division=0, labels=sorted(set(y_te.tolist())))
    clases_ordenadas = sorted(set(y_te.tolist()))
    f1_dict = dict(zip(clases_ordenadas, f1_per_class))

    print("\n" + "=" * 78)
    print("F1-macro restringido a subconjuntos por umbral de muestras de entrenamiento")
    print("=" * 78)
    print(f"{'Umbral muestras/clase':<25}{'N clases':<12}{'N muestras test':<18}{'% del test':<12}{'F1-macro':<10}")

    for umbral in [0, 15, 30, 50, 75, 100, 150, 200]:
        clases_validas = [c for c, n in train_counts.items() if n >= umbral]
        mask_te = np.isin(y_te, clases_validas)
        if mask_te.sum() == 0:
            continue
        y_sub = y_te[mask_te]
        pred_sub = preds[mask_te]
        clases_en_test = sorted(set(y_sub.tolist()))
        f1_sub = f1_score(y_sub, pred_sub, average="macro", zero_division=0, labels=clases_en_test)
        pct = 100 * mask_te.sum() / len(y_te)
        print(f"{umbral:<25}{len(clases_en_test):<12}{mask_te.sum():<18}{pct:<12.1f}{f1_sub:<10.4f}")

    # Detalle del subconjunto que cruza F1>=0.70, si existe
    print("\nClases individuales con F1 >= 0.70 en el test holdout completo:")
    altas = sorted([(idx2label.get(c, c), f1_dict[c], train_counts.get(c, 0))
                     for c in clases_ordenadas if f1_dict[c] >= 0.70],
                    key=lambda t: -t[1])
    for nombre, f1v, n_tr in altas:
        print(f"  {nombre:<20} F1={f1v:.3f}  (muestras train: {n_tr})")
    print(f"\nTotal clases con F1>=0.70: {len(altas)} de {len(clases_ordenadas)}")


if __name__ == "__main__":
    main()
