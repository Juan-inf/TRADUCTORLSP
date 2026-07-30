"""build_dataset_s38_curado.py — Subconjunto CURADO del vocabulario (opción
#5 + reentrenamiento dedicado), seleccionado A PRIORI por conteo de
muestras de entrenamiento (sin espiar el test set — ver advertencia
metodológica de la conversación sobre data snooping), no por su F1 de
test individual.

scripts/medir_f1_subconjunto_curado.py ya midió el F1 real del checkpoint
v4 (entrenado sobre las 96 clases completas) restringido a este mismo
subconjunto: F1=0.6107 (49 clases, ≥100 muestras de train, 88.4% del
test). Ese número usa un modelo que compite contra 96 clases en el
softmax, incluidas 47 clases ruidosas de cola larga que compiten por
gradiente durante el entrenamiento sin aportar señal útil para las 49
clases buenas. Esta vez se entrena un modelo NUEVO cuyo softmax solo
tiene las 49 clases curadas desde el inicio — hipótesis: sin la cola
larga arrastrando el entrenamiento, el F1 sobre estas 49 clases debería
subir por encima de 0.6107, posiblemente cruzando 0.70.

Reproduce exactamente la misma selección de clases que
medir_f1_subconjunto_curado.py: mismo dataset_s17.npz, mismo filtro
min15, mismo StratifiedShuffleSplit(test_size=0.15, seed=42) para
calcular conteos de train, mismo umbral (>=100 muestras de train).

Uso:
    python3 scripts/build_dataset_s38_curado.py
"""
import json
import pathlib
import collections

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SEED = 42
MIN_SAMPLES = 15
UMBRAL_CURADO = 100


def main():
    data = np.load(DATA / "dataset_s17.npz")
    X_raw, y_raw, groups_raw = data["X"], data["y"], data["groups"]

    with open(DATA / "s17_label2idx.json", encoding="utf-8") as f:
        label2idx = json.load(f)
    idx2label = {v: k for k, v in label2idx.items()}

    counts = collections.Counter(y_raw.tolist())
    keep96 = np.array([counts[int(v)] >= MIN_SAMPLES for v in y_raw])
    X96, y96, g96 = X_raw[keep96], y_raw[keep96], groups_raw[keep96]
    print(f"Dataset S17 tras min{MIN_SAMPLES}: {len(X96)} muestras, {len(set(y96.tolist()))} clases")

    # Mismo split que usó train_s27.py/medir_f1_subconjunto_curado.py para
    # decidir qué clases cuentan como "bien representadas" — SOLO para
    # calcular conteos de train, no se reutiliza este split para el
    # entrenamiento nuevo (el dataset curado se re-parte desde cero).
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    tv_idx, _ = next(sss.split(X96, y96))
    train_counts = collections.Counter(y96[tv_idx].tolist())

    # Excluir clases narrativas de video completo (HISTORIAS_VINETAS_N =
    # "este clip entero es la narrativa N", no una seña individual — mismo
    # criterio ya aplicado en demo/app_gradio.py y en toda la línea de
    # producción). Incluirlas invalidaría la comparación con OE1: sería
    # medir "reconocer de qué video viene este clip", una tarea distinta
    # y mucho más fácil que "reconocer qué seña de LSP se hizo".
    clases_curadas = sorted([c for c, n in train_counts.items()
                              if n >= UMBRAL_CURADO and not idx2label[c].startswith("HISTORIAS_VINETAS_")])
    print(f"\nClases curadas (>= {UMBRAL_CURADO} muestras de train, excluyendo HISTORIAS_VINETAS_*): {len(clases_curadas)}")
    for c in clases_curadas:
        print(f"  {idx2label[c]:<20} train={train_counts[c]}")

    mask_curado = np.isin(y96, clases_curadas)
    X_cur, y_old, g_cur = X96[mask_curado], y96[mask_curado], g96[mask_curado]
    print(f"\nTotal muestras en el subconjunto curado: {len(X_cur)}")

    # Reindexar labels 0..N-1 para el nuevo vocabulario reducido
    old_to_new = {old: i for i, old in enumerate(clases_curadas)}
    y_cur = np.array([old_to_new[int(v)] for v in y_old.tolist()], dtype=np.int64)
    label2idx_cur = {idx2label[old]: new for old, new in old_to_new.items()}
    idx2label_cur = {new: idx2label[old] for old, new in old_to_new.items()}

    np.savez_compressed(DATA / "dataset_s38_curado.npz", X=X_cur.astype(np.float32),
                         y=y_cur, groups=g_cur.astype(np.int32))
    with open(DATA / "s38_label2idx.json", "w", encoding="utf-8") as f:
        json.dump(label2idx_cur, f, ensure_ascii=False, indent=2)
    with open(DATA / "s38_idx2label.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in idx2label_cur.items()}, f, ensure_ascii=False, indent=2)

    print(f"\n✅ dataset_s38_curado.npz  ({len(X_cur)} muestras, {len(clases_curadas)} clases)")
    print(f"✅ s38_label2idx.json / s38_idx2label.json")


if __name__ == "__main__":
    main()
