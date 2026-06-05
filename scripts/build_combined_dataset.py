"""
build_combined_dataset.py — Combina Glosas PKL + Keypoints/pkl en dataset NPZ.
Features: pose(33×2) + left_hand(21×2) + right_hand(21×2) = 150 dims/frame × T=30
Salida: data/dataset_lstm.npz  +  data/lstm_label2idx.json
"""

import pickle, json, re, pathlib, warnings
import numpy as np
from collections import Counter

warnings.filterwarnings("ignore")

ROOT      = pathlib.Path(__file__).parent.parent
OUT_DIR   = ROOT / "data"
N_FRAMES  = 30
N_DIMS    = 150  # pose(66) + left_hand(42) + right_hand(42)

PKL_SOURCES = [
    ROOT / "data" / "Keypoints" / "pkl",          # 3684 muestras, 1086 clases
    ROOT / "data" / "Keypoints" / "glosas_pkl",   # ~263 muestras, ~143 clases
]


def label_from_filename(name):
    stem = pathlib.Path(name).stem
    m = re.match(r"^(.+)_(\d+)$", stem)
    return m.group(1).upper() if m else stem.upper()


def pkl_to_sequence(pkl_path):
    """PKL → array [T=30, 150]. Incluye pose + ambas manos."""
    try:
        with open(pkl_path, "rb") as f:
            frames = pickle.load(f)
        if not frames:
            return None

        seq = []
        for fr in frames:
            row = []
            for key, n in (("pose", 33), ("left_hand", 21), ("right_hand", 21)):
                d = fr.get(key, {})
                x = d.get("x", [])
                y = d.get("y", [])
                if len(x) != n:
                    x, y = [0.0] * n, [0.0] * n
                row.extend(x)
                row.extend(y)
            seq.append(row)  # 150 dims

        # Resamplear / pad a N_FRAMES
        if len(seq) > N_FRAMES:
            idx = np.linspace(0, len(seq) - 1, N_FRAMES, dtype=int)
            seq = [seq[i] for i in idx]
        while len(seq) < N_FRAMES:
            seq.append(seq[-1] if seq else [0.0] * N_DIMS)

        return np.array(seq[:N_FRAMES], dtype=np.float32)
    except Exception:
        return None


# ── Cargar todas las fuentes ──────────────────────────────────────────────────

print("Cargando PKL de todas las fuentes...")
X_list, y_list = [], []

for src_dir in PKL_SOURCES:
    if not src_dir.exists():
        print(f"  ⚠️  {src_dir} no existe, saltando")
        continue
    pkls = sorted(src_dir.rglob("*.pkl"))
    loaded = 0
    for p in pkls:
        seq = pkl_to_sequence(p)
        if seq is None:
            continue
        label = label_from_filename(p.name)
        X_list.append(seq)
        y_list.append(label)
        loaded += 1
    print(f"  {src_dir.name}: {loaded} muestras cargadas")

print(f"\nTotal muestras: {len(X_list)}")
print(f"Clases únicas:  {len(set(y_list))}")

# Distribución
conteo = Counter(y_list)
min_c, max_c = min(conteo.values()), max(conteo.values())
mean_c = np.mean(list(conteo.values()))
print(f"Muestras/clase: min={min_c}  max={max_c}  media={mean_c:.1f}")

# ── Construir arrays y mapeo ──────────────────────────────────────────────────

classes   = sorted(set(y_list))
label2idx = {c: i for i, c in enumerate(classes)}
idx2label = {i: c for c, i in label2idx.items()}

X = np.array(X_list, dtype=np.float32)  # [N, 30, 150]
y = np.array([label2idx[l] for l in y_list], dtype=np.int64)

print(f"\nX shape: {X.shape}  y shape: {y.shape}")
print(f"Ejemplo clases: {classes[:8]}")

# ── Guardar ───────────────────────────────────────────────────────────────────

npz_path = OUT_DIR / "dataset_lstm.npz"
np.savez_compressed(npz_path, X=X, y=y)
print(f"\n✅ {npz_path}  ({npz_path.stat().st_size/1e6:.1f} MB)")

with open(OUT_DIR / "lstm_label2idx.json", "w", encoding="utf-8") as f:
    json.dump(label2idx, f, ensure_ascii=False, indent=2)
print(f"✅ data/lstm_label2idx.json  ({len(label2idx)} clases)")
