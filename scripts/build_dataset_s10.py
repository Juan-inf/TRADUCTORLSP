"""
build_dataset_s10.py — Dataset Sprint 10 con cap de abecedario y group IDs.

Cambios vs build_combined_dataset.py (S9):
  - Abecedario limitado a CAP_ABC muestras/clase (default 10, de 150 en S9)
  - Array 'groups' guardado en el NPZ para StratifiedKFold sin leakage
  - Salida: data/dataset_s10.npz  +  data/s10_label2idx.json
"""

import pickle, json, re, pathlib, warnings
import numpy as np
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")

ROOT     = pathlib.Path(__file__).parent.parent
OUT_DIR  = ROOT / "data"
N_FRAMES = 30
N_DIMS   = 150  # pose(66) + left_hand(42) + right_hand(42)
CAP_ABC  = 10   # máximo de muestras por clase del abecedario (era 150 en S9)

PKL_SOURCES = [
    (ROOT / "data" / "Keypoints" / "pkl",            "pkl",        False),
    (ROOT / "data" / "Keypoints" / "glosas_pkl",     "glosas",     False),
    (ROOT / "data" / "Keypoints" / "abecedario_pkl", "abecedario", True),
]


def label_from_filename(name: str) -> str:
    stem = pathlib.Path(name).stem
    m = re.match(r"^(.+)_(\d+)$", stem)
    return m.group(1).upper() if m else stem.upper()


def pkl_to_sequence(pkl_path: pathlib.Path) -> np.ndarray | None:
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
            seq.append(row)

        if len(seq) > N_FRAMES:
            idx = np.linspace(0, len(seq) - 1, N_FRAMES, dtype=int)
            seq = [seq[i] for i in idx]
        while len(seq) < N_FRAMES:
            seq.append(seq[-1] if seq else [0.0] * N_DIMS)

        return np.array(seq[:N_FRAMES], dtype=np.float32)
    except Exception:
        return None


print("=== build_dataset_s10.py ===")
print(f"  Cap abecedario: {CAP_ABC} muestras/clase  (S9 era 150)")
print()

X_list, y_list, group_list = [], [], []
abc_counter: dict[str, int] = defaultdict(int)

for src_dir, src_name, is_abc in PKL_SOURCES:
    if not src_dir.exists():
        print(f"  AVISO: {src_dir} no existe, saltando")
        continue

    pkls = sorted(src_dir.rglob("*.pkl"))
    loaded = skipped = 0
    for p in pkls:
        label = p.parent.name.upper() if is_abc else label_from_filename(p.name)

        if is_abc:
            if abc_counter[label] >= CAP_ABC:
                skipped += 1
                continue
            abc_counter[label] += 1

        seq = pkl_to_sequence(p)
        if seq is None:
            continue

        group_id = f"{src_name}/{label}"  # grupo = fuente + clase
        X_list.append(seq)
        y_list.append(label)
        group_list.append(group_id)
        loaded += 1

    if is_abc:
        print(f"  {src_name}: {loaded} muestras cargadas  "
              f"({skipped} omitidas por cap={CAP_ABC})")
    else:
        print(f"  {src_name}: {loaded} muestras cargadas")

print(f"\n  Total muestras : {len(X_list)}")
print(f"  LSP - Vocabulario-palabras únicas  : {len(set(y_list))}")

# ── Construir arrays ──────────────────────────────────────────────────────────

counts  = Counter(y_list)
classes = sorted(set(y_list))

label2idx = {c: i for i, c in enumerate(classes)}
idx2label = {i: c for c, i in label2idx.items()}

X      = np.array(X_list, dtype=np.float32)                    # [N, 30, 150]
y      = np.array([label2idx[l] for l in y_list], dtype=np.int64)
groups = np.array(group_list)                                   # [N] str

# Convertir groups a enteros para sklearn
unique_groups   = sorted(set(group_list))
group2int       = {g: i for i, g in enumerate(unique_groups)}
groups_int      = np.array([group2int[g] for g in group_list], dtype=np.int32)

vals = sorted(counts.values(), reverse=True)
print(f"  Muestras/clase: min={min(vals)}  max={max(vals)}  media={np.mean(vals):.1f}")
print(f"  LSP - Vocabulario-palabras con ≥2  : {sum(1 for v in vals if v >= 2)}")
print(f"  LSP - Vocabulario-palabras con ≥5  : {sum(1 for v in vals if v >= 5)}")

# Distribución por fuente
for src_name in ["pkl", "glosas", "abecedario"]:
    mask  = np.array([g.startswith(src_name) for g in group_list])
    n_src = int(mask.sum())
    cls_src = len(set(np.array(y_list)[mask]))
    print(f"  {src_name:<12}: {n_src:>4} muestras  {cls_src:>4} LSP - Vocabulario-palabras")

# ── Guardar ───────────────────────────────────────────────────────────────────

npz_path = OUT_DIR / "dataset_s10.npz"
np.savez_compressed(npz_path, X=X, y=y, groups=groups_int)
print(f"\n  X shape: {X.shape}  |  y shape: {y.shape}  |  groups: {groups_int.shape}")
print(f"\n✅ {npz_path}  ({npz_path.stat().st_size / 1e6:.1f} MB)")

json_path = OUT_DIR / "s10_label2idx.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(label2idx, f, ensure_ascii=False, indent=2)
print(f"✅ {json_path}  ({len(label2idx)} LSP - Vocabulario-palabras)")

# Guardar también groups mapping
gmap_path = OUT_DIR / "s10_groups.json"
with open(gmap_path, "w", encoding="utf-8") as f:
    json.dump({"int2group": {v: k for k, v in group2int.items()},
               "n_groups": len(unique_groups)}, f, indent=2)
print(f"✅ {gmap_path}  ({len(unique_groups)} grupos)")

print("\n[BUILD S10 COMPLETADO]")
