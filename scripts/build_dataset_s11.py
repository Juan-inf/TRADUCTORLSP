"""
build_dataset_s11.py — Dataset Sprint 11

Mejoras vs S10 (build_dataset_s10.py):
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Cambio             S10 (build_s10)     S11 (build_s11)                │
  │  Abecedario cap     10 / clase          SIN CAP → 150 / clase          │
  │  Total muestras     4,176               7,536  (+80 %)                 │
  │  Normalización      ninguna             strip + upper + alias           │
  │  Source tag         str group           int + source_name array         │
  │  Outputs            1 npz               1 npz + label/group JSONs       │
  └─────────────────────────────────────────────────────────────────────────┘

Outputs:
  data/dataset_s11.npz        X [N,30,150], y [N], groups [N]
  data/s11_label2idx.json
  data/s11_idx2label.json
  data/s11_groups.json
  data/s11_sources.npy        (source string per sample, para análisis)
"""

import pickle, json, re, pathlib, warnings
import numpy as np
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")

ROOT     = pathlib.Path(__file__).parent.parent
OUT_DIR  = ROOT / "data"
N_FRAMES = 30
N_DIMS   = 150   # pose_x/y (66) + left_hand_x/y (42) + right_hand_x/y (42)
SEED     = 42

# ── Fuentes de datos ──────────────────────────────────────────────────────────
# (directorio, nombre_fuente, es_abecedario)
PKL_SOURCES = [
    (ROOT / "data" / "Keypoints" / "pkl",            "vineta",      False),
    (ROOT / "data" / "Keypoints" / "glosas_pkl",     "glosa",       False),
    (ROOT / "data" / "Keypoints" / "abecedario_pkl", "abecedario",  True),
]

# Alias de normalización: nombres que deben unificarse
LABEL_ALIAS = {
    "AHI":  "AHÍ",
    "QUE?": "QUÉ?",
    "SI":   "SÍ",
    "TU":   "TÚ",
    "MAS":  "MÁS",
    "VIO":  "VIÓ",
    "MA":   "MÁ",
}


def normalize_label(raw: str) -> str:
    s = raw.strip().upper()
    return LABEL_ALIAS.get(s, s)


def label_from_filename(name: str) -> str:
    stem = pathlib.Path(name).stem
    m = re.match(r"^(.+)_(\d+)$", stem)
    raw = m.group(1) if m else stem
    return normalize_label(raw)


def pkl_to_sequence(pkl_path: pathlib.Path) -> np.ndarray | None:
    """PKL → array float32 [30, 150]. Resamplea / padea si hace falta."""
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

        if len(seq) > N_FRAMES:
            idx = np.linspace(0, len(seq) - 1, N_FRAMES, dtype=int)
            seq = [seq[i] for i in idx]
        while len(seq) < N_FRAMES:
            seq.append(seq[-1] if seq else [0.0] * N_DIMS)

        return np.array(seq[:N_FRAMES], dtype=np.float32)
    except Exception:
        return None


# ── Carga de todos los PKL ───────────────────────────────────────────────────

print("=" * 65)
print("build_dataset_s11.py — Sprint 11: Dataset completo sin caps")
print("=" * 65)

X_list, y_list, group_list, source_list = [], [], [], []

for src_dir, src_name, is_abc in PKL_SOURCES:
    if not src_dir.exists():
        print(f"  AVISO: {src_dir} no existe, saltando")
        continue

    pkls  = sorted(src_dir.rglob("*.pkl"))
    loaded = skipped = errors = 0

    for p in pkls:
        label = p.parent.name.upper() if is_abc else label_from_filename(p.name)
        label = normalize_label(label)

        seq = pkl_to_sequence(p)
        if seq is None:
            errors += 1
            continue

        group_id = f"{src_name}/{label}"
        X_list.append(seq)
        y_list.append(label)
        group_list.append(group_id)
        source_list.append(src_name)
        loaded += 1

    print(f"  {src_name:<12}: {loaded:>5} muestras  "
          f"({errors} errores)")

print(f"\n  Total muestras cargadas : {len(X_list)}")
print(f"  LSP - Vocabulario-palabras únicas (raw)     : {len(set(y_list))}")

# ── Construir arrays ──────────────────────────────────────────────────────────

counts  = Counter(y_list)
classes = sorted(set(y_list))
label2idx = {c: i for i, c in enumerate(classes)}
idx2label = {i: c for c, i in label2idx.items()}

X      = np.array(X_list, dtype=np.float32)          # [N, 30, 150]
y_raw  = np.array([label2idx[l] for l in y_list], dtype=np.int64)
sources = np.array(source_list)

# Convertir grupos a enteros
unique_groups = sorted(set(group_list))
group2int     = {g: i for i, g in enumerate(unique_groups)}
groups_int    = np.array([group2int[g] for g in group_list], dtype=np.int32)

# ── Filtrar LSP - Vocabulario-palabras con < 2 muestras ──────────────────────────────────────────

counts_idx = Counter(y_raw.tolist())
keep_mask  = np.array([counts_idx[int(v)] >= 2 for v in y_raw])
X          = X[keep_mask]
y_raw      = y_raw[keep_mask]
groups_int = groups_int[keep_mask]
sources    = sources[keep_mask]

# Re-mapear índices continuos
old_ids   = sorted(set(y_raw.tolist()))
remap     = {old: new for new, old in enumerate(old_ids)}
y         = np.array([remap[int(v)] for v in y_raw], dtype=np.int64)
n_classes = len(old_ids)
idx2label = {remap[old]: idx2label[old] for old in old_ids}
label2idx = {v: k for k, v in idx2label.items()}

# Re-hacer grupos después del filtrado
y_list_f      = [idx2label[int(v)] for v in y]
group_list_f  = [f"{source_list_f}/{lbl}"
                 for source_list_f, lbl in zip(sources.tolist(), y_list_f)]
unique_groups2 = sorted(set(group_list_f))
g2int2         = {g: i for i, g in enumerate(unique_groups2)}
groups_int     = np.array([g2int2[g] for g in group_list_f], dtype=np.int32)

# ── Estadísticas ──────────────────────────────────────────────────────────────

counts_final = Counter(y.tolist())
vals = sorted(counts_final.values(), reverse=True)

print(f"\n  ── Estadísticas S11 ──")
print(f"  Muestras (≥2/clase) : {len(X)}")
print(f"  LSP - Vocabulario-palabras activas      : {n_classes}")
print(f"  Muestras/clase: min={min(vals)}  max={max(vals)}  "
      f"media={np.mean(vals):.1f}  mediana={np.median(vals):.0f}")
print(f"  LSP - Vocabulario-palabras con ≥2   : {sum(1 for v in vals if v >= 2)}")
print(f"  LSP - Vocabulario-palabras con ≥5   : {sum(1 for v in vals if v >= 5)}")
print(f"  LSP - Vocabulario-palabras con ≥10  : {sum(1 for v in vals if v >= 10)}")
print(f"  LSP - Vocabulario-palabras con ≥50  : {sum(1 for v in vals if v >= 50)}")
print(f"  LSP - Vocabulario-palabras con ≥100 : {sum(1 for v in vals if v >= 100)}")

print(f"\n  ── Por fuente ──")
for src in ["vineta", "glosa", "abecedario"]:
    mask  = sources == src
    n_src = int(mask.sum())
    cls_s = len(set(y[mask].tolist()))
    print(f"  {src:<12}: {n_src:>5} muestras  {cls_s:>4} LSP - Vocabulario-palabras")

print(f"\n  ── Comparativa S10 vs S11 ──")
print(f"  S10: 4,176 muestras  482 LSP - Vocabulario-palabras activas  3.6 media/clase")
print(f"  S11: {len(X):,} muestras  {n_classes} LSP - Vocabulario-palabras activas  {np.mean(vals):.1f} media/clase")
pct = (len(X) / 4176 - 1) * 100
print(f"  Incremento muestras: +{pct:.0f}%")

# ── Guardar ───────────────────────────────────────────────────────────────────

npz_path = OUT_DIR / "dataset_s11.npz"
np.savez_compressed(npz_path, X=X, y=y, groups=groups_int)
print(f"\n✅ {npz_path}  ({npz_path.stat().st_size / 1e6:.1f} MB)")
print(f"   X={X.shape}  y={y.shape}  groups={groups_int.shape}")

json_path = OUT_DIR / "s11_label2idx.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(label2idx, f, ensure_ascii=False, indent=2)
print(f"✅ {json_path}  ({len(label2idx)} LSP - Vocabulario-palabras)")

idx2label_path = OUT_DIR / "s11_idx2label.json"
with open(idx2label_path, "w", encoding="utf-8") as f:
    json.dump({str(k): v for k, v in idx2label.items()}, f,
              ensure_ascii=False, indent=2)
print(f"✅ {idx2label_path}")

gmap_path = OUT_DIR / "s11_groups.json"
with open(gmap_path, "w", encoding="utf-8") as f:
    json.dump({
        "int2group": {str(v): k for k, v in g2int2.items()},
        "n_groups":  len(unique_groups2),
    }, f, ensure_ascii=False, indent=2)
print(f"✅ {gmap_path}  ({len(unique_groups2)} grupos)")

np.save(OUT_DIR / "s11_sources.npy", sources)
print(f"✅ {OUT_DIR / 's11_sources.npy'}")

print("\n[BUILD S11 COMPLETADO]")
