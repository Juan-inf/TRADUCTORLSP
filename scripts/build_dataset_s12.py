"""
build_dataset_s12.py — Dataset Sprint 12

Extiende S11 incorporando las fuentes PUCP descargadas por download_pucp_datasets.py.

Mejoras vs S11 (build_dataset_s11.py):
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Cambio              S11                    S12                          │
  │  Fuentes             vineta+glosa+abc        + vocabulario_lsp_p + dgi156          │
  │  Muestras esperadas  6,855                   ~10,000–14,000              │
  │  Clases esperadas    479                      ~500–600                   │
  │  Media/clase         14.3                     ~20–30                     │
  │  Groups              por fuente/clase         por señante_id             │
  └──────────────────────────────────────────────────────────────────────────┘

Outputs:
  data/dataset_s12.npz          X [N,30,150], y [N], groups [N]
  data/s12_label2idx.json
  data/s12_idx2label.json
  data/s12_sources.npy

Uso:
  .venv310/bin/python3 scripts/build_dataset_s12.py
  .venv310/bin/python3 scripts/build_dataset_s12.py --min-muestras 3
  .venv310/bin/python3 scripts/build_dataset_s12.py --solo-stats   # sin guardar
"""

import argparse
import json
import pathlib
import pickle
import re
import warnings
from collections import Counter, defaultdict

import numpy as np

warnings.filterwarnings("ignore")

ROOT     = pathlib.Path(__file__).parent.parent
OUT_DIR  = ROOT / "data"
N_FRAMES = 30
N_DIMS   = 150
SEED     = 42

# ── Fuentes de datos ──────────────────────────────────────────────────────────
# (directorio_pkl, nombre_fuente, es_abecedario)
# Las fuentes PUCP se agregan automáticamente si el directorio existe.
PKL_SOURCES_BASE = [
    (ROOT / "data" / "Keypoints" / "pkl",            "vineta",    False),
    (ROOT / "data" / "Keypoints" / "glosas_pkl",     "glosa",     False),
    (ROOT / "data" / "Keypoints" / "abecedario_pkl", "abecedario", True),
]

PKL_SOURCES_PUCP = [
    (ROOT / "data" / "Keypoints" / "vocabulario_lsp_p_pkl",   "vocabulario_lsp_p",   False),
    (ROOT / "data" / "Keypoints" / "dgi156_pkl",    "dgi156",    False),
]

# ── Normalización de etiquetas ────────────────────────────────────────────────

LABEL_ALIAS = {
    "AHI":  "AHÍ",  "QUE?": "QUÉ?", "SI":  "SÍ",  "TU":  "TÚ",
    "MAS":  "MÁS",  "VIO":  "VIÓ",  "MA":  "MÁ",
    # Variantes tipográficas comunes en nombres de archivo
    "COMO": "CÓMO", "DONDE": "DÓNDE", "CUANDO": "CUÁNDO",
    "QUE":  "QUÉ",  "QUIEN": "QUIÉN",
}


def normalize_label(raw: str) -> str:
    s = raw.strip().upper()
    return LABEL_ALIAS.get(s, s)


def label_from_filename(name: str) -> str:
    stem = pathlib.Path(name).stem
    m = re.match(r"^(.+?)_(\d+)$", stem)
    raw = m.group(1) if m else stem
    return normalize_label(raw)


# ── Cargar un PKL → array [30, 150] ──────────────────────────────────────────

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
            if len(row) != N_DIMS:
                return None
            seq.append(row)

        if not seq:
            return None

        if len(seq) > N_FRAMES:
            idx = np.linspace(0, len(seq) - 1, N_FRAMES, dtype=int)
            seq = [seq[i] for i in idx]
        while len(seq) < N_FRAMES:
            seq.append(seq[-1])

        return np.array(seq[:N_FRAMES], dtype=np.float32)
    except Exception:
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Construye dataset S12 con fuentes PUCP")
    parser.add_argument("--min-muestras", type=int, default=2,
                        help="Mínimo de muestras por clase para incluirla (default: 2)")
    parser.add_argument("--solo-stats", action="store_true",
                        help="Solo mostrar estadísticas, sin guardar NPZ")
    args = parser.parse_args()

    MIN_SAMPLES = args.min_muestras

    print("=" * 65)
    print("build_dataset_s12.py — Sprint 12: Dataset ampliado con fuentes PUCP")
    print("=" * 65)

    # Determinar fuentes activas
    all_sources = PKL_SOURCES_BASE + [
        (d, n, a) for d, n, a in PKL_SOURCES_PUCP if d.exists()
    ]

    pucp_activas = [n for d, n, a in PKL_SOURCES_PUCP if d.exists()]
    pucp_faltantes = [n for d, n, a in PKL_SOURCES_PUCP if not d.exists()]

    if pucp_activas:
        print(f"\n  Fuentes PUCP activas : {pucp_activas}")
    if pucp_faltantes:
        print(f"  Fuentes PUCP faltantes: {pucp_faltantes}")
        flags = " ".join(f"--{n}" for n in pucp_faltantes)
        print(f"  → Ejecutar primero: python scripts/download_pucp_datasets.py {flags}")

    # ── Cargar todos los PKL ──────────────────────────────────────────────────

    X_list, y_list, group_list, source_list = [], [], [], []

    for src_dir, src_name, is_abc in all_sources:
        if not src_dir.exists():
            print(f"\n  AVISO: {src_dir} no existe — saltando {src_name}")
            continue

        pkls    = sorted(src_dir.rglob("*.pkl"))
        loaded  = skipped = errors = 0
        clases_src = set()

        for p in pkls:
            # Determinar etiqueta
            if is_abc:
                label = normalize_label(p.parent.name.upper())
            else:
                label = label_from_filename(p.name)

            seq = pkl_to_sequence(p)
            if seq is None:
                errors += 1
                continue

            # Group ID: para fuentes PUCP extraer signer_id del nombre si existe
            # Convención: <CLASE>_<S##>_<###>.pkl → group = src_name/S##
            m = re.match(r"^.+_(S\d{2})_\d+$", p.stem)
            if m:
                group_id = f"{src_name}/{m.group(1)}"
            else:
                group_id = f"{src_name}/{label}"

            X_list.append(seq)
            y_list.append(label)
            group_list.append(group_id)
            source_list.append(src_name)
            clases_src.add(label)
            loaded += 1

        print(f"  {src_name:<14}: {loaded:>5} muestras  "
              f"{len(clases_src):>4} clases  ({errors} errores)")

    print(f"\n  Total muestras cargadas : {len(X_list)}")
    print(f"  LSP - Vocabulario-palabras únicas (raw)     : {len(set(y_list))}")

    # ── Construir arrays ──────────────────────────────────────────────────────

    counts_raw = Counter(y_list)
    classes    = sorted(c for c, n in counts_raw.items() if n >= MIN_SAMPLES)
    label2idx  = {c: i for i, c in enumerate(classes)}
    idx2label  = {i: c for c, i in label2idx.items()}

    # Filtrar
    keep = [i for i, lbl in enumerate(y_list) if lbl in label2idx]
    X_list      = [X_list[i]      for i in keep]
    y_list      = [y_list[i]      for i in keep]
    group_list  = [group_list[i]  for i in keep]
    source_list = [source_list[i] for i in keep]

    X          = np.array(X_list, dtype=np.float32)
    y          = np.array([label2idx[l] for l in y_list], dtype=np.int64)
    sources_np = np.array(source_list)

    # Groups como enteros
    unique_groups = sorted(set(group_list))
    g2int         = {g: i for i, g in enumerate(unique_groups)}
    groups_int    = np.array([g2int[g] for g in group_list], dtype=np.int32)

    n_classes = len(classes)

    # ── Estadísticas ──────────────────────────────────────────────────────────

    counts_final = Counter(y.tolist())
    vals = sorted(counts_final.values(), reverse=True)

    print(f"\n  ── Estadísticas S12 ──")
    print(f"  Muestras (≥{MIN_SAMPLES}/clase): {len(X):,}")
    print(f"  LSP - Vocabulario-palabras activas            : {n_classes}")
    print(f"  Grupos (señantes/fuentes) : {len(unique_groups)}")
    print(f"  Muestras/clase: min={min(vals)}  max={max(vals)}  "
          f"media={np.mean(vals):.1f}  mediana={np.median(vals):.0f}")
    print(f"  LSP - Vocabulario-palabras con ≥5   : {sum(1 for v in vals if v >= 5)}")
    print(f"  LSP - Vocabulario-palabras con ≥10  : {sum(1 for v in vals if v >= 10)}")
    print(f"  LSP - Vocabulario-palabras con ≥50  : {sum(1 for v in vals if v >= 50)}")
    print(f"  LSP - Vocabulario-palabras con ≥100 : {sum(1 for v in vals if v >= 100)}")

    print(f"\n  ── Por fuente ──")
    for src_name in sorted(set(source_list)):
        mask  = sources_np == src_name
        n_src = int(mask.sum())
        cls_s = len(set(y[mask].tolist()))
        print(f"  {src_name:<14}: {n_src:>6} muestras  {cls_s:>4} LSP - Vocabulario-palabras")

    # Comparativa con S11
    print(f"\n  ── Comparativa S11 vs S12 ──")
    s11_npz = OUT_DIR / "dataset_s11.npz"
    if s11_npz.exists():
        s11 = np.load(s11_npz)
        n11, n12 = len(s11["y"]), len(X)
        pct_m = (n12 / n11 - 1) * 100
        print(f"  S11: {n11:,} muestras  |  S12: {n12:,} muestras  ({pct_m:+.0f}%)")
    else:
        print(f"  S11: no encontrado  |  S12: {len(X):,} muestras")

    # LSP - Vocabulario-palabras nuevas vs S11
    s11_json = OUT_DIR / "s11_label2idx.json"
    if s11_json.exists():
        with open(s11_json, encoding="utf-8") as f:
            s11_clases = set(json.load(f).keys())
        s12_clases    = set(label2idx.keys())
        nuevas_en_s12 = s12_clases - s11_clases
        print(f"  LSP - Vocabulario-palabras nuevas en S12 (vs S11): {len(nuevas_en_s12)}")
        if nuevas_en_s12:
            print(f"    Ejemplos: {sorted(nuevas_en_s12)[:10]}")

    if args.solo_stats:
        print("\n  [--solo-stats] No se guardó ningún archivo.")
        return

    # ── Guardar ───────────────────────────────────────────────────────────────

    npz_path = OUT_DIR / "dataset_s12.npz"
    np.savez_compressed(npz_path, X=X, y=y, groups=groups_int)
    sz = npz_path.stat().st_size / 1e6
    print(f"\n✅ {npz_path}  ({sz:.1f} MB)")
    print(f"   X={X.shape}  y={y.shape}  groups={groups_int.shape}")

    with open(OUT_DIR / "s12_label2idx.json", "w", encoding="utf-8") as f:
        json.dump(label2idx, f, ensure_ascii=False, indent=2)

    with open(OUT_DIR / "s12_idx2label.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in idx2label.items()}, f, ensure_ascii=False, indent=2)

    np.save(OUT_DIR / "s12_sources.npy", sources_np)

    print(f"✅ s12_label2idx.json  ({n_classes} LSP - Vocabulario-palabras)")
    print(f"✅ s12_idx2label.json")
    print(f"✅ s12_sources.npy")

    print(f"""
  ── Siguiente paso ──────────────────────────────────────────────
  Entrenar con dataset S12:
    .venv310/bin/python3 scripts/train_s11.py
    (modificar DATA_PATH = "data/dataset_s12.npz" y
              LABEL_JSON = "data/s12_label2idx.json")
  ────────────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
