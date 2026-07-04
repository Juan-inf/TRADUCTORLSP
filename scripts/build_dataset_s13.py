"""
build_dataset_s13.py — Dataset Sprint 13

Extiende S12 incorporando:
  - PUCP-AEC       (aec_pkl/)    — 506 glosas LSP, intérprete TV
  - PUCP-DGI156    (dgi156_pkl/) — 156 glosas LSP, múltiples señantes
  - LSA64          (lsa64_pkl/)  — 64 señas argentinas, 10 señantes

Mejoras vs S12 (dataset ampliado):
  ┌──────────────────────────────────────────────────────────────────┐
  │  Fuente         Clases  Muestras  Señantes                      │
  │  S12 base       501     7,102     mezclado                      │
  │  + AEC          +55     +1,639    +1 intérprete                 │
  │  + DGI156       +156    ~+1,000   múltiples                     │
  │  + LSP-Base        +64     +3,200    +10 señantes                  │
  │  S13 esperado   ~600    ~13,000   diverso                       │
  └──────────────────────────────────────────────────────────────────┘

Outputs:
  data/dataset_s13.npz
  data/s13_label2idx.json
  data/s13_idx2label.json
  data/s13_sources.npy

Uso:
  .venv310/bin/python3 scripts/build_dataset_s13.py
  .venv310/bin/python3 scripts/build_dataset_s13.py --min-muestras 5
  .venv310/bin/python3 scripts/build_dataset_s13.py --solo-stats
"""

import argparse, json, pathlib, pickle, re, warnings
from collections import Counter, defaultdict
import numpy as np

warnings.filterwarnings("ignore")

ROOT     = pathlib.Path(__file__).parent.parent
OUT_DIR  = ROOT / "data"
KP_DIR   = ROOT / "data" / "Keypoints"
N_FRAMES = 30
N_DIMS   = 150
SEED     = 42

# ── Fuentes ───────────────────────────────────────────────────────────────────
PKL_SOURCES_S12 = [
    (KP_DIR / "pkl",            "vineta",    False),
    (KP_DIR / "glosas_pkl",     "glosa",     False),
    (KP_DIR / "abecedario_pkl", "abecedario", True),
    (KP_DIR / "pucp305_pkl",    "pucp305",   False),
]

PKL_SOURCES_S13 = [
    (KP_DIR / "aec_pkl",    "aec",    False),
    (KP_DIR / "dgi156_pkl", "dgi156", False),
    (KP_DIR / "lsa64_pkl",  "lsa64",  False),
]

LABEL_ALIAS = {
    "AHI": "AHÍ", "QUE?": "QUÉ?", "SI": "SÍ", "TU": "TÚ",
    "MAS": "MÁS", "VIO": "VIÓ",
    "COMO": "CÓMO", "DONDE": "DÓNDE", "CUANDO": "CUÁNDO",
    "QUE": "QUÉ", "QUIEN": "QUIÉN", "CUANTO": "CUÁNTO",
    "LLEGÓ": "LLEGAR", "VENIR": "VENIR",
    # LSP-Base prefixes stripped for matching
}


def normalize_label(raw: str) -> str:
    s = raw.strip().upper()
    # Quitar prefijo LSP-Base_ para intentar matching con LSP
    if s.startswith("LSP-Base_"):
        s = s[6:]
    return LABEL_ALIAS.get(s, s)


def label_from_path(pkl_path: pathlib.Path, is_abc: bool) -> str:
    if is_abc:
        return normalize_label(pkl_path.parent.name.upper())
    # Para AEC: el directorio ya es el glosa normalizado
    # Para otros: <CLASE>_<n>.pkl
    stem = pkl_path.stem
    m    = re.match(r"^(.+?)_(\d+)$", stem)
    raw  = m.group(1) if m else stem
    # Si el directorio padre tiene la clase, usarlo (más fiable)
    parent_name = pkl_path.parent.name.upper()
    if parent_name and parent_name not in ("PKL", "KEYPOINTS"):
        return normalize_label(parent_name)
    return normalize_label(raw)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-muestras", type=int, default=5,
                    help="Mínimo muestras/clase (default: 5)")
    ap.add_argument("--solo-stats",   action="store_true")
    args = ap.parse_args()
    MIN_SAMPLES = args.min_muestras

    print("=" * 65)
    print("build_dataset_s13.py — Sprint 13: AEC + DGI156 + LSP-Base")
    print("=" * 65)

    all_sources = PKL_SOURCES_S12 + [
        (d, n, a) for d, n, a in PKL_SOURCES_S13 if d.exists()
    ]

    s13_activas  = [n for d, n, a in PKL_SOURCES_S13 if d.exists()]
    s13_faltantes = [n for d, n, a in PKL_SOURCES_S13 if not d.exists()]
    if s13_activas:
        print(f"\n  Fuentes S13 activas   : {s13_activas}")
    if s13_faltantes:
        print(f"  Fuentes S13 faltantes : {s13_faltantes}")
        print(f"  → Ejecutar primero: .venv311/bin/python3 scripts/download_s13_datasets.py")

    # ── Cargar todos los PKL ──────────────────────────────────────────────────
    X_list, y_list, group_list, source_list = [], [], [], []

    for src_dir, src_name, is_abc in all_sources:
        if not src_dir.exists():
            print(f"\n  AVISO: {src_dir.name} no existe — saltando {src_name}")
            continue

        pkls    = sorted(src_dir.rglob("*.pkl"))
        loaded  = skipped = errors = 0
        clases_src = set()

        for p in pkls:
            label = label_from_path(p, is_abc)
            seq   = pkl_to_sequence(p)
            if seq is None:
                errors += 1
                continue

            # Group ID
            m = re.match(r"^.+_(S\d{2})_\d+$", p.stem)
            if m:
                group_id = f"{src_name}/{m.group(1)}"
            elif src_name == "lsa64":
                # LSP-Base: <sign>_<subject>_<rep>.pkl → group = lsa64/S<subject>
                parts = p.stem.split("_")
                if len(parts) >= 3:
                    group_id = f"lsa64/S{parts[-2]}"
                else:
                    group_id = f"lsa64/{label}"
            elif src_name == "aec":
                # AEC: único señante (intérprete TV)
                group_id = "aec/S01"
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

    print(f"\n  Total cargado (bruto): {len(X_list)} muestras  "
          f"{len(set(y_list))} clases únicas")

    # ── Filtro min_samples ────────────────────────────────────────────────────
    counts_raw = Counter(y_list)
    classes    = sorted(c for c, n in counts_raw.items() if n >= MIN_SAMPLES)
    label2idx  = {c: i for i, c in enumerate(classes)}
    idx2label  = {i: c for c, i in label2idx.items()}

    keep = [i for i, lbl in enumerate(y_list) if lbl in label2idx]
    X_list      = [X_list[i]      for i in keep]
    y_list      = [y_list[i]      for i in keep]
    group_list  = [group_list[i]  for i in keep]
    source_list = [source_list[i] for i in keep]

    X          = np.array(X_list, dtype=np.float32)
    y          = np.array([label2idx[l] for l in y_list], dtype=np.int64)
    sources_np = np.array(source_list)

    unique_groups = sorted(set(group_list))
    g2int         = {g: i for i, g in enumerate(unique_groups)}
    groups_int    = np.array([g2int[g] for g in group_list], dtype=np.int32)

    n_classes = len(classes)

    # ── Estadísticas ──────────────────────────────────────────────────────────
    counts_final = Counter(y.tolist())
    vals = sorted(counts_final.values(), reverse=True)

    print(f"\n  ── Estadísticas S13 (≥{MIN_SAMPLES}/clase) ──")
    print(f"  Muestras  : {len(X):,}")
    print(f"  LSP - Vocabulario-palabras    : {n_classes}")
    print(f"  Grupos    : {len(unique_groups)}")
    print(f"  Media/cls : {np.mean(vals):.1f}  mediana={np.median(vals):.0f}  "
          f"min={min(vals)}  max={max(vals)}")
    print(f"  LSP - Vocabulario-palabras ≥5  : {sum(1 for v in vals if v >= 5)}")
    print(f"  LSP - Vocabulario-palabras ≥10 : {sum(1 for v in vals if v >= 10)}")
    print(f"  LSP - Vocabulario-palabras ≥50 : {sum(1 for v in vals if v >= 50)}")

    print(f"\n  ── Por fuente ──")
    for src in sorted(set(source_list)):
        mask  = sources_np == src
        n_src = int(mask.sum())
        cls_s = len(set(y[mask].tolist()))
        print(f"  {src:<14}: {n_src:>6} muestras  {cls_s:>4} LSP - Vocabulario-palabras")

    # Comparativa
    print(f"\n  ── Comparativa ──")
    for prev, path in [("S11", "data/dataset_s11.npz"),
                        ("S12", "data/dataset_s12.npz")]:
        ppath = ROOT / path
        if ppath.exists():
            d = np.load(ppath)
            n = len(d["y"])
            pct = (len(X) / n - 1) * 100
            print(f"  {prev}: {n:,} → S13: {len(X):,}  ({pct:+.0f}%)")

    # LSP - Vocabulario-palabras nuevas
    s12_path = ROOT / "data" / "s12_label2idx.json"
    if s12_path.exists():
        with open(s12_path, encoding="utf-8") as f:
            s12_cls = set(json.load(f).keys())
        nuevas = set(label2idx.keys()) - s12_cls
        print(f"  LSP - Vocabulario-palabras nuevas en S13 vs S12: {len(nuevas)}")
        print(f"  Ejemplos: {sorted(nuevas)[:15]}")

    if args.solo_stats:
        print("\n  [--solo-stats] Sin guardar.")
        return

    # ── Guardar ───────────────────────────────────────────────────────────────
    npz_path = OUT_DIR / "dataset_s13.npz"
    np.savez_compressed(npz_path, X=X, y=y, groups=groups_int)
    sz = npz_path.stat().st_size / 1e6
    print(f"\n✅ {npz_path}  ({sz:.1f} MB)")
    print(f"   X={X.shape}  y={y.shape}  groups={groups_int.shape}")

    with open(OUT_DIR / "s13_label2idx.json", "w", encoding="utf-8") as f:
        json.dump(label2idx, f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / "s13_idx2label.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in idx2label.items()},
                  f, ensure_ascii=False, indent=2)
    np.save(OUT_DIR / "s13_sources.npy", sources_np)

    print(f"✅ s13_label2idx.json  ({n_classes} LSP - Vocabulario-palabras)")
    print(f"✅ s13_idx2label.json")
    print(f"✅ s13_sources.npy")
    print(f"""
  ── Siguiente paso ──────────────────────────────────────────────
  Entrenar con dataset S13 (min≥{MIN_SAMPLES}):
    .venv310/bin/python3 scripts/train_s12.py \\
        [adaptar DATA_PATH a dataset_s13.npz]
  ────────────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
