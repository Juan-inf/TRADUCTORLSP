"""build_dataset_s15.py

Construye dataset S15 = S14 (LSP puro, sin lsa64) + nuevas instancias AEC.

Estrategia:
- Parte del dataset S14 base (s13.npz filtrado lsa64)
- Agrega instancias AEC que coincidan con el vocabulario S14 (53 clases, 847 instancias)
- Agrega instancias AEC de clases NUEVAS (453 clases) con min_muestras >=5 en AEC
- Resultante: más muestras para clases débiles + vocabulario expandido

Uso:
    python3 scripts/build_dataset_s15.py [--min-muestras N] [--solo-overlap]
"""

import argparse, json, pathlib, pickle, collections, sys
import numpy as np

ROOT      = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
AEC_DIR   = DATA_DIR / "external_lsp" / "aec"
AEC_DICT  = AEC_DIR / "dict.json"
AEC_PKL   = AEC_DIR / "Keypoints" / "pkl"

N_FRAMES = 30
N_DIMS   = 150

# ─── helpers ─────────────────────────────────────────────────────────────────

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


def load_s14_base() -> tuple:
    """Carga dataset S13, filtra lsa64, devuelve X, labels (strings), groups, sources."""
    npz = np.load(DATA_DIR / "dataset_s13.npz", allow_pickle=True)
    X_raw   = npz["X"]
    y_raw   = npz["y"]
    g_raw   = npz["groups"]
    src_raw = np.load(DATA_DIR / "s13_sources.npy", allow_pickle=True)

    mask = src_raw != "lsa64"
    X_raw, y_raw, g_raw, src_raw = X_raw[mask], y_raw[mask], g_raw[mask], src_raw[mask]

    with open(DATA_DIR / "s13_idx2label.json") as f:
        idx2label_s13 = {int(k): v for k, v in json.load(f).items()}

    labels = np.array([idx2label_s13[int(i)] for i in y_raw])
    return X_raw, labels, g_raw, src_raw


def load_aec_instances(aec_dict: dict, s14_glosses: set, solo_overlap: bool,
                       min_muestras: int) -> tuple:
    """Lee PKL de AEC y devuelve X, labels, groups, sources."""
    Xs, Ls, Gs, Ss = [], [], [], []
    skipped, loaded = 0, 0

    # Determina qué clases incluir
    for k in aec_dict:
        gloss  = aec_dict[k]["gloss"].lower().strip()
        is_new = gloss not in s14_glosses

        if solo_overlap and is_new:
            continue

        instances = aec_dict[k]["instances"]
        if is_new and len(instances) < min_muestras:
            continue

        for inst in instances:
            rel = inst["keypoints_path"]
            # "./Data/AEC/Keypoints/pkl/ira_alegria/yo_1.pkl"
            # → AEC_PKL / "ira_alegria" / "yo_1.pkl"
            parts = pathlib.Path(rel).parts
            try:
                kp_idx  = [i for i, p in enumerate(parts) if p == "pkl"][0]
                sub     = parts[kp_idx + 1]
                fname   = parts[kp_idx + 2]
                pkl_path = AEC_PKL / sub / fname
            except (IndexError, ValueError):
                skipped += 1
                continue

            if not pkl_path.exists():
                skipped += 1
                continue

            seq = pkl_to_sequence(pkl_path)
            if seq is None:
                skipped += 1
                continue

            Xs.append(seq)
            Ls.append(gloss)
            Gs.append(inst.get("signer_id", -1))
            Ss.append("aec")
            loaded += 1

    print(f"  AEC: {loaded} instancias cargadas, {skipped} omitidas")
    if not Xs:
        return None, None, None, None

    return (np.array(Xs, dtype=np.float32),
            np.array(Ls),
            np.array(Gs),
            np.array(Ss))


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-muestras", type=int, default=5)
    ap.add_argument("--solo-overlap", action="store_true",
                    help="Solo agregar clases que ya están en S14 (no clases nuevas)")
    args = ap.parse_args()

    print("=" * 60)
    print("Build Dataset S15")
    print("  S14 base + AEC (LSP Peruano)")
    print("=" * 60)

    # 1. Carga base S14
    print("\n1. Cargando base S14...")
    X_base, L_base, G_base, S_base = load_s14_base()
    print(f"   Base S14: {len(X_base)} muestras, {len(set(L_base))} clases")

    # 2. Carga AEC
    print("\n2. Procesando AEC PKLs...")
    with open(AEC_DICT) as f:
        aec_dict = json.load(f)

    s14_glosses = set(L_base)
    X_aec, L_aec, G_aec, S_aec = load_aec_instances(
        aec_dict, s14_glosses, args.solo_overlap, args.min_muestras)

    # 3. Merge
    print("\n3. Mergeando datasets...")
    if X_aec is not None:
        X_all = np.concatenate([X_base, X_aec], axis=0)
        L_all = np.concatenate([L_base, L_aec])
        G_all = np.concatenate([G_base, G_aec])
        S_all = np.concatenate([S_base, S_aec])
    else:
        X_all, L_all, G_all, S_all = X_base, L_base, G_base, S_base

    # 4. Filtro min_muestras
    counts = collections.Counter(L_all)
    keep_mask = np.array([counts[l] >= args.min_muestras for l in L_all])
    X_all, L_all, G_all, S_all = (X_all[keep_mask], L_all[keep_mask],
                                   G_all[keep_mask], S_all[keep_mask])

    # 5. Construye label2idx
    classes = sorted(set(L_all))
    label2idx = {c: i for i, c in enumerate(classes)}
    idx2label = {i: c for c, i in label2idx.items()}
    y_all = np.array([label2idx[l] for l in L_all], dtype=np.int64)

    n_classes = len(classes)
    print(f"   Total muestras: {len(X_all)}")
    print(f"   Total clases:   {n_classes}")
    counts_final = collections.Counter(L_all)
    print(f"   Samples/cls: min={min(counts_final.values())} "
          f"max={max(counts_final.values())} "
          f"mean={np.mean(list(counts_final.values())):.1f} "
          f"median={np.median(list(counts_final.values())):.0f}")

    # Fuentes
    src_counts = collections.Counter(S_all)
    print(f"   Fuentes: {dict(src_counts)}")

    # 6. Guarda
    out_npz = DATA_DIR / "dataset_s15.npz"
    np.savez_compressed(out_npz, X=X_all, y=y_all, groups=G_all)
    np.save(DATA_DIR / "s15_sources.npy", S_all)

    with open(DATA_DIR / "s15_label2idx.json", "w", encoding="utf-8") as f:
        json.dump(label2idx, f, ensure_ascii=False, indent=2)
    with open(DATA_DIR / "s15_idx2label.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in idx2label.items()}, f, ensure_ascii=False, indent=2)

    print(f"\n✅ dataset_s15.npz guardado ({len(X_all)} muestras, {n_classes} clases)")
    print(f"✅ s15_label2idx.json ({n_classes} clases)")
    print(f"✅ s15_sources.npy")


if __name__ == "__main__":
    main()
