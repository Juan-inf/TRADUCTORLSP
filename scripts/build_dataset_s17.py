"""build_dataset_s17.py

Construye dataset S17 — igual a S16 pero con FIX de grupos cross-source.

PROBLEMA EN S16:
  HISTORIAS_VINETAS_XX aparece en dgi156_pkl (grupo 2000+X) Y en pkl/vineta
  (grupo 3000+Y). GroupShuffleSplit los trata como grupos independientes:
    → dgi156 va a training, vineta va a holdout → cross-source contamination
    → modelo entrenado en dgi156 evalúa en vineta → F1≈0 en 234 samples (HV_23)

FIX EN S17:
  Post-procesamiento: para toda clase que aparece en >1 fuente, se asigna
  el MISMO grupo a todas sus muestras usando hash(clase_nombre) % 1000 + 4000.
  Esto garantiza que GroupShuffleSplit mantiene juntos todos los samples de esa
  clase (de cualquier fuente) → siempre van juntos a training o a holdout.

Uso:
    python3 scripts/build_dataset_s17.py [--min-muestras N] [--estadisticas]
"""

import argparse
import json
import pathlib
import pickle
import collections
import sys
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

ROOT    = pathlib.Path(__file__).resolve().parent.parent
DATA    = ROOT / "data"
KP_DIR  = DATA / "Keypoints"

N_FRAMES = 30
N_DIMS   = 150
SEED     = 42


# ─── helpers ──────────────────────────────────────────────────────────────────

def pkl_to_sequence(pkl_path: pathlib.Path) -> np.ndarray | None:
    try:
        with open(pkl_path, "rb") as f:
            frames = pickle.load(f)
        if not frames:
            return None
        seq = []
        for fr in frames:
            if isinstance(fr, dict):
                row = []
                for key, n in (("pose", 33), ("left_hand", 21), ("right_hand", 21)):
                    d = fr.get(key, {})
                    if isinstance(d, dict):
                        x = d.get("x", [])
                        y = d.get("y", [])
                    elif hasattr(d, "landmark"):
                        x = [lm.x for lm in d.landmark]
                        y = [lm.y for lm in d.landmark]
                    else:
                        x, y = [], []
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


# ─── configuración de fuentes (idéntica a S16) ────────────────────────────────

SOURCE_CONFIG = {
    "abecedario_pkl": {
        "etiqueta": "abecedario",
        "estrategia": "por_subgrupo",
        "n_subgrupos": 3,
        "grupo_offset": 1000,
    },
    "dgi156_pkl": {
        "etiqueta": "dgi156",
        "estrategia": "por_clase",
        "grupo_offset": 2000,
    },
    "pkl": {
        "etiqueta": "vineta",
        "estrategia": "por_clase",
        "grupo_offset": 3000,
    },
    "aec_pkl": {
        "etiqueta": "aec",
        "estrategia": "por_subgrupo",
        "n_subgrupos": 5,
        "grupo_offset": 5000,
    },
    "vocabulario_lsp_p_pkl": {
        "etiqueta": "vocabulario_lsp_p",
        "estrategia": "por_señante",
        "grupo_offset": 7000,
    },
    "glosas_pkl": {
        "etiqueta": "glosa",
        "estrategia": "hash_nombre",
        "grupo_offset": 8000,
    },
}

SOURCE_CONFIG_EXTRA = {
    "dgi156_github": {
        "etiqueta": "dgi156_github",
        "estrategia": "hash_nombre",
        "grupo_offset": 9000,
    },
}


def inferir_grupo(pkl_path: pathlib.Path, clase_idx: int,
                  config: dict, muestra_idx: int) -> int:
    offset     = config.get("grupo_offset", 0)
    estrategia = config.get("estrategia", "por_clase")

    if estrategia == "por_clase":
        return offset + clase_idx
    elif estrategia == "por_subgrupo":
        n       = config.get("n_subgrupos", 3)
        subgrupo = muestra_idx % n
        return offset + clase_idx * 100 + subgrupo
    elif estrategia == "por_señante":
        stem  = pkl_path.stem
        parts = stem.rsplit("_", 1)
        señante = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else hash(stem) % 10
        return offset + señante
    elif estrategia == "hash_nombre":
        return offset + abs(hash(pkl_path.parent.name)) % 500
    return offset + clase_idx


def cargar_fuente(kp_base: pathlib.Path, config: dict) -> tuple:
    Xs, Ls, Gs, Ss = [], [], [], []
    etiqueta = config["etiqueta"]

    if not kp_base.exists():
        return np.array([]), np.array([]), np.array([]), np.array([])

    clases = sorted(d for d in kp_base.iterdir() if d.is_dir())
    n_ok = n_err = 0

    for clase_idx, clase_dir in enumerate(clases):
        clase_name = clase_dir.name
        pkls = sorted(clase_dir.glob("*.pkl"))
        for m_idx, pkl_path in enumerate(pkls):
            seq = pkl_to_sequence(pkl_path)
            if seq is None:
                n_err += 1
                continue
            grupo = inferir_grupo(pkl_path, clase_idx, config, m_idx)
            Xs.append(seq)
            Ls.append(clase_name.upper())
            Gs.append(grupo)
            Ss.append(etiqueta)
            n_ok += 1

    print(f"  {etiqueta:<18}: {n_ok:>5} muestras, {len(clases):>4} clases  [{n_err} errores]")

    if not Xs:
        return np.array([]), np.array([]), np.array([]), np.array([])

    return (np.array(Xs, dtype=np.float32),
            np.array(Ls),
            np.array(Gs, dtype=np.int32),
            np.array(Ss))


# ─── FIX S17: unificar grupos para clases multi-fuente ───────────────────────

def balancear_grupos_hv(L_all: np.ndarray, G_all: np.ndarray,
                        S_all: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Crea sub-grupos balanceados para clases dgi156 ∩ vineta (HV classes).

    Raíz del problema HE3 en S26: HV_XX tenía grupos separados por fuente
    (dgi156→offset 2000, vineta→offset 3000). GroupShuffleSplit podía poner
    dgi156 en training y vineta en holdout → el modelo nunca aprendió features
    de vineta para esa clase → F1≈0 con 234 samples (HV_23).

    Solución: para cada clase HV, barajar TODAS las muestras (de ambas fuentes)
    y asignar 5 sub-grupos por posición. Cada sub-grupo contendrá ~50% dgi156
    y ~50% vineta. Así:
      - El modelo entrenado en sub-grupos 0-3 ve AMBAS fuentes → aprende HV bien
      - El holdout (sub-grupo 4) también tiene ambas fuentes → F1 ≠ 0

    NO se modifica abecedario/AEC (que usan por_subgrupo propio).
    """
    G_new = G_all.copy()
    rng   = np.random.default_rng(seed)

    clases_dgi156  = set(L_all[S_all == "dgi156"])
    clases_vineta  = set(L_all[S_all == "vineta"])
    clases_hv_cruce = sorted(clases_dgi156 & clases_vineta)

    if not clases_hv_cruce:
        print("  No hay clases dgi156∩vineta — grupos sin cambio.")
        return G_new

    N_SUBGRUPOS = 5   # 5 sub-grupos por clase HV (≈ 20% en holdout si se elige 1)
    BASE_OFFSET = 20000  # fuera del rango de todos los demás offsets (máx ≈ 9000)

    print(f"\n  [S17 FIX] Sub-grupos balanceados para clases dgi156∩vineta: {len(clases_hv_cruce)}")
    for class_idx, label in enumerate(clases_hv_cruce):
        mask    = L_all == label
        indices = np.where(mask)[0]

        # Shuffle para mezclar dgi156 y vineta aleatoriamente
        indices_shuffled = rng.permutation(indices)

        # Asignar sub-grupos por posición (round-robin) — sin colisiones entre clases
        base = BASE_OFFSET + class_idx * N_SUBGRUPOS
        for pos, idx in enumerate(indices_shuffled):
            G_new[idx] = base + (pos % N_SUBGRUPOS)

        n_dgi  = int((mask & (S_all == "dgi156")).sum())
        n_vin  = int((mask & (S_all == "vineta")).sum())
        grupos = sorted(set(G_new[indices].tolist()))
        print(f"    {label:<35}  dgi156={n_dgi:3d}  vineta={n_vin:3d}"
              f"  → {N_SUBGRUPOS} sub-grupos {grupos[0]}..{grupos[-1]}")

    return G_new


# ─── análisis HE3 ─────────────────────────────────────────────────────────────

def analizar_he3(y_labels: np.ndarray, groups: np.ndarray,
                 sources: np.ndarray) -> dict:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    tv_idx, heldout_idx = next(gss.split(np.zeros(len(y_labels)), y_labels, groups))

    clases_train   = set(y_labels[tv_idx])
    clases_holdout = set(y_labels[heldout_idx])
    clases_solo_holdout = clases_holdout - clases_train

    cnt_holdout  = collections.Counter(y_labels[heldout_idx])
    muestras_cero = sum(cnt_holdout[c] for c in clases_solo_holdout)
    total_holdout = len(heldout_idx)
    src_holdout   = collections.Counter(sources[heldout_idx])

    return {
        "n_train": len(tv_idx),
        "n_holdout": total_holdout,
        "clases_train": len(clases_train),
        "clases_holdout": len(clases_holdout),
        "clases_sin_training": sorted(clases_solo_holdout),
        "n_clases_sin_training": len(clases_solo_holdout),
        "muestras_holdout_cero_training": muestras_cero,
        "pct_holdout_cero": 100 * muestras_cero / total_holdout if total_holdout else 0,
        "fuentes_holdout": dict(src_holdout),
    }


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-muestras", type=int, default=15)
    ap.add_argument("--estadisticas", action="store_true")
    args = ap.parse_args()

    print("=" * 65)
    print("Build Dataset S17 — S16 + FIX grupos cross-source HV")
    print(f"  min_muestras = {args.min_muestras}")
    print("=" * 65)
    print()

    # ── Cargar fuentes ────────────────────────────────────────────────────────
    print("Cargando fuentes de keypoints:")
    all_X, all_L, all_G, all_S = [], [], [], []

    for carpeta, config in SOURCE_CONFIG.items():
        kp_base = KP_DIR / carpeta
        X, L, G, S = cargar_fuente(kp_base, config)
        if len(X) > 0:
            all_X.append(X); all_L.append(L)
            all_G.append(G); all_S.append(S)

    for carpeta, config in SOURCE_CONFIG_EXTRA.items():
        kp_base = KP_DIR / carpeta
        if kp_base.exists():
            X, L, G, S = cargar_fuente(kp_base, config)
            if len(X) > 0:
                all_X.append(X); all_L.append(L)
                all_G.append(G); all_S.append(S)

    if not all_X:
        print("ERROR: No se encontraron datos en data/Keypoints/")
        sys.exit(1)

    X_all = np.concatenate(all_X, axis=0)
    L_all = np.concatenate(all_L)
    G_all = np.concatenate(all_G)
    S_all = np.concatenate(all_S)

    print(f"\nTotal antes de filtro: {len(X_all)} muestras, {len(set(L_all))} clases")

    # ── Filtro min_muestras ───────────────────────────────────────────────────
    cnts = collections.Counter(L_all)
    mask = np.array([cnts[l] >= args.min_muestras for l in L_all])
    X_all = X_all[mask]
    L_all = L_all[mask]
    G_all = G_all[mask]
    S_all = S_all[mask]
    print(f"Total después de min{args.min_muestras}: {len(X_all)} muestras, {len(set(L_all))} clases")

    # ── Distribución por fuente ───────────────────────────────────────────────
    print("\nDistribución por fuente (post-filtro):")
    src_cnts = collections.Counter(S_all)
    for src, n in sorted(src_cnts.items()):
        clases_src = len(set(L_all[S_all == src]))
        print(f"  {src:<18}: {n:>5} muestras, {clases_src:>4} clases")

    # ── FIX CROSS-SOURCE: sub-grupos balanceados para HV ─────────────────────
    print("\nAplicando FIX cross-source (sub-grupos balanceados dgi156+vineta):")
    G_all = balancear_grupos_hv(L_all, G_all, S_all)

    # ── Análisis HE3 con grupos corregidos ───────────────────────────────────
    print("\nAnálisis HE3 (con grupos S17 — post-fix):")
    he3 = analizar_he3(L_all, G_all, S_all)
    print(f"  Train+Val : {he3['n_train']}")
    print(f"  Holdout   : {he3['n_holdout']}")
    print(f"  Clases holdout sin training: {he3['n_clases_sin_training']}")
    if he3["clases_sin_training"]:
        print(f"  Clases afectadas: {he3['clases_sin_training']}")
    print(f"  Muestras holdout con F1=0 garantizado: {he3['muestras_holdout_cero_training']} "
          f"({he3['pct_holdout_cero']:.1f}%)")
    print(f"  Fuentes holdout: {he3['fuentes_holdout']}")

    vs_s16 = 193
    mejora = vs_s16 - he3["muestras_holdout_cero_training"]
    print(f"\n  vs S16: {vs_s16} → {he3['muestras_holdout_cero_training']} "
          f"({'↓' if mejora > 0 else '↑'}{abs(mejora)} muestras con F1=0 garantizado)")

    if args.estadisticas:
        print("\n[--estadisticas] No se guarda nada.")
        return

    # ── Construir label2idx ───────────────────────────────────────────────────
    classes   = sorted(set(L_all))
    label2idx = {c: i for i, c in enumerate(classes)}
    idx2label = {i: c for c, i in label2idx.items()}
    y_all     = np.array([label2idx[l] for l in L_all], dtype=np.int64)

    cnts_final = collections.Counter(L_all)
    print(f"\nEstadísticas finales:")
    print(f"  Clases:   {len(classes)}")
    print(f"  Muestras: {len(X_all)}")
    print(f"  Min/Max/Med por clase: "
          f"{min(cnts_final.values())} / {max(cnts_final.values())} / "
          f"{np.median(list(cnts_final.values())):.0f}")

    # ── Guardar ───────────────────────────────────────────────────────────────
    np.savez_compressed(DATA / "dataset_s17.npz", X=X_all, y=y_all, groups=G_all)
    np.save(DATA / "s17_sources.npy", S_all)

    with open(DATA / "s17_label2idx.json", "w", encoding="utf-8") as f:
        json.dump(label2idx, f, ensure_ascii=False, indent=2)
    with open(DATA / "s17_idx2label.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in idx2label.items()}, f, ensure_ascii=False, indent=2)
    with open(DATA / "s17_he3_report.json", "w", encoding="utf-8") as f:
        json.dump(he3, f, ensure_ascii=False, indent=2)

    print(f"\n✅ dataset_s17.npz  ({len(X_all)} muestras, {len(classes)} clases)")
    print(f"✅ s17_label2idx.json")
    print(f"✅ s17_sources.npy")
    print(f"✅ s17_he3_report.json")
    print(f"\nSiguiente paso:")
    print(f"  python3 scripts/train_s27.py --skip-hpo")


if __name__ == "__main__":
    main()
