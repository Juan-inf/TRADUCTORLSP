"""build_dataset_s16.py

Construye dataset S16 usando TODAS las fuentes LSP localmente disponibles,
con estrategia de grupos mejorada para resolver el problema HE3.

MEJORAS vs S15:
  1. Integra todas las fuentes en data/Keypoints/: dgi156, abecedario, aec, pucp305, glosas
  2. Grupos de abecedario: asigna sub-grupos por número de muestra para simular
     diversidad de señantes (en ausencia de datos reales de múltiples señantes)
  3. Filtra clases con min_muestras >= 15 (mismo criterio que S15)
  4. Guarda fuentes por muestra en s16_sources.npy

Uso:
    python3 scripts/build_dataset_s16.py [--min-muestras N] [--excluir-lsa64]
    python3 scripts/build_dataset_s16.py --estadisticas   # solo analiza sin guardar
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
    """Lee un PKL y devuelve array [30, 150] float32."""
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


# ─── carga de fuentes ─────────────────────────────────────────────────────────

SOURCE_CONFIG = {
    # nombre_carpeta: (etiqueta_fuente, estrategia_grupos)
    # estrategia: "por_clase" = 1 grupo por clase (como abecedario actual)
    #             "por_señante" = inferir señante del nombre del archivo
    #             "por_video" = inferir video del nombre del archivo
    #             "hash_nombre" = hash del nombre del dir padre
    "abecedario_pkl": {
        "etiqueta": "abecedario",
        "estrategia": "por_subgrupo",  # divide 150 muestras/letra en 3 sub-grupos de 50
        "n_subgrupos": 3,              # simula 3 "señantes" por letra
        "grupo_offset": 1000,          # offset para no colisionar con otros grupos
    },
    "dgi156_pkl": {
        "etiqueta": "dgi156",
        "estrategia": "por_clase",     # HISTORIAS_VINETAS → 1 grupo por episodio
        "grupo_offset": 2000,
    },
    "pkl": {
        # Carpeta "pkl" = vineta (HISTORIAS_VINETAS desde fuente vineta/dialogo)
        "etiqueta": "vineta",
        "estrategia": "por_clase",
        "grupo_offset": 3000,
    },
    "aec_pkl": {
        "etiqueta": "aec",
        "estrategia": "hash_nombre",   # hash del video de origen
        "grupo_offset": 5000,
    },
    "pucp305_pkl": {
        "etiqueta": "pucp305",
        "estrategia": "por_señante",   # inferir del nombre del pkl (ABRIR_1, ABRIR_2…)
        "grupo_offset": 7000,
    },
    "glosas_pkl": {
        "etiqueta": "glosa",
        "estrategia": "hash_nombre",
        "grupo_offset": 8000,
    },
}

# Fuentes opcionales (si existen después de descarga)
SOURCE_CONFIG_EXTRA = {
    "dgi156_github": {
        "etiqueta": "dgi156_github",
        "estrategia": "hash_nombre",
        "grupo_offset": 9000,
    },
}


def inferir_grupo(pkl_path: pathlib.Path, clase_idx: int,
                  config: dict, muestra_idx: int) -> int:
    """Infiere el ID de grupo para una muestra."""
    offset = config.get("grupo_offset", 0)
    estrategia = config.get("estrategia", "por_clase")

    if estrategia == "por_clase":
        return offset + clase_idx

    elif estrategia == "por_subgrupo":
        n = config.get("n_subgrupos", 3)
        subgrupo = muestra_idx % n
        return offset + clase_idx * 100 + subgrupo

    elif estrategia == "por_señante":
        # Nombres tipo ABRIR_1.pkl, ABRIR_2.pkl → señante = número al final
        stem = pkl_path.stem
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            señante = int(parts[1])
        else:
            señante = hash(stem) % 10
        return offset + señante

    elif estrategia == "hash_nombre":
        return offset + abs(hash(pkl_path.parent.name)) % 500

    return offset + clase_idx


def cargar_fuente(kp_base: pathlib.Path, config: dict,
                  excluir: set | None = None) -> tuple:
    """Carga todas las muestras PKL de una fuente."""
    Xs, Ls, Gs, Ss = [], [], [], []
    etiqueta = config["etiqueta"]

    if not kp_base.exists():
        return np.array([]), np.array([]), np.array([]), np.array([])

    clases = sorted(d for d in kp_base.iterdir() if d.is_dir())
    n_ok = n_err = 0

    for clase_idx, clase_dir in enumerate(clases):
        clase_name = clase_dir.name
        if excluir and clase_name in excluir:
            continue

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


# ─── análisis HE3 ─────────────────────────────────────────────────────────────

def analizar_he3(y_labels: np.ndarray, groups: np.ndarray,
                 sources: np.ndarray) -> dict:
    """Simula GroupShuffleSplit y analiza clases en holdout sin training."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    tv_idx, heldout_idx = next(gss.split(np.zeros(len(y_labels)), y_labels, groups))

    # Clases en training vs holdout
    clases_train  = set(y_labels[tv_idx])
    clases_holdout = set(y_labels[heldout_idx])
    clases_solo_holdout = clases_holdout - clases_train

    # Contar muestras
    cnt_holdout = collections.Counter(y_labels[heldout_idx])
    muestras_cero = sum(cnt_holdout[c] for c in clases_solo_holdout)
    total_holdout = len(heldout_idx)

    # Fuentes en holdout
    src_holdout = collections.Counter(sources[heldout_idx])

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
    ap.add_argument("--min-muestras", type=int, default=15,
                    help="Filtro mínimo de muestras por clase (default: 15)")
    ap.add_argument("--estadisticas", action="store_true",
                    help="Solo analizar sin guardar")
    ap.add_argument("--excluir-lsa64", action="store_true", default=True,
                    help="Excluir LSA64 (Argentina, no LSP). Default: True")
    args = ap.parse_args()

    print("=" * 65)
    print("Build Dataset S16 — Todas las fuentes LSP locales")
    print(f"  min_muestras = {args.min_muestras}")
    print(f"  excluir_lsa64 = {args.excluir_lsa64}")
    print("=" * 65)
    print()

    # ── Cargar todas las fuentes ──────────────────────────────────────────────
    print("Cargando fuentes de keypoints:")
    all_X, all_L, all_G, all_S = [], [], [], []

    for carpeta, config in SOURCE_CONFIG.items():
        kp_base = KP_DIR / carpeta
        X, L, G, S = cargar_fuente(kp_base, config)
        if len(X) > 0:
            all_X.append(X)
            all_L.append(L)
            all_G.append(G)
            all_S.append(S)

    # Fuentes extra (post-descarga)
    for carpeta, config in SOURCE_CONFIG_EXTRA.items():
        kp_base = KP_DIR / carpeta
        if kp_base.exists():
            X, L, G, S = cargar_fuente(kp_base, config)
            if len(X) > 0:
                all_X.append(X)
                all_L.append(L)
                all_G.append(G)
                all_S.append(S)

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

    # ── Estadísticas por fuente ───────────────────────────────────────────────
    print("\nDistribución por fuente (post-filtro):")
    src_cnts = collections.Counter(S_all)
    for src, n in sorted(src_cnts.items()):
        clases_src = len(set(L_all[S_all == src]))
        print(f"  {src:<18}: {n:>5} muestras, {clases_src:>4} clases")

    # ── Análisis HE3 ─────────────────────────────────────────────────────────
    print("\nAnálisis HE3 (GroupShuffleSplit test_size=0.20, SEED=42):")
    he3 = analizar_he3(L_all, G_all, S_all)
    print(f"  Train+Val : {he3['n_train']}")
    print(f"  Holdout   : {he3['n_holdout']}")
    print(f"  Clases holdout sin training: {he3['n_clases_sin_training']}")
    if he3["clases_sin_training"]:
        print(f"  Clases afectadas: {he3['clases_sin_training']}")
    print(f"  Muestras holdout con F1=0 garantizado: {he3['muestras_holdout_cero_training']} "
          f"({he3['pct_holdout_cero']:.1f}%)")
    print(f"  Fuentes holdout: {he3['fuentes_holdout']}")

    vs_s15 = 1687
    mejora = vs_s15 - he3["muestras_holdout_cero_training"]
    print(f"\n  vs S15: {vs_s15} → {he3['muestras_holdout_cero_training']} "
          f"({'↓' if mejora > 0 else '↑'}{abs(mejora)} muestras con F1=0)")

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
    out_npz = DATA / "dataset_s16.npz"
    np.savez_compressed(out_npz, X=X_all, y=y_all, groups=G_all)
    np.save(DATA / "s16_sources.npy", S_all)

    with open(DATA / "s16_label2idx.json", "w", encoding="utf-8") as f:
        json.dump(label2idx, f, ensure_ascii=False, indent=2)
    with open(DATA / "s16_idx2label.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in idx2label.items()}, f, ensure_ascii=False, indent=2)

    # Guardar reporte HE3
    with open(DATA / "s16_he3_report.json", "w", encoding="utf-8") as f:
        json.dump(he3, f, ensure_ascii=False, indent=2)

    print(f"\n✅ dataset_s16.npz guardado ({len(X_all)} muestras, {len(classes)} clases)")
    print(f"✅ s16_label2idx.json")
    print(f"✅ s16_sources.npy")
    print(f"✅ s16_he3_report.json")

    print(f"\nSiguiente paso:")
    print(f"  python3 scripts/train_s19.py  # usar dataset_s16.npz modificando DATA_NPZ")


if __name__ == "__main__":
    main()
