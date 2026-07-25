"""build_dataset_s18.py

Extiende dataset_s17: además de las fuentes ya usadas (que tratan cada
carpeta Historias_vinetas_N como UNA clase narrativa), agrega una segunda
lectura de las MISMAS carpetas (dgi156_pkl, pkl/vineta) usando el nombre de
archivo individual como etiqueta real — cada .pkl dentro de una carpeta
narrativa es en realidad una glosa (seña) segmentada, no la narrativa
completa. Esa información nunca se usó porque cargar_fuente() (S17 y
anteriores) solo lee el nombre de la carpeta.

Encontrado en vivo (2026-07-18) auditando datos sin usar para el objetivo
de F1: combinando dgi156_pkl + pkl(vineta) a nivel de archivo hay 744 glosas
únicas reales (7151 archivos) — 27 ya están en el vocabulario de 96 clases
(+1344 muestras extra, ej. QUÉ pasa de 15→186), y 85 son candidatas a clase
nueva (≥15 muestras: MUJER, HOMBRE, CAMINAR, MAMÁ, CASA, ...).

Decisión de diseño (confirmada con el usuario): NO se elimina
HISTORIAS_VINETAS_N como clase — se agregan las glosas reales COMO CLASES
ADICIONALES. El mismo archivo físico contribuye entonces a dos vistas
distintas (la narrativa completa Y la glosa individual); ambas vistas
comparten el mismo número de grupo (basado en el índice de su carpeta
narrativa) para que GroupShuffleSplit las mantenga siempre juntas en
train u holdout — evita que el modelo "vea" en entrenamiento, bajo una
etiqueta, exactamente los mismos landmarks que después evalúa en holdout
bajo la otra etiqueta.

Uso:
    python3 scripts/build_dataset_s18.py [--min-muestras N] [--estadisticas]
"""

import argparse
import json
import pathlib
import pickle
import re
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


def limpiar_gloss(stem: str) -> str:
    """'QUÉ__855' -> 'QUÉ' ; 'QUÉ_-QUÉ__851' -> 'QUÉ' ; 'YO_539' -> 'YO'."""
    stem = re.sub(r'_\d+$', '', stem)   # quita sufijo numérico final
    stem = stem.split("-")[0]           # glosa compuesta "A-B" -> "A"
    stem = stem.rstrip("_")             # artefacto de guion bajo colgante
    return stem


# ─── configuración de fuentes (idéntica a S17) ────────────────────────────────

SOURCE_CONFIG = {
    "abecedario_pkl": {
        "etiqueta": "abecedario",
        "estrategia": "por_subgrupo",
        "n_subgrupos": 20,
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
        "n_subgrupos": 15,
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

# NUEVO EN S18: segunda lectura de las mismas carpetas narrativas, a nivel de
# glosa individual (nombre de archivo). Mismos offsets base que su
# contraparte narrativa +100000, para poder derivar el mismo grupo por video.
SOURCE_CONFIG_GLOSAS = {
    "dgi156_pkl": {"etiqueta": "dgi156_gloss", "grupo_offset": 2000},
    "pkl":        {"etiqueta": "vineta_gloss", "grupo_offset": 3000},
}
GLOSA_GRUPO_OFFSET = 100000  # espacio de grupos separado del resto


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


def cargar_fuente_glosas(kp_base: pathlib.Path, config: dict) -> tuple:
    """NUEVO EN S18 — segunda lectura de carpetas narrativas a nivel de
    glosa individual. El grupo de cada muestra es el MISMO que usaría la
    vista narrativa para ese archivo (offset + índice de carpeta), para que
    ambas vistas del mismo video físico caigan siempre juntas en
    train/holdout."""
    Xs, Ls, Gs, Ss = [], [], [], []
    etiqueta = config["etiqueta"]
    offset = config.get("grupo_offset", 0)

    if not kp_base.exists():
        return np.array([]), np.array([]), np.array([]), np.array([])

    carpetas = sorted(d for d in kp_base.iterdir() if d.is_dir())
    n_ok = n_err = n_skip = 0

    for carpeta_idx, carpeta_dir in enumerate(carpetas):
        grupo = offset + carpeta_idx  # idéntico al que usa cargar_fuente() "por_clase"
        for pkl_path in sorted(carpeta_dir.glob("*.pkl")):
            gloss = limpiar_gloss(pkl_path.stem)
            if len(gloss) <= 1:
                n_skip += 1
                continue
            seq = pkl_to_sequence(pkl_path)
            if seq is None:
                n_err += 1
                continue
            Xs.append(seq)
            Ls.append(gloss.upper())
            Gs.append(grupo)
            Ss.append(etiqueta)
            n_ok += 1

    n_clases = len(set(Ls))
    print(f"  {etiqueta:<18}: {n_ok:>5} muestras, {n_clases:>4} glosas  [{n_err} errores, {n_skip} sin nombre válido]")

    if not Xs:
        return np.array([]), np.array([]), np.array([]), np.array([])

    return (np.array(Xs, dtype=np.float32),
            np.array(Ls),
            np.array(Gs, dtype=np.int32),
            np.array(Ss))


# ─── FIX S17/S18: unificar grupos para clases multi-fuente ───────────────────

def balancear_grupos_cruzados(L_all: np.ndarray, G_all: np.ndarray,
                              S_all: np.ndarray, pares: list,
                              seed: int = SEED) -> np.ndarray:
    """Generalización del fix S17 (dgi156∩vineta): para cada par de fuentes
    dado, cualquier clase que aparezca en AMBAS se resub-agrupa en
    sub-grupos balanceados por posición, evitando que GroupShuffleSplit
    mande toda una fuente a training y la otra a holdout para esa clase
    (el bug original de S16, F1≈0 en HV_23 por contaminación cross-source)."""
    G_new = G_all.copy()
    rng   = np.random.default_rng(seed)
    N_SUBGRUPOS = 10
    base_offset = 20000

    for src_a, src_b in pares:
        clases_a = set(L_all[S_all == src_a])
        clases_b = set(L_all[S_all == src_b])
        cruce = sorted(clases_a & clases_b)
        if not cruce:
            print(f"  No hay clases {src_a}∩{src_b} — sin cambio.")
            continue

        print(f"\n  [FIX] Sub-grupos balanceados {src_a}∩{src_b}: {len(cruce)} clases")
        for class_idx, label in enumerate(cruce):
            mask    = L_all == label
            indices = np.where(mask)[0]
            indices_shuffled = rng.permutation(indices)
            base = base_offset + abs(hash((src_a, src_b, label))) % 1_000_000
            for pos, idx in enumerate(indices_shuffled):
                G_new[idx] = base + (pos % N_SUBGRUPOS)
            n_a = int((mask & (S_all == src_a)).sum())
            n_b = int((mask & (S_all == src_b)).sum())
            print(f"    {label:<25}  {src_a}={n_a:3d}  {src_b}={n_b:3d}")
        base_offset += 2_000_000  # evitar colisiones entre pares distintos

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
    ap.add_argument("--solo-vocab-actual", action="store_true",
                     help="Fase 1: densifica únicamente las 96 clases del vocabulario "
                          "actual (data/s27_label2idx.json) con las glosas reales "
                          "extraídas de dgi156_pkl/vineta; no agrega clases nuevas.")
    ap.add_argument("--excluir-dgi156-gloss", action="store_true",
                     help="Encontrado en vivo (2026-07-18): los clips de "
                          "dgi156_gloss tienen SIEMPRE exactamente 30 frames "
                          "(ventana fija, no segmentación real de la seña) — a "
                          "diferencia de vineta_gloss, que sí varía naturalmente "
                          "(1-34 frames). Probable ruido/mal etiquetado. Con este "
                          "flag se excluye esa fuente, dejando solo vineta_gloss.")
    args = ap.parse_args()

    print("=" * 65)
    print("Build Dataset S18 — S17 + glosas reales de carpetas narrativas")
    print(f"  min_muestras = {args.min_muestras}")
    print("=" * 65)
    print()

    print("Cargando fuentes de keypoints (vista narrativa, igual que S17):")
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

    print("\nCargando fuentes NUEVAS (vista glosa individual — S18):")
    for carpeta, config in SOURCE_CONFIG_GLOSAS.items():
        if args.excluir_dgi156_gloss and config["etiqueta"] == "dgi156_gloss":
            print(f"  {config['etiqueta']:<18}: excluida (--excluir-dgi156-gloss, ventanas fijas sospechosas)")
            continue
        kp_base = KP_DIR / carpeta
        X, L, G, S = cargar_fuente_glosas(kp_base, config)
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

    # Normalizar Unicode (NFC) en las etiquetas — sin esto, tildes en forma
    # decompuesta vs precompuesta crean clases duplicadas invisibles
    # (encontrado en vivo: 'AHÍ' y 'CÓMO' aparecían 2 veces en S28).
    import unicodedata
    def _nfc(s): return unicodedata.normalize("NFC", s)
    L_all = np.array([_nfc(l) for l in L_all])

    if args.solo_vocab_actual:
        vocab_actual_path = DATA / "s27_label2idx.json"
        with open(vocab_actual_path, encoding="utf-8") as f:
            vocab_actual = {_nfc(k).upper() for k in json.load(f).keys()}
        mask_vocab = np.array([_nfc(l).upper() in vocab_actual for l in L_all])
        print(f"\n[--solo-vocab-actual] Filtrando a las {len(vocab_actual)} clases de "
              f"s27_label2idx.json: {len(X_all)} -> {mask_vocab.sum()} muestras")
        X_all, L_all, G_all, S_all = X_all[mask_vocab], L_all[mask_vocab], G_all[mask_vocab], S_all[mask_vocab]

    print(f"\nTotal antes de filtro: {len(X_all)} muestras, {len(set(L_all))} clases")

    cnts = collections.Counter(L_all)
    mask = np.array([cnts[l] >= args.min_muestras for l in L_all])
    X_all = X_all[mask]
    L_all = L_all[mask]
    G_all = G_all[mask]
    S_all = S_all[mask]
    print(f"Total después de min{args.min_muestras}: {len(X_all)} muestras, {len(set(L_all))} clases")

    print("\nDistribución por fuente (post-filtro):")
    src_cnts = collections.Counter(S_all)
    for src, n in sorted(src_cnts.items()):
        clases_src = len(set(L_all[S_all == src]))
        print(f"  {src:<18}: {n:>5} muestras, {clases_src:>4} clases")

    print("\nAplicando FIX cross-source (grupos balanceados, generalizado S17+S18):")
    G_all = balancear_grupos_cruzados(L_all, G_all, S_all, pares=[
        ("dgi156", "vineta"),              # fix original S17 (vista narrativa)
        ("dgi156_gloss", "vineta_gloss"),  # mismo fix, vista glosa (nuevo S18)
    ])

    print("\nAnálisis HE3 (con grupos S18):")
    he3 = analizar_he3(L_all, G_all, S_all)
    print(f"  Train+Val : {he3['n_train']}")
    print(f"  Holdout   : {he3['n_holdout']}")
    print(f"  Clases holdout sin training: {he3['n_clases_sin_training']}")
    if he3["clases_sin_training"]:
        print(f"  Clases afectadas: {he3['clases_sin_training'][:20]}"
              f"{'...' if len(he3['clases_sin_training']) > 20 else ''}")
    print(f"  Muestras holdout con F1=0 garantizado: {he3['muestras_holdout_cero_training']} "
          f"({he3['pct_holdout_cero']:.1f}%)")
    print(f"  Fuentes holdout: {he3['fuentes_holdout']}")

    if args.estadisticas:
        print("\n[--estadisticas] No se guarda nada.")
        return

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

    tag = "s18" + ("b" if args.solo_vocab_actual else "") + ("c" if args.excluir_dgi156_gloss else "")
    np.savez_compressed(DATA / f"dataset_{tag}.npz", X=X_all, y=y_all, groups=G_all)
    np.save(DATA / f"{tag}_sources.npy", S_all)

    with open(DATA / f"{tag}_label2idx.json", "w", encoding="utf-8") as f:
        json.dump(label2idx, f, ensure_ascii=False, indent=2)
    with open(DATA / f"{tag}_idx2label.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in idx2label.items()}, f, ensure_ascii=False, indent=2)
    with open(DATA / f"{tag}_he3_report.json", "w", encoding="utf-8") as f:
        json.dump(he3, f, ensure_ascii=False, indent=2)

    print(f"\n✅ dataset_{tag}.npz  ({len(X_all)} muestras, {len(classes)} clases)")
    print(f"✅ {tag}_label2idx.json")
    print(f"✅ {tag}_sources.npy")
    print(f"✅ {tag}_he3_report.json")


if __name__ == "__main__":
    main()
