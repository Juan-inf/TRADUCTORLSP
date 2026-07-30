"""build_dataset_s35_merge.py

Extiende build_dataset_s32_merge.py (dataset_s18 raw + s31_continuo, SRT x
landmarks) agregando una TERCERA fuente nueva: s34_eaf — segmentos de glosa
extraídos de los 274 archivos data/Glosas/*/*_ORACION_*.eaf (anotación
lingüística ELAN, no derivada de audio), ver build_dataset_s34_eaf.py.

Riesgo de fuga cruzada: los 274 ORACION provienen de la MISMA carpeta
data/Glosas/ que ya se usa como fuente "glosa" (glosas_pkl, clips de seña
aislada tipo HOMBRE_1.mp4) dentro de dataset_s18/s32 — mismo señante/sesión
de grabación bajo dos esquemas de etiquetado distintos (seña aislada vs.
oración continua). Se extiende balancear_grupos_cruzados con el par
("glosa", "s34_eaf") para que ambas vistas del mismo origen físico caigan
siempre juntas en train o en holdout, replicando el fix ya validado en
S17/S18/S32 para dgi156/vineta y sus variantes.

Uso:
    python3 scripts/build_dataset_s35_merge.py [--min-muestras N]
"""
import argparse
import json
import pathlib
import sys
import unicodedata
import collections

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
KP_DIR = DATA / "Keypoints"

sys.path.insert(0, str(ROOT))
import scripts.build_dataset_s18 as b18
from scripts.build_dataset_s31_continuo import parse_srt  # noqa: F401 (mantiene compat import)
from scripts.build_dataset_s32_merge import cargar_fuente_s31_continuo

SEED = 42


def nfc(s):
    return unicodedata.normalize("NFC", s)


# IX = glosa lingüística de señalamiento/índice pronominal (apunta a un
# referente espacial variable: persona, lugar, objeto ya mencionado) — a
# diferencia de toda otra glosa de este dataset (señas léxicas de patrón
# visual fijo, ej. "COMER", "DINERO"), IX no tiene una forma visual estable:
# la dirección del señalamiento cambia con el referente de cada oración.
# Era la 2a clase más frecuente (113/668 muestras, ~17%) y probablemente
# fue ruido de etiqueta puro para un clasificador de ventana fija — mismo
# criterio ya aplicado a HISTORIAS_VINETAS_* (excluir clases sin forma
# visual fija de "seña única"). Ver resultado negativo de S35 (WER peor
# que S33 en ambos benchmarks) que motivó este ajuste.
GLOSAS_EXCLUIDAS_S34 = {"IX"}


def cargar_fuente_s34_eaf():
    """Carga dataset_s34.npz (ya extraído por build_dataset_s34_eaf.py) y lo
    devuelve en el mismo formato (X, L, G, S) que las demás fuentes, listo
    para concatenar. Los grupos ya vienen asignados por oración física
    (20000 + índice de archivo .eaf), sin necesidad de re-inferirlos."""
    npz_path = DATA / "dataset_s34.npz"
    if not npz_path.exists():
        print("  s34_eaf           :     0 muestras — dataset_s34.npz no encontrado, saltando")
        return np.array([]), np.array([]), np.array([]), np.array([])

    d = np.load(npz_path)
    X, y, G = d["X"], d["y"], d["groups"]
    idx2label = json.load(open(DATA / "s34_idx2label.json", encoding="utf-8"))
    idx2label = {int(k): v for k, v in idx2label.items()}
    L = np.array([idx2label[i] for i in y.tolist()])
    S = np.array(["s34_eaf"] * len(X))

    keep = np.array([l not in GLOSAS_EXCLUIDAS_S34 for l in L])
    n_excl = (~keep).sum()
    X, L, G, S = X[keep], L[keep], G[keep], S[keep]

    print(f"  s34_eaf           : {len(X):>5} muestras, {len(set(L)):>4} glosas  "
          f"[oraciones ELAN, glosa x milisegundo — {n_excl} muestras de {GLOSAS_EXCLUIDAS_S34} excluidas, "
          f"10 oraciones reservadas para WER]")
    return X.astype(np.float32), L, G.astype(np.int32), S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-muestras", type=int, default=15)
    args = ap.parse_args()

    print("=" * 65)
    print("build_dataset_s35_merge — dataset_s18 (raw) + s31_continuo + s34_eaf")
    print("=" * 65)
    print()
    print("Cargando fuentes de keypoints (vista narrativa, igual que S18):")
    all_X, all_L, all_G, all_S = [], [], [], []

    for carpeta, config in b18.SOURCE_CONFIG.items():
        kp_base = KP_DIR / carpeta
        X, L, G, S = b18.cargar_fuente(kp_base, config)
        if len(X) > 0:
            all_X.append(X); all_L.append(L); all_G.append(G); all_S.append(S)

    for carpeta, config in b18.SOURCE_CONFIG_EXTRA.items():
        kp_base = KP_DIR / carpeta
        if kp_base.exists():
            X, L, G, S = b18.cargar_fuente(kp_base, config)
            if len(X) > 0:
                all_X.append(X); all_L.append(L); all_G.append(G); all_S.append(S)

    print("\nCargando fuentes de glosa individual (S18):")
    for carpeta, config in b18.SOURCE_CONFIG_GLOSAS.items():
        kp_base = KP_DIR / carpeta
        X, L, G, S = b18.cargar_fuente_glosas(kp_base, config)
        if len(X) > 0:
            all_X.append(X); all_L.append(L); all_G.append(G); all_S.append(S)

    print("\nCargando fuente S32 (SRT x landmarks, contexto narrativo continuo):")
    X, L, G, S = cargar_fuente_s31_continuo()
    if len(X) > 0:
        all_X.append(X); all_L.append(L); all_G.append(G); all_S.append(S)

    print("\nCargando fuente NUEVA (S35 — oraciones ELAN x milisegundo, ver build_dataset_s34_eaf.py):")
    X, L, G, S = cargar_fuente_s34_eaf()
    if len(X) > 0:
        all_X.append(X); all_L.append(L); all_G.append(G); all_S.append(S)

    X_all = np.concatenate(all_X, axis=0)
    L_all = np.concatenate(all_L)
    G_all = np.concatenate(all_G)
    S_all = np.concatenate(all_S)
    L_all = np.array([nfc(l) for l in L_all])

    print(f"\nTotal antes de filtro: {len(X_all)} muestras, {len(set(L_all))} clases")

    cnts = collections.Counter(L_all)
    mask_pre = np.array([cnts[l] >= args.min_muestras for l in L_all])
    print(f"(antes del rebalanceo de grupos) tras min{args.min_muestras}: "
          f"{mask_pre.sum()} muestras, {len(set(L_all[mask_pre]))} clases")

    print("\nAplicando FIX cross-source (grupos balanceados, extendido con s34_eaf):")
    G_all = b18.balancear_grupos_cruzados(L_all, G_all, S_all, pares=[
        ("dgi156", "vineta"),
        ("dgi156_gloss", "vineta_gloss"),
        ("dgi156_gloss", "s31_continuo"),
        ("vineta_gloss", "s31_continuo"),
        ("glosa", "s34_eaf"),
        ("dgi156_gloss", "s34_eaf"),
        ("vineta_gloss", "s34_eaf"),
    ])

    mask = np.array([cnts[l] >= args.min_muestras for l in L_all])
    X_all, L_all, G_all, S_all = X_all[mask], L_all[mask], G_all[mask], S_all[mask]
    print(f"\nDespués de min{args.min_muestras}: {len(X_all)} muestras, {len(set(L_all))} clases")

    print("\nDistribución por fuente (post-filtro):")
    src_cnts = collections.Counter(S_all)
    for src, n in sorted(src_cnts.items()):
        clases_src = len(set(L_all[S_all == src]))
        print(f"  {src:<18}: {n:>5} muestras, {clases_src:>4} clases")

    print("\nAnálisis HE3:")
    he3 = b18.analizar_he3(L_all, G_all, S_all)
    print(f"  Train+Val : {he3['n_train']}")
    print(f"  Holdout   : {he3['n_holdout']}")
    print(f"  Clases holdout sin training: {he3['n_clases_sin_training']}")
    print(f"  Muestras holdout F1=0 garantizado: {he3['muestras_holdout_cero_training']} "
          f"({he3['pct_holdout_cero']:.1f}%)")

    classes = sorted(set(L_all.tolist()))
    label2idx = {c: i for i, c in enumerate(classes)}
    idx2label = {i: c for c, i in label2idx.items()}
    y_all = np.array([label2idx[l] for l in L_all.tolist()], dtype=np.int64)

    np.savez_compressed(DATA / "dataset_s35.npz", X=X_all, y=y_all, groups=G_all)
    np.save(DATA / "s35_sources.npy", S_all)
    with open(DATA / "s35_label2idx.json", "w", encoding="utf-8") as f:
        json.dump(label2idx, f, ensure_ascii=False, indent=2)
    with open(DATA / "s35_idx2label.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in idx2label.items()}, f, ensure_ascii=False, indent=2)
    with open(DATA / "s35_he3_report.json", "w", encoding="utf-8") as f:
        json.dump(he3, f, ensure_ascii=False, indent=2)

    print(f"\n✅ dataset_s35.npz  ({len(X_all)} muestras, {len(classes)} clases)")
    print(f"✅ s35_label2idx.json / s35_idx2label.json / s35_sources.npy / s35_he3_report.json")


if __name__ == "__main__":
    main()
