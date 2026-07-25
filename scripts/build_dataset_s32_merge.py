"""build_dataset_s32_merge.py

S31 entrenó AISLADO sobre solo 1,538 muestras (contexto narrativo real vía
SRT × landmarks) y perdió contra producción en WER real (2.827 vs 1.763,
ver ENTREGABLE_PLAN_DE_DESPLIEGUE_S13.md §12) — la causa diagnosticada fue
volumen insuficiente para generalización persona-independiente, no que el
enfoque esté mal. Este sprint implementa el camino recomendado en §12.5:
combinar las muestras de contexto narrativo (S31) con el corpus grande de
clips ya aislados (mismas fuentes que dataset_s18, sin el filtro
--solo-vocab-actual, para no descartar clases de S31 que no estén en el
vocabulario de 96 de S27) en un solo dataset, con la MISMA técnica de
rebalanceo de grupos cross-source ya validada en S17/S18
(balancear_grupos_cruzados) — extendida para incluir la nueva fuente
"s31_continuo" en los pares a rebalancear, evitando que el mismo video
narrativo aparezca en train bajo una fuente y en holdout bajo otra (la
causa raíz original del bug de S16/S17).

Uso:
    python3 scripts/build_dataset_s32_merge.py [--min-muestras N]
"""
import argparse
import json
import pathlib
import re
import sys
import unicodedata
import collections

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
KP_DIR = DATA / "Keypoints"

sys.path.insert(0, str(ROOT))
from src.features.landmarks import kp_seq_to_features
import scripts.build_dataset_s18 as b18
from scripts.build_dataset_s31_continuo import parse_srt, SRT_DIR, limpiar_gloss

N_FRAMES = 30
N_DIMS = 150
SEED = 42


def nfc(s):
    return unicodedata.normalize("NFC", s)


def cargar_fuente_s31_continuo():
    """Réplica de la lógica de build_dataset_s31_continuo.py PERO sin el
    filtro min-muestras (se aplica al final, sobre el dataset ya combinado)
    y devolviendo (X, L, G, S) en el mismo formato que cargar_fuente() de
    build_dataset_s18, para poder concatenar directamente."""
    manifest = pd.read_csv(DATA / "manifest.csv")
    fps_por_video = dict(zip(manifest["num_vineta"], manifest["fps"]))
    segs = pd.read_csv(DATA / "manifest_segments.csv")

    srt_por_video = {}
    for srt_path in sorted(SRT_DIR.glob("*.srt")):
        m = re.search(r"\((\d+)\)", srt_path.name)
        if not m:
            continue
        num = int(m.group(1))
        entradas = parse_srt(srt_path)
        if entradas:
            srt_por_video[num] = sorted(entradas, key=lambda e: e[0])

    with open(DATA / "s31_videos_wer_holdout.json", encoding="utf-8") as f:
        wer_info = json.load(f)
    videos_excluidos = set(wer_info["videos_wer"])  # nunca entrenar con estos

    Xs, Ls, Gs = [], [], []
    for num_vineta, grp in segs.groupby("num_vineta"):
        if num_vineta not in srt_por_video or num_vineta in videos_excluidos:
            continue
        fps = fps_por_video.get(num_vineta, 29.97)
        entradas = srt_por_video[num_vineta]
        for _, row in grp.iterrows():
            centro_seg = (row["start_frame"] + row["end_frame"]) / 2.0 / fps
            gloss = None
            for t0, t1, g in entradas:
                if t0 <= centro_seg <= t1:
                    gloss = g
                    break
                if t0 > centro_seg:
                    break
            if gloss is None:
                continue
            kp_path = ROOT / row["kp_path"]
            if not kp_path.exists():
                continue
            kp_seq = np.load(kp_path)
            feat = kp_seq_to_features(kp_seq)
            Xs.append(feat)
            Ls.append(gloss)
            Gs.append(9000 + int(num_vineta))

    print(f"  s31_continuo      : {len(Xs):>5} muestras, {len(set(Ls)):>4} glosas  "
          f"[{len(videos_excluidos)} videos excluidos — reservados para WER]")
    if not Xs:
        return np.array([]), np.array([]), np.array([]), np.array([])
    return (np.stack(Xs).astype(np.float32), np.array(Ls),
            np.array(Gs, dtype=np.int32), np.array(["s31_continuo"] * len(Xs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-muestras", type=int, default=15)
    args = ap.parse_args()

    print("=" * 65)
    print("build_dataset_s32_merge — dataset_s18 (raw) + s31_continuo")
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

    print("\nCargando fuente NUEVA (S32 — SRT x landmarks, contexto narrativo continuo):")
    X, L, G, S = cargar_fuente_s31_continuo()
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

    print("\nAplicando FIX cross-source (grupos balanceados, extendido con s31_continuo):")
    G_all = b18.balancear_grupos_cruzados(L_all, G_all, S_all, pares=[
        ("dgi156", "vineta"),
        ("dgi156_gloss", "vineta_gloss"),
        ("dgi156_gloss", "s31_continuo"),
        ("vineta_gloss", "s31_continuo"),
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

    np.savez_compressed(DATA / "dataset_s32.npz", X=X_all, y=y_all, groups=G_all)
    np.save(DATA / "s32_sources.npy", S_all)
    with open(DATA / "s32_label2idx.json", "w", encoding="utf-8") as f:
        json.dump(label2idx, f, ensure_ascii=False, indent=2)
    with open(DATA / "s32_idx2label.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in idx2label.items()}, f, ensure_ascii=False, indent=2)
    with open(DATA / "s32_he3_report.json", "w", encoding="utf-8") as f:
        json.dump(he3, f, ensure_ascii=False, indent=2)

    print(f"\n✅ dataset_s32.npz  ({len(X_all)} muestras, {len(classes)} clases)")
    print(f"✅ s32_label2idx.json / s32_idx2label.json / s32_sources.npy / s32_he3_report.json")


if __name__ == "__main__":
    main()
