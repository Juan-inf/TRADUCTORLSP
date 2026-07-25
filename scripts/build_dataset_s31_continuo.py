"""build_dataset_s31_continuo.py

Construye un dataset NUEVO a partir de dos recursos ya existentes en el
repositorio pero nunca cruzados entre sí:

  - data/landmarks/*.npy + data/manifest_segments.csv — keypoints ya
    extraídos con MediaPipe en ventanas deslizantes (30 frames, stride 15)
    sobre 26 videos narrativos completos ("Historias vinetas"), generados
    por scripts/preprocess_sliding_window.py. Cada ventana estaba
    etiquetada con la clase del VIDEO ENTERO (p.ej. "vineta_002"), inútil
    para clasificar señas individuales dentro de la narración.
  - data/SRT/SRT_SEGMENTED_SIGN/*.srt — 4,092 glosas anotadas a mano por
    seña individual con timestamp exacto (milisegundos) en esos mismos 26
    videos (falta el video 9, sin landmarks extraídos).

Para cada ventana, se busca la glosa del SRT cuyo rango de tiempo contiene
el CENTRO de la ventana. Si el centro cae en un hueco sin anotación (pausa
entre señas), la ventana se descarta. El resultado es un dataset de señas
individuales pero extraídas de contexto narrativo continuo real — a
diferencia de dataset_s18b (clips ya aislados), esto entrena y evalúa la
tarea real que falla hoy: reconocer una seña dentro de una narración
seguida, no un clip cortado a mano.

Split: a nivel de VIDEO completo (no de ventana) — ventanas contiguas del
mismo video comparten hasta 15 de 30 frames, así que mezclarlas entre train
y test filtraría datos. Se reservan videos completos nunca vistos en
entrenamiento para poder correr una evaluación WER real después
(scripts/evaluar_wer_s31.py) reconstruyendo la secuencia de glosas
predichas sobre el video entero y comparándola con el SRT real.

Uso:
    python3 scripts/build_dataset_s31_continuo.py [--min-muestras N]
"""
import argparse
import json
import pathlib
import re
import unicodedata
import collections

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LAND_DIR = DATA / "landmarks"
SRT_DIR = DATA / "SRT" / "SRT_SEGMENTED_SIGN"

import sys
sys.path.insert(0, str(ROOT))
from src.features.landmarks import kp_seq_to_features

SEED = 42
N_VIDEOS_WER_HOLDOUT = 5  # videos completos reservados solo para evaluación WER


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def limpiar_gloss(texto: str) -> str:
    """Normaliza el texto de un bloque SRT a una etiqueta de clase única.

    Los anotadores usaron variantes/alternativas ('UNA/LA', 'PEQUEÑA O
    NIÑA'), paréntesis para la raíz de una flexión ('UN(A)'), comillas y
    signos de interrogación sueltos ('QUÉ PASÓ?'). Se toma la primera
    alternativa como etiqueta canónica — perder matices lingüísticos es
    aceptable para clasificación, mantener 2 clases para la misma seña no
    lo es. Deletreo manual (S-O-F-I-A, guiones entre letras sueltas) se
    descarta explícitamente: es una secuencia de letras, no una seña única,
    y confundiría a un clasificador de ventana fija de 30 frames.
    """
    t = texto.strip().strip('"').strip()
    if not t:
        return ""
    # deletreo manual: 2+ guiones entre letras sueltas (S-O-F-I-A)
    if re.fullmatch(r"([A-ZÑÁÉÍÓÚ]-){2,}[A-ZÑÁÉÍÓÚ]?", t.upper()):
        return ""
    t = t.split("/")[0].strip()
    t = re.split(r"\s+O\s+", t, maxsplit=1)[0].strip()
    t = t.rstrip("?").rstrip("!").strip()
    t = re.sub(r"\s+", " ", t)
    return nfc(t.upper())


def parse_srt(path: pathlib.Path) -> list[tuple[float, float, str]]:
    """Devuelve [(inicio_seg, fin_seg, gloss_limpia), ...], sin entradas vacías."""
    texto = path.read_text(encoding="utf-8", errors="replace")
    bloques = re.split(r"\n\s*\n", texto.strip())
    out = []
    tc_re = re.compile(
        r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})"
    )

    def a_seg(h, m, s, ms):
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

    for b in bloques:
        lineas = [l for l in b.split("\n") if l.strip() != ""]
        if len(lineas) < 3:
            continue
        # el índice puede venir en la línea 0 o ausente — buscar timecode en cualquiera
        tc_line = next((l for l in lineas if tc_re.search(l)), None)
        if tc_line is None:
            continue
        m = tc_re.search(tc_line)
        t0 = a_seg(*m.groups()[0:4])
        t1 = a_seg(*m.groups()[4:8])
        texto_idx = lineas.index(tc_line)
        gloss_raw = " ".join(lineas[texto_idx + 1:]) if texto_idx + 1 < len(lineas) else ""
        gloss = limpiar_gloss(gloss_raw)
        if gloss:
            out.append((t0, t1, gloss))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-muestras", type=int, default=15)
    args = ap.parse_args()

    print("=" * 65)
    print("build_dataset_s31_continuo — cruce SRT (glosas reales) x landmarks")
    print("=" * 65)

    manifest = pd.read_csv(DATA / "manifest.csv")
    fps_por_video = dict(zip(manifest["num_vineta"], manifest["fps"]))

    segs = pd.read_csv(DATA / "manifest_segments.csv")
    print(f"Ventanas en manifest_segments.csv: {len(segs)}")

    srt_por_video = {}
    for srt_path in sorted(SRT_DIR.glob("*.srt")):
        m = re.search(r"\((\d+)\)", srt_path.name)
        if not m:
            continue
        num = int(m.group(1))
        entradas = parse_srt(srt_path)
        if entradas:
            srt_por_video[num] = entradas
    print(f"Videos con SRT parseado: {len(srt_por_video)}  "
          f"(total glosas: {sum(len(v) for v in srt_por_video.values())})")

    videos_disponibles = sorted(set(segs["num_vineta"]) & set(srt_por_video.keys()))
    print(f"Videos con landmarks Y SRT: {len(videos_disponibles)}")

    rng = np.random.RandomState(SEED)
    videos_wer = sorted(rng.choice(videos_disponibles, size=min(N_VIDEOS_WER_HOLDOUT, len(videos_disponibles)),
                                    replace=False).tolist())
    videos_train_pool = [v for v in videos_disponibles if v not in videos_wer]
    print(f"Videos reservados 100% para evaluación WER (nunca en train): {videos_wer}")
    print(f"Videos disponibles para dataset de entrenamiento: {len(videos_train_pool)}")

    Xs, Ls, Gs = [], [], []
    sin_gloss = 0
    for num_vineta, grp in segs.groupby("num_vineta"):
        if num_vineta not in videos_train_pool:
            continue
        fps = fps_por_video.get(num_vineta, 29.97)
        entradas = srt_por_video[num_vineta]
        # búsqueda lineal ordenada por tiempo — los SRT ya vienen ordenados
        entradas = sorted(entradas, key=lambda e: e[0])
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
                sin_gloss += 1
                continue
            kp_path = ROOT / row["kp_path"]
            if not kp_path.exists():
                continue
            kp_seq = np.load(kp_path)  # [30, 75, 3] = [T, N_KP, xyz]
            feat = kp_seq_to_features(kp_seq)  # [30, 150]
            Xs.append(feat)
            Ls.append(gloss)
            Gs.append(9000 + int(num_vineta))

    print(f"\nVentanas sin glosa (caen en pausa/hueco sin anotación): {sin_gloss}")
    print(f"Ventanas etiquetadas: {len(Xs)}  |  glosas únicas (antes de filtro): {len(set(Ls))}")

    X_all = np.stack(Xs).astype(np.float32)
    L_all = np.array(Ls)
    G_all = np.array(Gs, dtype=np.int32)

    cnts = collections.Counter(L_all.tolist())
    mask = np.array([cnts[l] >= args.min_muestras for l in L_all])
    X_all, L_all, G_all = X_all[mask], L_all[mask], G_all[mask]
    print(f"Después de min{args.min_muestras}: {len(X_all)} muestras, {len(set(L_all.tolist()))} clases")

    print("\nDistribución (top 20 por frecuencia):")
    for gloss, n in collections.Counter(L_all.tolist()).most_common(20):
        print(f"  {gloss:<20}: {n}")

    classes = sorted(set(L_all.tolist()))
    label2idx = {c: i for i, c in enumerate(classes)}
    idx2label = {i: c for c, i in label2idx.items()}
    y_all = np.array([label2idx[l] for l in L_all.tolist()], dtype=np.int64)

    np.savez_compressed(DATA / "dataset_s31.npz", X=X_all, y=y_all, groups=G_all)
    with open(DATA / "s31_label2idx.json", "w", encoding="utf-8") as f:
        json.dump(label2idx, f, ensure_ascii=False, indent=2)
    with open(DATA / "s31_idx2label.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in idx2label.items()}, f, ensure_ascii=False, indent=2)
    with open(DATA / "s31_videos_wer_holdout.json", "w", encoding="utf-8") as f:
        json.dump({"videos_wer": videos_wer, "videos_train_pool": videos_train_pool}, f, indent=2)

    print(f"\n✅ dataset_s31.npz  ({len(X_all)} muestras, {len(classes)} clases)")
    print(f"✅ s31_label2idx.json / s31_idx2label.json")
    print(f"✅ s31_videos_wer_holdout.json — {len(videos_wer)} videos reservados para WER")


if __name__ == "__main__":
    main()
