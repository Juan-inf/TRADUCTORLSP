"""evaluar_wer_s31.py — Primera medición REAL de WER en este proyecto.

Corre sobre los 5 videos narrativos completos reservados en
s31_videos_wer_holdout.json (nunca vistos en entrenamiento, ni de S31 ni de
ningún sprint anterior — son "Historias vinetas" completas). Para cada uno:

  1. Reconstruye la secuencia de glosas PREDICHA recorriendo las ventanas
     deslizantes ya extraídas en data/landmarks/ (30 frames, stride 15,
     mismo criterio que la demo en vivo), clasificando cada una y colapsando
     detecciones consecutivas iguales (una "racha" de la misma predicción =
     una detección), igual que el criterio real de uso continuo.
  2. Reconstruye la secuencia de glosas REAL desde el SRT correspondiente.
  3. Calcula WER (distancia de edición de Levenshtein / len(referencia)) —
     restringido al vocabulario del modelo evaluado, para no penalizar por
     clases que nunca pudo predecir por diseño.

Corre DOS modelos sobre los mismos 5 videos para comparación directa:
  - S31   (bilstm_s31.onnx)      — entrenado sobre glosas en contexto continuo
  - v4+S29 (ensemble, producción) — el pipeline real que usa la demo hoy,
    incluida SegmentadorPausas (no ventana fija) — para saber si S31 mejora
    de verdad sobre lo que ya está en producción, no solo sobre un baseline
    de juguete.

Uso:
    python3 scripts/evaluar_wer_s31.py
"""
import json
import pathlib
import sys
import unicodedata

import numpy as np
import onnxruntime as ort
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT))

from scripts.build_dataset_s31_continuo import parse_srt, SRT_DIR
from src.features.segmentacion import SegmentadorPausas
from src.features.landmarks import kp_seq_to_features, normalize_sample, resample_to_n_frames

N_FRAMES = 30
STRIDE = 15
CONF_UMBRAL = 0.20


def nfc(s):
    return unicodedata.normalize("NFC", s)


def levenshtein(ref: list, hyp: list) -> int:
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            tmp = dp[j]
            costo = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + costo)
            prev = tmp
    return dp[m]


def cargar_modelo(onnx_path, label2idx_path):
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    l2i = json.load(open(label2idx_path, encoding="utf-8"))
    i2l = {v: nfc(k).upper() for k, v in l2i.items()}
    return sess, i2l


def predecir_ventana(sess, i2l, feat):
    logits = sess.run(None, {"sequence": feat})[0][0]
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    idx = int(probs.argmax())
    return i2l.get(idx, "?"), float(probs[idx])


def secuencia_predicha_s31(num_vineta, sess, i2l, vocab_permitido):
    """Recorre las ventanas ya extraídas de data/landmarks/ en orden temporal,
    clasifica cada una con el modelo S31, y colapsa rachas consecutivas
    iguales — el mismo criterio con el que un usuario real vería la
    transcripción actualizarse."""
    segs = pd.read_csv(DATA / "manifest_segments.csv")
    grp = segs[segs["num_vineta"] == num_vineta].sort_values("start_frame")

    secuencia = []
    ultimo = None
    for _, row in grp.iterrows():
        kp_path = ROOT / row["kp_path"]
        if not kp_path.exists():
            continue
        kp_seq = np.load(kp_path)
        feat = normalize_sample(kp_seq_to_features(kp_seq))[np.newaxis].astype(np.float32)
        gloss, conf = predecir_ventana(sess, i2l, feat)
        if conf < CONF_UMBRAL or gloss not in vocab_permitido:
            continue
        if gloss != ultimo:
            secuencia.append(gloss)
            ultimo = gloss
    return secuencia


def secuencia_predicha_produccion(num_vineta, sess_v4, i2l_v4, sess_s29, i2l_s29,
                                   vocab_comun, clase_a_idx):
    """Mismo criterio que usa la demo hoy: SegmentadorPausas (no ventana
    fija) + ensemble v4+S29, sobre los frames YA extraídos (misma fuente de
    keypoints que S31, comparación justa — la diferencia es solo el
    algoritmo de segmentación + el modelo, no la calidad de MediaPipe)."""
    segs = pd.read_csv(DATA / "manifest_segments.csv")
    grp = segs[segs["num_vineta"] == num_vineta].sort_values("start_frame")

    # Reconstruir un stream de frames individuales a partir de las ventanas
    # (start_frame de cada ventana, con stride=15 < tamaño=30 → tomamos solo
    # los primeros STRIDE frames de cada ventana para no repetir frames)
    frames_kp = []
    for _, row in grp.iterrows():
        kp_path = ROOT / row["kp_path"]
        if not kp_path.exists():
            continue
        kp_seq = np.load(kp_path)
        frames_kp.extend([kp_seq[i] for i in range(min(STRIDE, len(kp_seq)))])

    segmentador = SegmentadorPausas()
    secuencia = []
    for kp in frames_kp:
        segmento = segmentador.push(kp)
        if segmento is None:
            continue
        kp_seq = resample_to_n_frames(segmento, N_FRAMES)
        feat = normalize_sample(kp_seq_to_features(kp_seq))[np.newaxis].astype(np.float32)

        probs_v4_full = _softmax(sess_v4.run(None, {"sequence": feat})[0][0])
        probs_s29_full = _softmax(sess_s29.run(None, {"sequence": feat})[0][0])
        probs_ens = np.zeros(len(vocab_comun))
        for idx_v4, nombre in i2l_v4.items():
            if nombre in clase_a_idx:
                probs_ens[clase_a_idx[nombre]] += probs_v4_full[idx_v4]
        for idx_s29, nombre in i2l_s29.items():
            if nombre in clase_a_idx:
                probs_ens[clase_a_idx[nombre]] += probs_s29_full[idx_s29]
        probs_ens /= 2
        idx = int(probs_ens.argmax())
        gloss, conf = vocab_comun[idx], float(probs_ens[idx])

        if conf < CONF_UMBRAL or gloss.startswith("HISTORIAS_VINETAS_"):
            continue
        secuencia.append(gloss)
    return secuencia


def _softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


def secuencia_real(num_vineta, vocab_permitido):
    srt_path = SRT_DIR / f"Historias vinetas ({num_vineta}).srt"
    entradas = parse_srt(srt_path)
    return [g for _, _, g in entradas if g in vocab_permitido]


def wer(ref, hyp):
    if not ref:
        return None
    return levenshtein(ref, hyp) / len(ref)


def main():
    with open(DATA / "s31_videos_wer_holdout.json") as f:
        info = json.load(f)
    videos_wer = info["videos_wer"]
    print(f"Videos de evaluación WER (nunca vistos en entrenamiento): {videos_wer}")
    print("=" * 70)

    sess_s31, i2l_s31 = cargar_modelo(ROOT / "checkpoints/bilstm_s31.onnx", DATA / "s31_label2idx.json")
    vocab_s31 = set(i2l_s31.values())

    sess_v4, i2l_v4 = cargar_modelo(ROOT / "checkpoints/bilstm_s27.onnx", DATA / "s27_label2idx.json")
    sess_s29, i2l_s29 = cargar_modelo(ROOT / "checkpoints/bilstm_s29.onnx", DATA / "s29_label2idx.json")
    vocab_comun = sorted(set(i2l_v4.values()) | set(i2l_s29.values()))
    clase_a_idx = {c: i for i, c in enumerate(vocab_comun)}
    vocab_produccion = set(vocab_comun)

    resultados_s31, resultados_prod = [], []

    for num in videos_wer:
        print(f"\n--- Video {num} ---")
        ref_s31 = secuencia_real(num, vocab_s31)
        hyp_s31 = secuencia_predicha_s31(num, sess_s31, i2l_s31, vocab_s31)
        w_s31 = wer(ref_s31, hyp_s31)
        print(f"  [S31]        ref(vocab S31, n={len(ref_s31)}): {ref_s31[:15]}{'...' if len(ref_s31) > 15 else ''}")
        print(f"  [S31]        hyp: {hyp_s31[:15]}{'...' if len(hyp_s31) > 15 else ''}")
        print(f"  [S31]        WER = {w_s31}")
        if w_s31 is not None:
            resultados_s31.append(w_s31)

        ref_prod = secuencia_real(num, vocab_produccion)
        hyp_prod = secuencia_predicha_produccion(num, sess_v4, i2l_v4, sess_s29, i2l_s29,
                                                  vocab_comun, clase_a_idx)
        w_prod = wer(ref_prod, hyp_prod)
        print(f"  [producción] ref(vocab v4+S29, n={len(ref_prod)}): {ref_prod[:15]}{'...' if len(ref_prod) > 15 else ''}")
        print(f"  [producción] hyp: {hyp_prod[:15]}{'...' if len(hyp_prod) > 15 else ''}")
        print(f"  [producción] WER = {w_prod}")
        if w_prod is not None:
            resultados_prod.append(w_prod)

    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    if resultados_s31:
        print(f"S31        — WER medio: {np.mean(resultados_s31):.3f}  (n={len(resultados_s31)} videos)")
    if resultados_prod:
        print(f"Producción — WER medio: {np.mean(resultados_prod):.3f}  (n={len(resultados_prod)} videos)")

    with open(DATA / "s31_wer_resultados.json", "w", encoding="utf-8") as f:
        json.dump({
            "wer_s31_por_video": resultados_s31,
            "wer_produccion_por_video": resultados_prod,
            "wer_s31_medio": float(np.mean(resultados_s31)) if resultados_s31 else None,
            "wer_produccion_medio": float(np.mean(resultados_prod)) if resultados_prod else None,
            "videos_evaluados": videos_wer,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✅ data/s31_wer_resultados.json")


if __name__ == "__main__":
    main()
