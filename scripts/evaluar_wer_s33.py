"""evaluar_wer_s32.py — Igual que evaluar_wer_s31.py, pero evaluando
checkpoints/bilstm_s32.onnx (S18 + s31_continuo combinados, ver
scripts/build_dataset_s32_merge.py y scripts/train_s32.py) sobre los MISMOS
5 videos reservados (s31_videos_wer_holdout.json) — nunca vistos por S32
tampoco, ya que build_dataset_s32_merge.py excluye esos num_vineta de la
fuente s31_continuo.

Corre 3 pipelines sobre los mismos 5 videos:
  - S31   (bilstm_s31.onnx)      — entrenado aislado, 1,538 muestras (perdió)
  - S32   (bilstm_s32.onnx)      — S18 + s31_continuo combinados
  - v4+S29 (ensemble, producción) — pipeline real de la demo hoy

Uso:
    python3 scripts/evaluar_wer_s32.py
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


def levenshtein(ref, hyp):
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


def secuencia_predicha_ventana_fija(num_vineta, sess, i2l, vocab_permitido):
    segs = pd.read_csv(DATA / "manifest_segments.csv")
    grp = segs[segs["num_vineta"] == num_vineta].sort_values("start_frame")
    secuencia, ultimo = [], None
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


def _softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


def secuencia_predicha_produccion(num_vineta, sess_v4, i2l_v4, sess_s29, i2l_s29,
                                   vocab_comun, clase_a_idx):
    segs = pd.read_csv(DATA / "manifest_segments.csv")
    grp = segs[segs["num_vineta"] == num_vineta].sort_values("start_frame")
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
    print(f"Videos de evaluación WER (nunca vistos en entrenamiento — ni S31 ni S32 ni S33): {videos_wer}")
    print("=" * 70)

    sess_s31, i2l_s31 = cargar_modelo(ROOT / "checkpoints/bilstm_s31.onnx", DATA / "s31_label2idx.json")
    vocab_s31 = set(i2l_s31.values())

    sess_s32, i2l_s32 = cargar_modelo(ROOT / "checkpoints/bilstm_s32.onnx", DATA / "s32_label2idx.json")
    vocab_s32 = set(i2l_s32.values())

    sess_s33, i2l_s33 = cargar_modelo(ROOT / "checkpoints/bilstm_s33.onnx", DATA / "s32_label2idx.json")
    vocab_s33 = set(i2l_s33.values())

    sess_v4, i2l_v4 = cargar_modelo(ROOT / "checkpoints/bilstm_s27.onnx", DATA / "s27_label2idx.json")
    sess_s29, i2l_s29 = cargar_modelo(ROOT / "checkpoints/bilstm_s29.onnx", DATA / "s29_label2idx.json")
    vocab_comun = sorted(set(i2l_v4.values()) | set(i2l_s29.values()))
    clase_a_idx = {c: i for i, c in enumerate(vocab_comun)}
    vocab_produccion = set(vocab_comun)

    resultados = {"s31": [], "s32": [], "s33": [], "produccion": []}

    for num in videos_wer:
        print(f"\n--- Video {num} ---")

        ref_s31 = secuencia_real(num, vocab_s31)
        hyp_s31 = secuencia_predicha_ventana_fija(num, sess_s31, i2l_s31, vocab_s31)
        w_s31 = wer(ref_s31, hyp_s31)
        print(f"  [S31] ref(n={len(ref_s31)}): {ref_s31[:12]}...")
        print(f"  [S31] hyp: {hyp_s31[:12]}...   WER={w_s31}")
        if w_s31 is not None:
            resultados["s31"].append(w_s31)

        ref_s32 = secuencia_real(num, vocab_s32)
        hyp_s32 = secuencia_predicha_ventana_fija(num, sess_s32, i2l_s32, vocab_s32)
        w_s32 = wer(ref_s32, hyp_s32)
        print(f"  [S32] ref(vocab 270cls, n={len(ref_s32)}): {ref_s32[:12]}...")
        print(f"  [S32] hyp: {hyp_s32[:12]}...   WER={w_s32}")
        if w_s32 is not None:
            resultados["s32"].append(w_s32)

        ref_s33 = secuencia_real(num, vocab_s33)
        hyp_s33 = secuencia_predicha_ventana_fija(num, sess_s33, i2l_s33, vocab_s33)
        w_s33 = wer(ref_s33, hyp_s33)
        print(f"  [S33] ref(vocab 270cls, n={len(ref_s33)}): {ref_s33[:12]}...")
        print(f"  [S33] hyp: {hyp_s33[:12]}...   WER={w_s33}")
        if w_s33 is not None:
            resultados["s33"].append(w_s33)

        ref_prod = secuencia_real(num, vocab_produccion)
        hyp_prod = secuencia_predicha_produccion(num, sess_v4, i2l_v4, sess_s29, i2l_s29,
                                                  vocab_comun, clase_a_idx)
        w_prod = wer(ref_prod, hyp_prod)
        print(f"  [prod] ref(n={len(ref_prod)}): {ref_prod[:12]}...")
        print(f"  [prod] hyp: {hyp_prod[:12]}...   WER={w_prod}")
        if w_prod is not None:
            resultados["produccion"].append(w_prod)

    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    for nombre, valores in resultados.items():
        if valores:
            print(f"{nombre:<12} — WER medio: {np.mean(valores):.3f}  (n={len(valores)} videos)")

    with open(DATA / "s33_wer_resultados.json", "w", encoding="utf-8") as f:
        json.dump({
            "wer_por_video": resultados,
            "wer_medio": {k: (float(np.mean(v)) if v else None) for k, v in resultados.items()},
            "videos_evaluados": videos_wer,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✅ data/s33_wer_resultados.json")


if __name__ == "__main__":
    main()
