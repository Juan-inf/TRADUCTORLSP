"""evaluar_wer_s37_ensemble_bench.py — S33 y S36 quedaron en un resultado
MIXTO (ver conversación): S36 mejora WER en el benchmark oficial de 5
videos (0.981 vs 1.027) pero empeora en las 10 oraciones ELAN nuevas
(1.125 vs 1.062) — ninguno gana limpio en los dos. En vez de otro
entrenamiento largo, se prueba la misma técnica que ya usa producción
(ensemble v4+S29): promediar softmax de S33+S36 sobre su vocabulario
común, para ver si el ensemble hereda lo mejor de ambos sin otro ciclo de
entrenamiento de horas.

Corre en los mismos 5 videos del benchmark oficial (S31→S36):
  - S33            (bilstm_s33.onnx)
  - S36            (bilstm_s36.onnx)
  - ensemble S33+S36
  - v4+S29 (ensemble, producción) — pipeline real de la demo hoy

Uso:
    python3 scripts/evaluar_wer_s37_ensemble_bench.py
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


def secuencia_predicha_ensemble_ventana_fija(num_vineta, sess_a, i2l_a, sess_b, i2l_b,
                                              vocab_comun, clase_a_idx):
    segs = pd.read_csv(DATA / "manifest_segments.csv")
    grp = segs[segs["num_vineta"] == num_vineta].sort_values("start_frame")
    secuencia, ultimo = [], None
    for _, row in grp.iterrows():
        kp_path = ROOT / row["kp_path"]
        if not kp_path.exists():
            continue
        kp_seq = np.load(kp_path)
        feat = normalize_sample(kp_seq_to_features(kp_seq))[np.newaxis].astype(np.float32)

        probs_a_full = _softmax(sess_a.run(None, {"sequence": feat})[0][0])
        probs_b_full = _softmax(sess_b.run(None, {"sequence": feat})[0][0])
        probs_ens = np.zeros(len(vocab_comun))
        for idx_a, nombre in i2l_a.items():
            if nombre in clase_a_idx:
                probs_ens[clase_a_idx[nombre]] += probs_a_full[idx_a]
        for idx_b, nombre in i2l_b.items():
            if nombre in clase_a_idx:
                probs_ens[clase_a_idx[nombre]] += probs_b_full[idx_b]
        probs_ens /= 2
        idx = int(probs_ens.argmax())
        gloss, conf = vocab_comun[idx], float(probs_ens[idx])
        if conf < CONF_UMBRAL or gloss not in vocab_comun:
            continue
        if gloss != ultimo:
            secuencia.append(gloss)
            ultimo = gloss
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

    sess_s36, i2l_s36 = cargar_modelo(ROOT / "checkpoints/bilstm_s36.onnx", DATA / "s35_label2idx.json")
    vocab_s36 = set(i2l_s36.values())

    sess_v4, i2l_v4 = cargar_modelo(ROOT / "checkpoints/bilstm_s27.onnx", DATA / "s27_label2idx.json")
    sess_s29, i2l_s29 = cargar_modelo(ROOT / "checkpoints/bilstm_s29.onnx", DATA / "s29_label2idx.json")
    vocab_comun = sorted(set(i2l_v4.values()) | set(i2l_s29.values()))
    clase_a_idx = {c: i for i, c in enumerate(vocab_comun)}
    vocab_produccion = set(vocab_comun)

    vocab_comun_36 = sorted(vocab_s33 | vocab_s36)
    clase_a_idx_36 = {c: i for i, c in enumerate(vocab_comun_36)}

    resultados = {"s31": [], "s32": [], "s33": [], "s36": [], "ensemble_s33_s36": [], "produccion": []}

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

        ref_s36 = secuencia_real(num, vocab_s36)
        hyp_s36 = secuencia_predicha_ventana_fija(num, sess_s36, i2l_s36, vocab_s36)
        w_s36 = wer(ref_s36, hyp_s36)
        print(f"  [S36] ref(vocab 274cls, n={len(ref_s36)}): {ref_s36[:12]}...")
        print(f"  [S36] hyp: {hyp_s36[:12]}...   WER={w_s36}")
        if w_s36 is not None:
            resultados["s36"].append(w_s36)

        ref_ens = secuencia_real(num, set(vocab_comun_36))
        hyp_ens = secuencia_predicha_ensemble_ventana_fija(num, sess_s33, i2l_s33, sess_s36, i2l_s36,
                                                            vocab_comun_36, clase_a_idx_36)
        w_ens = wer(ref_ens, hyp_ens)
        print(f"  [ens33+36] ref(n={len(ref_ens)}): {ref_ens[:12]}...")
        print(f"  [ens33+36] hyp: {hyp_ens[:12]}...   WER={w_ens}")
        if w_ens is not None:
            resultados["ensemble_s33_s36"].append(w_ens)

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

    with open(DATA / "s37_ensemble_bench_wer_resultados.json", "w", encoding="utf-8") as f:
        json.dump({
            "wer_por_video": resultados,
            "wer_medio": {k: (float(np.mean(v)) if v else None) for k, v in resultados.items()},
            "videos_evaluados": videos_wer,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✅ data/s37_ensemble_bench_wer_resultados.json")


if __name__ == "__main__":
    main()
