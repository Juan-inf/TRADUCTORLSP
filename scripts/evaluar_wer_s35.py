"""evaluar_wer_s35.py — WER real sobre las 10 oraciones ELAN (data/Glosas/
*_ORACION_*.eaf) reservadas por build_dataset_s34_eaf.py, nunca vistas en
entrenamiento por s34_eaf/s35 ni por ningún otro dataset previo (son
archivos físicamente distintos a los usados en S18/S31/S32/S33).

A diferencia de evaluar_wer_s31/s32/s33.py (que reconstruían la secuencia
sobre los 5 videos "Historias Viñetas" usando manifest_segments.csv ya
precomputado), aquí no hay landmarks precomputados: se extraen en vivo con
MediaPipe Holistic desde el .mp4 y se segmentan con SegmentadorPausas
(mismo componente que usa la demo en producción), exactamente el pipeline
real que vería un usuario.

Referencia (ground truth): la secuencia de glosas de la tier GLOSA del
.eaf, anotada a mano con precisión de milisegundo — mejor calidad que el
SRT usado en S31-S33 (ver conversación: anotación lingüística, no
derivada del audio hablado).

Compara 3 pipelines sobre las mismas 10 oraciones:
  - S33     (bilstm_s33.onnx)      — mejor línea anterior (S18+s31_continuo)
  - S35     (bilstm_s35.onnx)      — + s34_eaf (esta línea nueva)
  - producción (ensemble v4+S29)   — pipeline real de la demo hoy

Uso:
    python3 scripts/evaluar_wer_s35.py
"""
import json
import pathlib
import sys
import unicodedata

import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT))

from src.features.segmentacion import SegmentadorPausas
from src.features.landmarks import kp_seq_to_features, normalize_sample, resample_to_n_frames

CONF_UMBRAL = 0.20
mp_holistic = mp.solutions.holistic


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


def wer(ref, hyp):
    if not ref:
        return None
    return levenshtein(ref, hyp) / len(ref)


def cargar_modelo(onnx_path, label2idx_path):
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    l2i = json.load(open(label2idx_path, encoding="utf-8"))
    i2l = {v: nfc(k).upper() for k, v in l2i.items()}
    return sess, i2l


def _softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


def extraer_frames_kp(mp4_path):
    """Devuelve lista de kp [75,3] (left,right,pose) para todos los frames del video."""
    cap = cv2.VideoCapture(str(mp4_path))
    frames_kp = []
    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1,
                               min_detection_confidence=0.3,
                               min_tracking_confidence=0.3) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)
            kp = np.zeros((75, 3), dtype=np.float32)
            if res.left_hand_landmarks:
                for i, lm in enumerate(res.left_hand_landmarks.landmark):
                    kp[i] = [lm.x, lm.y, lm.z]
            if res.right_hand_landmarks:
                for i, lm in enumerate(res.right_hand_landmarks.landmark):
                    kp[21 + i] = [lm.x, lm.y, lm.z]
            if res.pose_landmarks:
                for i, lm in enumerate(res.pose_landmarks.landmark):
                    kp[42 + i] = [lm.x, lm.y, lm.z]
            frames_kp.append(kp)
    cap.release()
    return frames_kp


def secuencia_predicha_un_modelo(frames_kp, sess, i2l, vocab_permitido):
    segmentador = SegmentadorPausas()
    secuencia = []
    segmentos = []
    for kp in frames_kp:
        seg = segmentador.push(kp)
        if seg is not None:
            segmentos.append(seg)
    ultimo = segmentador.flush()
    if ultimo is not None:
        segmentos.append(ultimo)

    for seg in segmentos:
        kp_seq = resample_to_n_frames(seg, 30)
        feat = normalize_sample(kp_seq_to_features(kp_seq))[np.newaxis].astype(np.float32)
        logits = sess.run(None, {"sequence": feat})[0][0]
        probs = _softmax(logits)
        idx = int(probs.argmax())
        gloss, conf = i2l.get(idx, "?"), float(probs[idx])
        if conf < CONF_UMBRAL or gloss not in vocab_permitido:
            continue
        secuencia.append(gloss)
    return secuencia


def secuencia_predicha_produccion(frames_kp, sess_v4, i2l_v4, sess_s29, i2l_s29,
                                   vocab_comun, clase_a_idx):
    segmentador = SegmentadorPausas()
    segmentos = []
    for kp in frames_kp:
        seg = segmentador.push(kp)
        if seg is not None:
            segmentos.append(seg)
    ultimo = segmentador.flush()
    if ultimo is not None:
        segmentos.append(ultimo)

    secuencia = []
    for seg in segmentos:
        kp_seq = resample_to_n_frames(seg, 30)
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


def secuencia_real(glosas, vocab_permitido):
    return [nfc(g.upper()) for _, _, g in glosas if nfc(g.upper()) in vocab_permitido]


def main():
    with open(DATA / "s34_oraciones_wer_holdout.json", encoding="utf-8") as f:
        holdout = json.load(f)["holdout"]
    print(f"Oraciones ELAN de evaluación WER (nunca vistas en entrenamiento — ni s34_eaf ni S35): {len(holdout)}")
    print("=" * 70)

    sess_s33, i2l_s33 = cargar_modelo(ROOT / "checkpoints/bilstm_s33.onnx", DATA / "s32_label2idx.json")
    vocab_s33 = set(i2l_s33.values())

    sess_s35, i2l_s35 = cargar_modelo(ROOT / "checkpoints/bilstm_s35.onnx", DATA / "s35_label2idx.json")
    vocab_s35 = set(i2l_s35.values())

    sess_v4, i2l_v4 = cargar_modelo(ROOT / "checkpoints/bilstm_s27.onnx", DATA / "s27_label2idx.json")
    sess_s29, i2l_s29 = cargar_modelo(ROOT / "checkpoints/bilstm_s29.onnx", DATA / "s29_label2idx.json")
    vocab_comun = sorted(set(i2l_v4.values()) | set(i2l_s29.values()))
    clase_a_idx = {c: i for i, c in enumerate(vocab_comun)}
    vocab_produccion = set(vocab_comun)

    resultados = {"s33": [], "s35": [], "produccion": []}

    for entry in holdout:
        mp4 = ROOT / entry["mp4"]
        glosas = entry["glosas"]
        print(f"\n--- {mp4.name} ---")
        frames_kp = extraer_frames_kp(mp4)
        print(f"  frames extraídos: {len(frames_kp)}")

        ref_s33 = secuencia_real(glosas, vocab_s33)
        hyp_s33 = secuencia_predicha_un_modelo(frames_kp, sess_s33, i2l_s33, vocab_s33)
        w_s33 = wer(ref_s33, hyp_s33)
        print(f"  [S33] ref(n={len(ref_s33)}): {ref_s33}")
        print(f"  [S33] hyp: {hyp_s33}   WER={w_s33}")
        if w_s33 is not None:
            resultados["s33"].append(w_s33)

        ref_s35 = secuencia_real(glosas, vocab_s35)
        hyp_s35 = secuencia_predicha_un_modelo(frames_kp, sess_s35, i2l_s35, vocab_s35)
        w_s35 = wer(ref_s35, hyp_s35)
        print(f"  [S35] ref(n={len(ref_s35)}): {ref_s35}")
        print(f"  [S35] hyp: {hyp_s35}   WER={w_s35}")
        if w_s35 is not None:
            resultados["s35"].append(w_s35)

        ref_prod = secuencia_real(glosas, vocab_produccion)
        hyp_prod = secuencia_predicha_produccion(frames_kp, sess_v4, i2l_v4, sess_s29, i2l_s29,
                                                  vocab_comun, clase_a_idx)
        w_prod = wer(ref_prod, hyp_prod)
        print(f"  [prod] ref(n={len(ref_prod)}): {ref_prod}")
        print(f"  [prod] hyp: {hyp_prod}   WER={w_prod}")
        if w_prod is not None:
            resultados["produccion"].append(w_prod)

    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    for nombre, valores in resultados.items():
        if valores:
            print(f"{nombre:<12} — WER medio: {np.mean(valores):.3f}  (n={len(valores)} oraciones)")

    with open(DATA / "s35_wer_resultados.json", "w", encoding="utf-8") as f:
        json.dump({
            "wer_por_oracion": resultados,
            "wer_medio": {k: (float(np.mean(v)) if v else None) for k, v in resultados.items()},
            "n_oraciones": len(holdout),
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✅ data/s35_wer_resultados.json")


if __name__ == "__main__":
    main()
