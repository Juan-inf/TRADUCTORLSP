"""validar_ensemble_v4_s29.py

Reconstruye, con rastreo de archivo físico, el split de test REAL que usó
train_s27.py (v4) y el que usó train_s29.py (Fase 1) — ambos deterministas
(SEED=42), pero sobre datasets con distinto orden/contenido, así que sus
índices de test NO son directamente comparables entre sí.

Intersecta ambos holdouts por archivo .pkl físico: solo esos archivos están
garantizados fuera del entrenamiento de AMBOS modelos. Evalúa v4, S29 y su
ensemble (promedio de probabilidades) exclusivamente sobre esa intersección
— la única comparación metodológicamente válida entre los dos.
"""
import json
import pathlib
import pickle
import re
import sys
import unicodedata
import collections

import numpy as np
import onnxruntime as ort
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = pathlib.Path(__file__).parent.parent
DATA = ROOT / "data"
KP_DIR = DATA / "Keypoints"
SEED = 42
N_FRAMES = 30
N_DIMS = 150


def nfc(s):
    return unicodedata.normalize("NFC", s)


sys.path.insert(0, str(ROOT / "scripts"))
import build_dataset_s17 as b17
import build_dataset_s18 as b18


# ─── Paso 1: reconstruir v4 (s17) con rastreo de archivo físico ────────────

def cargar_fuente_con_paths(kp_base, config):
    Xs, Ls, Gs, Ps = [], [], [], []
    if not kp_base.exists():
        return [], [], [], []
    clases = sorted(d for d in kp_base.iterdir() if d.is_dir())
    for clase_idx, clase_dir in enumerate(clases):
        clase_name = clase_dir.name
        pkls = sorted(clase_dir.glob("*.pkl"))
        for m_idx, pkl_path in enumerate(pkls):
            seq = b17.pkl_to_sequence(pkl_path)
            if seq is None:
                continue
            grupo = b17.inferir_grupo(pkl_path, clase_idx, config, m_idx)
            Xs.append(seq); Ls.append(nfc(clase_name.upper())); Gs.append(grupo)
            Ps.append(str(pkl_path))
    return Xs, Ls, Gs, Ps


print("=" * 65)
print("Reconstruyendo dataset v4 (S17) con rastreo de archivo físico...")
print("=" * 65)
all_X, all_L, all_G, all_S, all_P = [], [], [], [], []
for carpeta, config in b17.SOURCE_CONFIG.items():
    X, L, G, P = cargar_fuente_con_paths(KP_DIR / carpeta, config)
    if X:
        all_X += X; all_L += L; all_G += G; all_S += [config["etiqueta"]] * len(X); all_P += P
for carpeta, config in b17.SOURCE_CONFIG_EXTRA.items():
    kp_base = KP_DIR / carpeta
    if kp_base.exists():
        X, L, G, P = cargar_fuente_con_paths(kp_base, config)
        if X:
            all_X += X; all_L += L; all_G += G; all_S += [config["etiqueta"]] * len(X); all_P += P

X_all = np.array(all_X, dtype=np.float32)
L_all = np.array(all_L)
G_all = np.array(all_G, dtype=np.int32)
S_all = np.array(all_S)
P_all = np.array(all_P)

cnts = collections.Counter(L_all.tolist())
mask = np.array([cnts[l] >= 15 for l in L_all])
X_all, L_all, G_all, S_all, P_all = X_all[mask], L_all[mask], G_all[mask], S_all[mask], P_all[mask]
print(f"Total v4 (post-filtro): {len(X_all)} muestras, {len(set(L_all.tolist()))} clases")

G_all = b17.balancear_grupos_hv(L_all, G_all, S_all, seed=SEED)

classes_v4 = sorted(set(L_all.tolist()))
l2i_v4_repro = {c: i for i, c in enumerate(classes_v4)}
y_all = np.array([l2i_v4_repro[l] for l in L_all], dtype=np.int64)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
tv_idx, te_idx_v4 = next(sss.split(X_all, y_all))
v4_test_paths = set(P_all[te_idx_v4].tolist())
v4_test_path_to_label = {P_all[i]: L_all[i] for i in te_idx_v4}
print(f"v4 test-holdout reproducido: {len(v4_test_paths)} archivos únicos")


# ─── Paso 2: reconstruir S29 (s18b, --solo-vocab-actual) con rastreo ───────

def cargar_fuente_glosas_con_paths(kp_base, config):
    Xs, Ls, Gs, Ps = [], [], [], []
    offset = config.get("grupo_offset", 0)
    if not kp_base.exists():
        return [], [], [], []
    carpetas = sorted(d for d in kp_base.iterdir() if d.is_dir())
    for carpeta_idx, carpeta_dir in enumerate(carpetas):
        grupo = offset + carpeta_idx
        for pkl_path in sorted(carpeta_dir.glob("*.pkl")):
            gloss = b18.limpiar_gloss(pkl_path.stem)
            if len(gloss) <= 1:
                continue
            seq = b18.pkl_to_sequence(pkl_path)
            if seq is None:
                continue
            Xs.append(seq); Ls.append(nfc(gloss.upper())); Gs.append(grupo)
            Ps.append(str(pkl_path))
    return Xs, Ls, Gs, Ps


print("\n" + "=" * 65)
print("Reconstruyendo dataset S29 (S18b, Fase 1) con rastreo de archivo...")
print("=" * 65)
all_X2, all_L2, all_G2, all_S2, all_P2 = [], [], [], [], []
for carpeta, config in b18.SOURCE_CONFIG.items():
    X, L, G, P = cargar_fuente_con_paths(KP_DIR / carpeta, config)
    if X:
        all_X2 += X; all_L2 += L; all_G2 += G; all_S2 += [config["etiqueta"]] * len(X); all_P2 += P
for carpeta, config in b18.SOURCE_CONFIG_EXTRA.items():
    kp_base = KP_DIR / carpeta
    if kp_base.exists():
        X, L, G, P = cargar_fuente_con_paths(kp_base, config)
        if X:
            all_X2 += X; all_L2 += L; all_G2 += G; all_S2 += [config["etiqueta"]] * len(X); all_P2 += P
for carpeta, config in b18.SOURCE_CONFIG_GLOSAS.items():
    X, L, G, P = cargar_fuente_glosas_con_paths(KP_DIR / carpeta, config)
    if X:
        all_X2 += X; all_L2 += L; all_G2 += G; all_S2 += [config["etiqueta"]] * len(X); all_P2 += P

X_all2 = np.array(all_X2, dtype=np.float32)
L_all2 = np.array(all_L2)
G_all2 = np.array(all_G2, dtype=np.int32)
S_all2 = np.array(all_S2)
P_all2 = np.array(all_P2)

with open(DATA / "s27_label2idx.json", encoding="utf-8") as f:
    vocab_actual = {nfc(k).upper() for k in json.load(f).keys()}
mask_vocab = np.array([l in vocab_actual for l in L_all2])
X_all2, L_all2, G_all2, S_all2, P_all2 = (
    X_all2[mask_vocab], L_all2[mask_vocab], G_all2[mask_vocab], S_all2[mask_vocab], P_all2[mask_vocab]
)

cnts2 = collections.Counter(L_all2.tolist())
mask2 = np.array([cnts2[l] >= 15 for l in L_all2])
X_all2, L_all2, G_all2, S_all2, P_all2 = (
    X_all2[mask2], L_all2[mask2], G_all2[mask2], S_all2[mask2], P_all2[mask2]
)
print(f"Total S29 (post-filtro): {len(X_all2)} muestras, {len(set(L_all2.tolist()))} clases")

G_all2 = b18.balancear_grupos_cruzados(L_all2, G_all2, S_all2, pares=[
    ("dgi156", "vineta"),
    ("dgi156_gloss", "vineta_gloss"),
], seed=SEED)

classes_s29 = sorted(set(L_all2.tolist()))
l2i_s29_repro = {c: i for i, c in enumerate(classes_s29)}
y_all2 = np.array([l2i_s29_repro[l] for l in L_all2], dtype=np.int64)

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
tv_idx2, te_idx2 = next(sss2.split(X_all2, y_all2))
s29_test_paths = set(P_all2[te_idx2].tolist())
s29_test_path_to_label = {P_all2[i]: L_all2[i] for i in te_idx2}
print(f"S29 test-holdout reproducido: {len(s29_test_paths)} archivos únicos")


# ─── Paso 3: intersección — archivos fuera del entrenamiento de AMBOS ─────

interseccion = v4_test_paths & s29_test_paths
print(f"\n{'='*65}")
print(f"Intersección (fuera de entrenamiento de v4 Y de S29): {len(interseccion)} archivos")
print(f"{'='*65}")

if len(interseccion) < 30:
    print("MUY POCOS archivos en la intersección — resultado no sería confiable.")
    sys.exit(1)

# reconstruir X para la intersección desde P_all2 (ya tiene todas las vistas)
idx_by_path2 = {p: i for i, p in enumerate(P_all2)}
inter_paths = sorted(interseccion)
X_eval = np.stack([X_all2[idx_by_path2[p]] for p in inter_paths if p in idx_by_path2])
labels_eval = [s29_test_path_to_label[p] for p in inter_paths if p in idx_by_path2]
print(f"Muestras evaluables (con etiqueta real de palabra): {len(X_eval)}")
print(f"Clases distintas en la intersección: {len(set(labels_eval))}")


# ─── Paso 4: evaluar v4, S29 y ensemble sobre la intersección limpia ──────

def normalize_sample(x):
    mu, std = x.mean(), x.std()
    return ((x - mu) / std).astype(np.float32) if std >= 1e-8 else x


X_norm = np.stack([normalize_sample(X_eval[i]) for i in range(len(X_eval))])


def cargar_modelo(onnx_path, label2idx_path):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    l2i = json.load(open(label2idx_path, encoding="utf-8"))
    i2l = {v: nfc(k).upper() for k, v in l2i.items()}
    return sess, inp, i2l


sess_v4, inp_v4, i2l_v4 = cargar_modelo(str(ROOT / "checkpoints/bilstm_s27.onnx"), str(DATA / "s27_label2idx.json"))
sess_s29, inp_s29, i2l_s29 = cargar_modelo(str(ROOT / "checkpoints/bilstm_s29.onnx"), str(DATA / "s29_label2idx.json"))

vocab_comun = sorted(set(i2l_v4.values()) & set(i2l_s29.values()))
clase_a_idx = {c: i for i, c in enumerate(vocab_comun)}


def probs_alineadas(sess, inp_name, i2l, X):
    logits = sess.run(None, {inp_name: X.astype(np.float32)})[0]
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    out = np.zeros((len(X), len(vocab_comun)), dtype=np.float32)
    for local_idx, nombre in i2l.items():
        if nombre in clase_a_idx:
            out[:, clase_a_idx[nombre]] = probs[:, local_idx]
    return out


probs_v4 = probs_alineadas(sess_v4, inp_v4, i2l_v4, X_norm)
probs_s29 = probs_alineadas(sess_s29, inp_s29, i2l_s29, X_norm)
probs_ens = (probs_v4 + probs_s29) / 2

y_true = np.array([clase_a_idx.get(l, -1) for l in labels_eval])
mask_valida = y_true >= 0
y_true = y_true[mask_valida]
print(f"\nMuestras con clase válida en vocabulario común: {mask_valida.sum()} / {len(labels_eval)}")

pred_v4 = probs_v4[mask_valida].argmax(axis=1)
pred_s29 = probs_s29[mask_valida].argmax(axis=1)
pred_ens = probs_ens[mask_valida].argmax(axis=1)

f1_v4 = f1_score(y_true, pred_v4, average="macro")
f1_s29 = f1_score(y_true, pred_s29, average="macro")
f1_ens = f1_score(y_true, pred_ens, average="macro")

print(f"\n{'='*65}")
print(f"RESULTADO — holdout limpio (fuera del entrenamiento de AMBOS modelos)")
print(f"{'='*65}")
print(f"  N = {mask_valida.sum()} muestras, {len(set(y_true.tolist()))} clases")
print(f"  v4  solo  : F1-macro = {f1_v4:.4f}")
print(f"  S29 solo  : F1-macro = {f1_s29:.4f}")
print(f"  ENSEMBLE  : F1-macro = {f1_ens:.4f}")
