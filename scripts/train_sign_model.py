"""
train_sign_model.py — Entrena RandomForest sobre PKL de señas LSP y guarda el modelo.
Salida: checkpoints/rf_signs.pkl  +  data/sign_label2idx.json
"""
import pickle, json, re, pathlib, hashlib, warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

ROOT     = pathlib.Path(__file__).parent.parent
PKL_ROOT = ROOT / "data" / "Keypoints" / "pkl"
CKPT_DIR = ROOT / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

# ── Extracción de features (idéntica a semana8_hpo.py) ────────────────────────

def extract_features(pkl_path):
    try:
        with open(pkl_path, "rb") as f:
            frames = pickle.load(f)
        if not frames:
            return None
        vecs = []
        for fr in frames:
            parts = []
            for key in ("pose", "right_hand"):
                d = fr.get(key, {})
                x = d.get("x", [])
                y = d.get("y", [])
                if not x:
                    n = 33 if key == "pose" else 21
                    x, y = [0.0] * n, [0.0] * n
                parts.extend(x)
                parts.extend(y)
            vecs.append(parts)
        return np.mean(vecs, axis=0).astype(np.float32)
    except Exception:
        return None

def label_from_filename(name):
    stem = pathlib.Path(name).stem
    m = re.match(r"^(.+)_(\d+)$", stem)
    return m.group(1) if m else stem

# ── Cargar dataset ────────────────────────────────────────────────────────────

print("Cargando dataset PKL...")
X_list, y_list = [], []

for vineta_dir in sorted(PKL_ROOT.iterdir()):
    if not vineta_dir.is_dir():
        continue
    for pkl_path in sorted(vineta_dir.glob("*.pkl")):
        feat = extract_features(pkl_path)
        if feat is None:
            continue
        X_list.append(feat)
        y_list.append(label_from_filename(pkl_path.name))

X      = np.array(X_list)
labels = sorted(set(y_list))
label2idx = {c: i for i, c in enumerate(labels)}
y      = np.array([label2idx[l] for l in y_list])

print(f"  Muestras: {len(X)} | Clases: {len(labels)} | Features: {X.shape[1]}")
print(f"  Ejemplo clases: {labels[:8]}")

# Guardar mapeo de etiquetas
with open(ROOT / "data" / "sign_label2idx.json", "w", encoding="utf-8") as f:
    json.dump(label2idx, f, ensure_ascii=False, indent=2)
print("  sign_label2idx.json guardado")

# ── Entrenar sobre todos los datos ────────────────────────────────────────────

print("\nEntrenando RandomForest (n_estimators=60, max_depth=15, n_jobs=4)...")
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    RandomForestClassifier(
        n_estimators=60,
        max_depth=15,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=4,
    ))
])

pipe.fit(X, y)
y_pred   = pipe.predict(X)
f1_train = f1_score(y, y_pred, average="macro", zero_division=0)
print(f"  F1-macro train (resubstitución): {f1_train:.4f}")

# ── Guardar modelo ────────────────────────────────────────────────────────────

ckpt_path = CKPT_DIR / "rf_signs.pkl"
model_data = {
    "pipeline":   pipe,
    "label2idx":  label2idx,
    "idx2label":  {v: k for k, v in label2idx.items()},
    "n_classes":  len(labels),
    "n_features": X.shape[1],
    "f1_train":   f1_train,
}
with open(ckpt_path, "wb") as f:
    pickle.dump(model_data, f)

size_mb = ckpt_path.stat().st_size / 1e6
print(f"\nModelo guardado: {ckpt_path} ({size_mb:.1f} MB)")
print(f"  Clases (muestra): {labels[:5]}...")
print(f"  Total clases LSP: {len(labels)}")
print("\n[LISTO] Usa checkpoints/rf_signs.pkl para inferencia en tiempo real.")
