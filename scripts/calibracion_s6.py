"""
Calibración Sprint Semana 6 — Ablaciones de features, curvas de aprendizaje,
importancia de variables y calibración de probabilidades.

Keypoints (30, 75, 3):
  [0:21]  mano izquierda  — wrist=0
  [21:42] mano derecha    — wrist=21 (referencia de normalización)
  [42:75] pose MediaPipe  — shoulder_L=53, shoulder_R=54

Uso:
    source .venv310/bin/activate
    python scripts/calibracion_s6.py

Salidas en data/:
    calibracion_ablacion.csv
    calibracion_curvas.png
    calibracion_importancia.png
    calibracion_calibracion.png
    calibracion_resumen.txt
"""
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupKFold, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

np.random.seed(42)
print("=" * 65)
print("CALIBRACIÓN SPRINT SEMANA 6 — TRADUCTOR LSP")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────
# 1. CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────
print("\n[1/6] Cargando datos...")
t0 = time.time()

df = pd.read_csv(DATA_DIR / "manifest_segments.csv")
df = df[df["kp_path"].notna() & (df["kp_path"] != "")].reset_index(drop=True)
print(f"  Segmentos: {len(df)} | Clases: {df['clase'].nunique()} | "
      f"Train={df['split'].eq('train').sum()} Val={df['split'].eq('val').sum()} "
      f"Test={df['split'].eq('test').sum()}")

seqs, labels, video_ids = [], [], []
errores = 0
for _, row in df.iterrows():
    try:
        kp = np.load(ROOT / row["kp_path"])   # (30, 75, 3)
        seqs.append(kp.astype(np.float32))
        labels.append(row["clase"])
        video_ids.append(int(row["num_vineta"]))
    except Exception:
        errores += 1

seqs      = np.array(seqs)       # (N, 30, 75, 3)
video_ids = np.array(video_ids)

le    = LabelEncoder()
y_enc = le.fit_transform(labels)

print(f"  Cargados: {len(seqs)} segmentos (errores={errores}) en {time.time()-t0:.1f}s")
print(f"  Shape secuencia: {seqs.shape}  Clases: {len(le.classes_)}")
print(f"  Viñetas únicas (grupos): {np.unique(video_ids).tolist()}")

N_KP   = seqs.shape[2]   # 75
T      = seqs.shape[1]   # 30

# ─────────────────────────────────────────────────────────────────
# 2. EXTRACCIÓN DE FEATURES (3 variantes)
# ─────────────────────────────────────────────────────────────────
print("\n[2/6] Extrayendo features...")

# Constantes de índices (ver extract_landmarks_only.py)
LH_OFF   = 0    # mano izquierda wrist
RH_OFF   = 21   # mano derecha wrist  (referencia, ≈0 post-normalización)
POSE_OFF = 42   # pose landmarks

# Nombres de keypoints para el plot de importancia
KP_NAMES = (
    [f"lh_{i:02d}"   for i in range(21)] +   # 0-20
    [f"rh_{i:02d}"   for i in range(21)] +   # 21-41
    [f"pose_{i:02d}" for i in range(33)]      # 42-74
)
COORD_NAMES = ["x", "y", "z"]


def _stats(arr: np.ndarray) -> np.ndarray:
    """Agrega temporal: mean + std + min + max. arr:(T,...) → (4*...)."""
    return np.concatenate([
        arr.mean(0).flatten(),
        arr.std(0).flatten(),
        arr.min(0).flatten(),
        arr.max(0).flatten(),
    ])


def extract_baseline(seq: np.ndarray) -> np.ndarray:
    """
    seq: (30, 75, 3)
    Estadísticas temporales de coordenadas crudas → 75*3*4 = 900 features.
    """
    return _stats(seq)   # 900


def extract_var1(seq: np.ndarray) -> np.ndarray:
    """
    Baseline + velocity + acceleration + distancia entre muñecas.
    900 + 450 + 450 + 4 = 1804 features.
    """
    base = extract_baseline(seq)

    # Velocidad: diferencia de 1er orden sobre el tiempo
    vel = np.diff(seq, axis=0, prepend=seq[:1])         # (30, 75, 3)
    vel_stats = np.concatenate([vel.mean(0).flatten(),
                                 vel.std(0).flatten()])  # 450

    # Aceleración: diferencia de 2do orden
    acc = np.diff(vel, axis=0, prepend=vel[:1])
    acc_stats = np.concatenate([acc.mean(0).flatten(),
                                 acc.std(0).flatten()])  # 450

    # Distancia entre muñecas (left_wrist=0, right_wrist=21)
    lw = seq[:, LH_OFF,  :2]   # (T, 2)
    rw = seq[:, RH_OFF,  :2]   # (T, 2) — ≈0 por normalización
    dist = np.linalg.norm(lw - rw, axis=1)              # (T,)
    dist_stats = np.array([dist.mean(), dist.std(),
                            dist.min(),  dist.max()])    # 4

    return np.concatenate([base, vel_stats, acc_stats, dist_stats])   # 1804


def extract_var2(seq: np.ndarray) -> np.ndarray:
    """
    Var1 + orientación de palma + simetría bilateral + apertura de mano.
    1804 + 2 + 6 + 2 = 1814 features.
    """
    v1 = extract_var1(seq)

    # ── Orientación de palma derecha ────────────────────────────
    # Vectores desde muñeca derecha (21) a índice_mcp(26) y meñique_mcp(38)
    wrist = seq[:, RH_OFF + 0,  :2]   # (T, 2)
    idxm  = seq[:, RH_OFF + 5,  :2]   # (T, 2)
    pink  = seq[:, RH_OFF + 17, :2]   # (T, 2)
    v_i   = idxm  - wrist
    v_p   = pink  - wrist
    cross = v_i[:, 0]*v_p[:, 1] - v_i[:, 1]*v_p[:, 0]
    dot   = (v_i * v_p).sum(1)
    orient = np.arctan2(cross, dot + 1e-8)               # (T,) radianes
    orient_stats = np.array([orient.mean(), orient.std()])   # 2

    # ── Simetría bilateral (hombros, codos, muñecas de pose) ───
    # Pose indices: shoulder_L=11→kp[53], shoulder_R=12→kp[54]
    #               elbow_L=13→kp[55],   elbow_R=14→kp[56]
    #               wrist_L=15→kp[57],   wrist_R=16→kp[58]
    sym_pairs = [(POSE_OFF+11, POSE_OFF+12),
                 (POSE_OFF+13, POSE_OFF+14),
                 (POSE_OFF+15, POSE_OFF+16)]
    sym_feats = []
    for li, ri in sym_pairs:
        d = np.linalg.norm(seq[:, li, :2] - seq[:, ri, :2], axis=1)
        sym_feats.extend([d.mean(), d.std()])
    sym_stats = np.array(sym_feats)                      # 6

    # ── Apertura media de mano derecha ─────────────────────────
    # Fingertip ↔ MCP para los 5 dedos (mano derecha)
    tips = [RH_OFF+4,  RH_OFF+8,  RH_OFF+12, RH_OFF+16, RH_OFF+20]
    mcps = [RH_OFF+2,  RH_OFF+5,  RH_OFF+9,  RH_OFF+13, RH_OFF+17]
    aperture = np.mean([
        np.linalg.norm(seq[:, t, :2] - seq[:, m, :2], axis=1)
        for t, m in zip(tips, mcps)
    ], axis=0)                                           # (T,)
    aper_stats = np.array([aperture.mean(), aperture.std()])   # 2

    return np.concatenate([v1, orient_stats, sym_stats, aper_stats])   # 1814


print("  Extrayendo Baseline (900 f)...")
X_base = np.array([extract_baseline(s) for s in seqs])
print("  Extrayendo Variante 1 (1804 f)...")
X_var1 = np.array([extract_var1(s) for s in seqs])
print("  Extrayendo Variante 2 (1814 f)...")
X_var2 = np.array([extract_var2(s) for s in seqs])

print(f"  X_base={X_base.shape}  X_var1={X_var1.shape}  X_var2={X_var2.shape}")

# ─────────────────────────────────────────────────────────────────
# 3. ABLACIÓN CON GROUPKFOLD (5 folds, grupos = viñeta)
# ─────────────────────────────────────────────────────────────────
print("\n[3/6] Ablación GroupKFold(5) — por viñeta...")

gkf = GroupKFold(n_splits=5)

CONFIGS = {
    "Baseline (coords)":              X_base,
    "Var1 (+veloc+acel+dist)":        X_var1,
    "Var2 (+orient+simetría+apertura)": X_var2,
}

ablation_rows = []

for cfg_name, X_cfg in CONFIGS.items():
    fold_metrics = []
    print(f"\n  ── {cfg_name} ({X_cfg.shape[1]} features) ──")
    for fold_i, (tr_idx, te_idx) in enumerate(gkf.split(X_cfg, y_enc, groups=video_ids)):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(C=1.0, max_iter=300,
                                          solver="lbfgs", n_jobs=1,
                                          random_state=42)),
        ])
        pipe.fit(X_cfg[tr_idx], y_enc[tr_idx])

        t_inf = time.perf_counter()
        y_pred = pipe.predict(X_cfg[te_idx])
        lat_ms = (time.perf_counter() - t_inf) / len(te_idx) * 1000

        acc  = accuracy_score(y_enc[te_idx], y_pred)
        f1m  = f1_score(y_enc[te_idx], y_pred, average="macro",    zero_division=0)
        f1w  = f1_score(y_enc[te_idx], y_pred, average="weighted", zero_division=0)

        fold_metrics.append({"fold": fold_i+1, "acc": acc, "f1_macro": f1m,
                              "f1_weighted": f1w, "lat_ms": lat_ms})
        print(f"    fold {fold_i+1}: acc={acc:.3f} f1m={f1m:.3f} f1w={f1w:.3f} lat={lat_ms:.3f}ms")

    df_fold = pd.DataFrame(fold_metrics)
    row = {
        "Configuración": cfg_name,
        "Features":      X_cfg.shape[1],
        "Acc_mean":      df_fold["acc"].mean(),
        "Acc_std":       df_fold["acc"].std(),
        "F1m_mean":      df_fold["f1_macro"].mean(),
        "F1m_std":       df_fold["f1_macro"].std(),
        "F1w_mean":      df_fold["f1_weighted"].mean(),
        "Lat_ms":        df_fold["lat_ms"].mean(),
    }
    ablation_rows.append(row)
    print(f"  → acc={row['Acc_mean']:.3f}±{row['Acc_std']:.3f}  "
          f"f1m={row['F1m_mean']:.3f}±{row['F1m_std']:.3f}")

df_ablation = pd.DataFrame(ablation_rows)
df_ablation.to_csv(DATA_DIR / "calibracion_ablacion.csv", index=False)
print(f"\n  Guardado: data/calibracion_ablacion.csv")
print(df_ablation[["Configuración","Acc_mean","F1m_mean","F1m_std"]].to_string(index=False))

# ─────────────────────────────────────────────────────────────────
# 4. CURVAS DE APRENDIZAJE (mejor configuración)
# ─────────────────────────────────────────────────────────────────
print("\n[4/6] Curvas de aprendizaje...")

# Elegir la configuración con mejor F1m_mean
best_cfg = df_ablation.loc[df_ablation["F1m_mean"].idxmax(), "Configuración"]
best_X   = {"Baseline (coords)": X_base,
             "Var1 (+veloc+acel+dist)": X_var1,
             "Var2 (+orient+simetría+apertura)": X_var2}[best_cfg]
print(f"  Mejor config: {best_cfg}")

pipe_lc = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(C=1.0, max_iter=300, solver="lbfgs", n_jobs=1, random_state=42)),
])

train_sizes, tr_scores, val_scores = learning_curve(
    pipe_lc, best_X, y_enc,
    train_sizes=np.linspace(0.1, 1.0, 8),
    cv=GroupKFold(n_splits=5),
    groups=video_ids,
    scoring="f1_macro",
    n_jobs=1,   # n_jobs=-1 se traba con multiprocessing cuando F1=0
    shuffle=False,
)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f"Curvas de Aprendizaje — {best_cfg}\nTraductor LSP (GroupKFold/5, F1-macro)",
             fontsize=11, fontweight="bold")

# ── Panel 1: Curva de aprendizaje ──────────────────────────────
ax = axes[0]
tr_m  = tr_scores.mean(1);   tr_s  = tr_scores.std(1)
val_m = val_scores.mean(1);  val_s = val_scores.std(1)
ax.plot(train_sizes, tr_m,  "o-", color="#2196F3", lw=2, label="Train F1-macro")
ax.fill_between(train_sizes, tr_m-tr_s, tr_m+tr_s, alpha=0.15, color="#2196F3")
ax.plot(train_sizes, val_m, "s--", color="#FF5722", lw=2, label="Val F1-macro")
ax.fill_between(train_sizes, val_m-val_s, val_m+val_s, alpha=0.15, color="#FF5722")
ax.set_xlabel("Segmentos de entrenamiento"); ax.set_ylabel("F1-macro")
ax.set_title("Curva de aprendizaje"); ax.legend(); ax.grid(alpha=0.3)

# ── Panel 2: Diagnóstico bias/varianza ─────────────────────────
ax2 = axes[1]
gap = tr_m - val_m
ax2.bar(range(len(gap)), gap, color=["#e74c3c" if g > 0.15 else "#2ecc71" for g in gap])
ax2.axhline(0.15, color="red", ls="--", lw=1.5, label="Umbral overfitting (0.15)")
ax2.set_xticks(range(len(gap)))
ax2.set_xticklabels([f"{int(s)}" for s in train_sizes], rotation=30, ha="right")
ax2.set_xlabel("Tamaño de train set")
ax2.set_ylabel("Brecha Train - Val")
ax2.set_title("Diagnóstico bias/varianza\n(verde=OK, rojo=overfitting)")
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

# ── Panel 3: Tabla de ablaciones ───────────────────────────────
ax3 = axes[2]
ax3.axis("off")
tbl_data = []
for _, r in df_ablation.iterrows():
    tbl_data.append([
        r["Configuración"].replace(" (+", "\n(+"),
        f"{r['Acc_mean']:.3f}±{r['Acc_std']:.3f}",
        f"{r['F1m_mean']:.3f}±{r['F1m_std']:.3f}",
        f"{r['Lat_ms']:.3f}ms",
    ])
tbl = ax3.table(
    cellText=tbl_data,
    colLabels=["Configuración", "Accuracy", "F1-macro", "Latencia"],
    loc="center", cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.2, 2.0)
ax3.set_title("Tabla de ablaciones (5-fold)", fontweight="bold", pad=12)

plt.tight_layout()
plt.savefig(DATA_DIR / "calibracion_curvas.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Guardado: data/calibracion_curvas.png")

# ─────────────────────────────────────────────────────────────────
# 5. IMPORTANCIA DE VARIABLES (Permutation Importance)
# ─────────────────────────────────────────────────────────────────
print("\n[5/6] Importancia de variables (permutation_importance)...")

# Entrenar en 80% del total, evaluar en 20%
n_total = len(best_X)
n_tr    = int(n_total * 0.80)

# Split respetando viñetas: usar los grupos del primer fold de GroupKFold
gkf_single = GroupKFold(n_splits=5)
tr_idx_pi, te_idx_pi = next(gkf_single.split(best_X, y_enc, groups=video_ids))

pipe_pi = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(C=1.0, max_iter=300, solver="lbfgs", n_jobs=1, random_state=42)),
])
pipe_pi.fit(best_X[tr_idx_pi], y_enc[tr_idx_pi])

print(f"  Calculando PI sobre {len(te_idx_pi)} muestras (n_repeats=10)...")
t_pi = time.time()
pi = permutation_importance(
    pipe_pi, best_X[te_idx_pi], y_enc[te_idx_pi],
    n_repeats=10, scoring="f1_macro", n_jobs=1, random_state=42
)
print(f"  PI calculado en {time.time()-t_pi:.1f}s")

imp_means = pi.importances_mean
imp_stds  = pi.importances_std
top20_idx = np.argsort(imp_means)[::-1][:20]

# Construir nombres de features para la configuración elegida
if best_cfg == "Baseline (coords)":
    # 900 features: mean(225) + std(225) + min(225) + max(225)
    stat_names = ["mean", "std", "min", "max"]
    feat_names = []
    for stat in stat_names:
        for kp in KP_NAMES:
            for c in COORD_NAMES:
                feat_names.append(f"{stat}_{kp}_{c}")
elif best_cfg == "Var1 (+veloc+acel+dist)":
    # 1804 features
    feat_names = []
    for stat in ["mean", "std", "min", "max"]:
        for kp in KP_NAMES:
            for c in COORD_NAMES:
                feat_names.append(f"{stat}_{kp}_{c}")
    for stat in ["mean", "std"]:
        for kp in KP_NAMES:
            for c in COORD_NAMES:
                feat_names.append(f"vel_{stat}_{kp}_{c}")
    for stat in ["mean", "std"]:
        for kp in KP_NAMES:
            for c in COORD_NAMES:
                feat_names.append(f"acc_{stat}_{kp}_{c}")
    feat_names += ["dist_manos_mean", "dist_manos_std",
                   "dist_manos_min",  "dist_manos_max"]
else:
    # Var2: igual que Var1 + extras
    feat_names = []
    for stat in ["mean", "std", "min", "max"]:
        for kp in KP_NAMES:
            for c in COORD_NAMES:
                feat_names.append(f"{stat}_{kp}_{c}")
    for stat in ["mean", "std"]:
        for kp in KP_NAMES:
            for c in COORD_NAMES:
                feat_names.append(f"vel_{stat}_{kp}_{c}")
    for stat in ["mean", "std"]:
        for kp in KP_NAMES:
            for c in COORD_NAMES:
                feat_names.append(f"acc_{stat}_{kp}_{c}")
    feat_names += ["dist_manos_mean", "dist_manos_std",
                   "dist_manos_min",  "dist_manos_max"]
    feat_names += ["orient_mean", "orient_std"]
    feat_names += ["sym_shoulder_mean", "sym_shoulder_std",
                   "sym_elbow_mean",    "sym_elbow_std",
                   "sym_wrist_mean",    "sym_wrist_std"]
    feat_names += ["apertura_mean", "apertura_std"]

# Truncar por si hay discrepancia
feat_names = feat_names[:best_X.shape[1]]
if len(feat_names) < best_X.shape[1]:
    feat_names += [f"feat_{i}" for i in range(len(feat_names), best_X.shape[1])]

top20_names = [feat_names[i] for i in top20_idx]
top20_means = imp_means[top20_idx]
top20_stds  = imp_stds[top20_idx]

fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#e74c3c" if "vel_" in n or "acc_" in n
          else "#2196F3" if "orient" in n or "sym_" in n or "aper" in n or "dist_" in n
          else "#27ae60"
          for n in top20_names]
bars = ax.barh(range(20)[::-1], top20_means[::-1],
               xerr=top20_stds[::-1], color=colors[::-1],
               align="center", alpha=0.85, capsize=3)
ax.set_yticks(range(20)[::-1])
ax.set_yticklabels([n.replace("_", " ") for n in top20_names[::-1]], fontsize=8)
ax.set_xlabel("Caída en F1-macro al permutar (Importancia)")
ax.set_title(f"Top-20 Features más importantes — Permutation Importance\n"
             f"Configuración: {best_cfg}", fontweight="bold")

from matplotlib.patches import Patch
legend_elems = [
    Patch(color="#27ae60", label="Coordenadas base (mean/std/min/max)"),
    Patch(color="#e74c3c", label="Dinámicas (velocidad / aceleración)"),
    Patch(color="#2196F3", label="Lingüísticas (orientación, simetría, distancia)"),
]
ax.legend(handles=legend_elems, loc="lower right", fontsize=8)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(DATA_DIR / "calibracion_importancia.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Guardado: data/calibracion_importancia.png")

# Imprimir top-10
print("\n  Top-10 features más importantes:")
for rank, (name, m, s) in enumerate(zip(top20_names[:10],
                                         top20_means[:10], top20_stds[:10])):
    print(f"    {rank+1:2d}. {name:<40}  PI={m:.4f} ± {s:.4f}")

# ─────────────────────────────────────────────────────────────────
# 6. CALIBRACIÓN DE PROBABILIDADES (ECE + curvas)
# ─────────────────────────────────────────────────────────────────
print("\n[6/6] Calibración de probabilidades (ECE)...")

pipe_cal = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(C=1.0, max_iter=300, solver="lbfgs", n_jobs=1, random_state=42)),
])
pipe_cal.fit(best_X[tr_idx_pi], y_enc[tr_idx_pi])
probs = pipe_cal.predict_proba(best_X[te_idx_pi])   # (n_test, 26)
y_te  = y_enc[te_idx_pi]


def ece(y_true_bin, y_prob_1d, n_bins=10):
    """Expected Calibration Error (binning uniforme)."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob_1d >= lo) & (y_prob_1d < hi)
        if mask.sum() == 0:
            continue
        acc_b  = y_true_bin[mask].mean()
        conf_b = y_prob_1d[mask].mean()
        ece_val += (mask.mean()) * abs(acc_b - conf_b)
    return ece_val


# Calcular ECE por clase y promedio
# pipe_cal.classes_ puede tener menos clases que le.classes_ si alguna
# no apareció en el fold de entrenamiento (GroupKFold estricto por viñeta)
trained_classes = pipe_cal.named_steps["clf"].classes_  # índices numéricos vistos
ece_scores = []
for cls_i in range(len(le.classes_)):
    y_bin = (y_te == cls_i).astype(int)
    # Buscar columna correcta; si la clase no estaba en train → prob=0
    col = np.where(trained_classes == cls_i)[0]
    if len(col) == 0:
        prob_cls = np.zeros(len(y_te))
    else:
        prob_cls = probs[:, col[0]]
    ece_scores.append(ece(y_bin, prob_cls))

ece_mean = np.mean(ece_scores)
# Brier multiclase: MSE entre probs y one-hot targets alineados con trained_classes
y_te_onehot = np.zeros_like(probs)
for i, cls_i in enumerate(y_te):
    col = np.where(trained_classes == cls_i)[0]
    if len(col) > 0:
        y_te_onehot[i, col[0]] = 1.0
brier = float(np.mean(np.sum((probs - y_te_onehot) ** 2, axis=1)))
print(f"  ECE promedio:  {ece_mean:.4f}")
print(f"  Brier score:   {brier:.4f}")
if ece_mean < 0.05:
    calib_status = "✅ Bien calibrado (ECE < 0.05)"
elif ece_mean < 0.10:
    calib_status = "⚠️  Calibración aceptable (0.05 ≤ ECE < 0.10)"
else:
    calib_status = "❌ Mal calibrado — considerar Platt scaling o temperatura"
print(f"  Status: {calib_status}")

# Seleccionar las 6 clases con más muestras en test para graficar
class_counts_te = {c: (y_te == c).sum() for c in range(len(le.classes_))}
top6_cls = sorted(class_counts_te, key=lambda c: -class_counts_te[c])[:6]

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle(f"Curvas de Calibración por Clase (Top-6 clases en test)\n"
             f"ECE global = {ece_mean:.4f} — {calib_status}",
             fontsize=10, fontweight="bold")

for ax, cls_i in zip(axes.flat, top6_cls):
    y_bin = (y_te == cls_i).astype(int)
    col = np.where(trained_classes == cls_i)[0]
    prob_cls = probs[:, col[0]] if len(col) > 0 else np.zeros(len(y_te))
    if y_bin.sum() == 0:
        ax.set_visible(False)
        continue
    frac_pos, mean_pred = calibration_curve(y_bin, prob_cls, n_bins=8, strategy="uniform")
    ax.plot(mean_pred, frac_pos, "s-", color="#2196F3", lw=2, label="Modelo")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1.5, label="Perfecta")
    ax.fill_between([0, 1], [0, 0.1], [0.1, 0.2], alpha=0.07, color="red",
                     label="Zona error > 0.10")
    cls_name = le.classes_[cls_i]
    ece_i    = ece_scores[cls_i]
    n_pos    = y_bin.sum()
    ax.set_title(f"{cls_name} (n={n_pos})\nECE={ece_i:.4f}", fontsize=9)
    ax.set_xlabel("Prob. predicha"); ax.set_ylabel("Fracción positivos")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(DATA_DIR / "calibracion_calibracion.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Guardado: data/calibracion_calibracion.png")

# ─────────────────────────────────────────────────────────────────
# 7. RESUMEN FINAL
# ─────────────────────────────────────────────────────────────────
import datetime

resumen_lines = [
    "=" * 65,
    "CALIBRACIÓN SPRINT SEMANA 6 — TRADUCTOR LSP",
    f"Fecha: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
    "=" * 65,
    "",
    "── DATOS ─────────────────────────────────────────────────────",
    f"  Segmentos totales : {len(seqs)}",
    f"  Clases            : {len(le.classes_)}",
    f"  Viñetas (grupos)  : {len(np.unique(video_ids))}",
    f"  Shape entrada     : (30, 75, 3)",
    "",
    "── ABLACIÓN (GroupKFold / 5 folds) ───────────────────────────",
]
for _, r in df_ablation.iterrows():
    resumen_lines.append(
        f"  {r['Configuración']:<42}  "
        f"Acc={r['Acc_mean']:.3f}±{r['Acc_std']:.3f}  "
        f"F1m={r['F1m_mean']:.3f}±{r['F1m_std']:.3f}  "
        f"lat={r['Lat_ms']:.3f}ms"
    )
resumen_lines += [
    "",
    "── MEJOR CONFIGURACIÓN ────────────────────────────────────────",
    f"  {best_cfg}",
    f"  {best_X.shape[1]} features",
    "",
    "── TOP-10 FEATURES (Permutation Importance) ───────────────────",
]
for rank, (name, m, s) in enumerate(zip(top20_names[:10],
                                         top20_means[:10], top20_stds[:10])):
    resumen_lines.append(f"  {rank+1:2d}. {name:<40}  PI={m:.4f}±{s:.4f}")
resumen_lines += [
    "",
    "── CALIBRACIÓN DE PROBABILIDADES ─────────────────────────────",
    f"  ECE global  : {ece_mean:.4f}",
    f"  Brier score : {brier:.4f}",
    f"  Status      : {calib_status}",
    "",
    "── ARCHIVOS GENERADOS ─────────────────────────────────────────",
    "  data/calibracion_ablacion.csv",
    "  data/calibracion_curvas.png",
    "  data/calibracion_importancia.png",
    "  data/calibracion_calibracion.png",
    "  data/calibracion_resumen.txt",
    "",
    "── RECOMENDACIONES ────────────────────────────────────────────",
]

# Diagnóstico automático
best_row = df_ablation.loc[df_ablation["F1m_mean"].idxmax()]
if best_row["F1m_mean"] > df_ablation.iloc[0]["F1m_mean"] + 0.02:
    resumen_lines.append("  ✅ Las features derivadas mejoran >2% → incluir en pipeline DL.")
else:
    resumen_lines.append("  ⚠️  Las features derivadas no mejoran >2% → priorizar regularización.")

gap_final = float(tr_scores[-1].mean() - val_scores[-1].mean())
if gap_final > 0.15:
    resumen_lines.append("  ❌ Brecha train-val>15% → overfitting. Aumentar dropout/weight_decay.")
elif gap_final < 0.03 and val_scores[-1].mean() < 0.75:
    resumen_lines.append("  ❌ Val F1<75% y brecha pequeña → underfitting. Añadir capacidad al modelo.")
else:
    resumen_lines.append("  ✅ Brecha train-val aceptable — modelo bien generalizado.")

if ece_mean > 0.10:
    resumen_lines.append("  ⚠️  ECE>10% → aplicar temperatura scaling o Platt scaling.")
else:
    resumen_lines.append("  ✅ ECE aceptable — probabilidades suficientemente calibradas.")

resumen_lines.append("=" * 65)

resumen_txt = "\n".join(resumen_lines)
(DATA_DIR / "calibracion_resumen.txt").write_text(resumen_txt, encoding="utf-8")

print("\n" + resumen_txt)
print("\n✅ CALIBRACIÓN COMPLETADA")
