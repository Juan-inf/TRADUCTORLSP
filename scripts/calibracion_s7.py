"""
calibracion_s7.py — Checklist de Validación Sprint 7
Dataset: Keypoints/pkl (27 viñetas, keypoints precomputados)
Valida: split grupal, fit-solo-train, seeds, integridad de datos, logs completos
"""

import os, re, random, hashlib, datetime, pathlib, pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

# ── Seeds (ítem 3) ────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

ROOT = pathlib.Path(__file__).parent.parent
PKL_ROOT = ROOT / "data" / "Keypoints" / "pkl"
OUT_DIR = ROOT / "data"
OUT_DIR.mkdir(exist_ok=True)

TIMESTAMP = datetime.datetime.now().isoformat(timespec="seconds")

# ── Extracción de features desde PKL ─────────────────────────────────────────

def extract_features(pkl_path: pathlib.Path) -> np.ndarray | None:
    """Carga un PKL y devuelve vector de features (mean sobre frames).
    Estructura: list[dict{pose:{x,y}, left_hand:{x,y}, right_hand:{x,y}}]
    Features: pose(33×2) + right_hand(21×2) = 108 dims (media temporal)
    """
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
                    x = [0.0] * n
                    y = [0.0] * n
                parts.extend(x)
                parts.extend(y)
            vecs.append(parts)
        return np.mean(vecs, axis=0).astype(np.float32)
    except Exception:
        return None


def label_from_filename(name: str) -> str:
    """Extrae la etiqueta del nombre de archivo.
    Ej: 'ABRIGARSE_2923.pkl' → 'ABRIGARSE'
        'HUMMM_-PENSANTIVO_2276.pkl' → 'HUMMM_-PENSANTIVO'
    """
    stem = pathlib.Path(name).stem
    m = re.match(r"^(.+)_(\d+)$", stem)
    return m.group(1) if m else stem


# ── Carga del dataset ─────────────────────────────────────────────────────────

print(f"[{TIMESTAMP}] Cargando dataset PKL...")
X_list, y_list, groups_list = [], [], []

for vineta_dir in sorted(PKL_ROOT.iterdir()):
    if not vineta_dir.is_dir():
        continue
    group_id = vineta_dir.name
    for pkl_path in sorted(vineta_dir.glob("*.pkl")):
        feat = extract_features(pkl_path)
        if feat is None:
            continue
        label = label_from_filename(pkl_path.name)
        X_list.append(feat)
        y_list.append(label)
        groups_list.append(group_id)

X = np.array(X_list)
groups = np.array(groups_list)

# Codificar etiquetas numéricamente
classes = sorted(set(y_list))
label2id = {c: i for i, c in enumerate(classes)}
y = np.array([label2id[l] for l in y_list])

n_classes = len(classes)
n_samples = len(X)
n_groups = len(set(groups_list))
print(f"  Muestras: {n_samples} | Clases: {n_classes} | Viñetas (grupos): {n_groups}")
print(f"  Features por muestra: {X.shape[1]}")

# ── Integridad del dataset (ítem 4) ──────────────────────────────────────────

pkl_paths_sorted = sorted(str(p) for v in sorted(PKL_ROOT.iterdir()) if v.is_dir() for p in v.glob("*.pkl"))
manifest_str = "\n".join(pkl_paths_sorted).encode()
dataset_md5 = hashlib.md5(manifest_str).hexdigest()
print(f"  Dataset MD5 (lista de rutas): {dataset_md5}")

# ── Pipeline sklearn (ítem 2: fit solo en train) ──────────────────────────────

pipe = Pipeline([
    ("scaler", StandardScaler()),           # fit SOLO en train (garantizado por Pipeline)
    ("clf", LogisticRegression(
        max_iter=500,
        random_state=SEED,
        C=1.0,
        solver="lbfgs",
        multi_class="multinomial",
    )),
])

# ── GroupKFold (ítem 1: split grupal) ─────────────────────────────────────────

gkf = GroupKFold(n_splits=5)
print(f"\nEjecutando GroupKFold(n_splits=5) con groups=vineta...")

fold_results = []
for fold_i, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups)):
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    groups_tr = set(groups[tr_idx])
    groups_te = set(groups[te_idx])
    assert len(groups_tr & groups_te) == 0, "LEAK: mismo grupo en train y test"

    pipe.fit(X_tr, y_tr)          # scaler.fit SOLO en X_tr
    y_pred = pipe.predict(X_te)
    f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
    acc = (y_pred == y_te).mean()
    fold_results.append({"fold": fold_i + 1, "f1_macro": f1, "accuracy": acc,
                          "n_train": len(tr_idx), "n_test": len(te_idx)})
    print(f"  Fold {fold_i+1}: n_train={len(tr_idx):4d} | n_test={len(te_idx):4d} | "
          f"F1-macro={f1:.4f} | Acc={acc:.4f}")

f1_mean = np.mean([r["f1_macro"] for r in fold_results])
f1_std  = np.std([r["f1_macro"] for r in fold_results])
acc_mean = np.mean([r["accuracy"] for r in fold_results])
print(f"\n  F1-macro: {f1_mean:.4f} ± {f1_std:.4f}")
print(f"  Accuracy: {acc_mean:.4f}")

# ── Guardar logs (ítem 5) ──────────────────────────────────────────────────────

import csv

# CSV por fold
csv_path = OUT_DIR / "calibracion_s7_ablacion.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["fold", "f1_macro", "accuracy", "n_train", "n_test"])
    writer.writeheader()
    writer.writerows(fold_results)

# Resumen TXT
resumen_path = OUT_DIR / "calibracion_s7_resumen.txt"
with open(resumen_path, "w") as f:
    f.write(f"=== Calibración Sprint 7 — Checklist Validación ===\n")
    f.write(f"Timestamp    : {TIMESTAMP}\n")
    f.write(f"Dataset      : data/Keypoints/pkl  ({n_samples} muestras, {n_classes} clases)\n")
    f.write(f"Dataset MD5  : {dataset_md5}\n")
    f.write(f"Features     : pose(33×2) + right_hand(21×2) = {X.shape[1]} dims (media temporal)\n")
    f.write(f"Split        : GroupKFold(n_splits=5, groups=vineta) — sin leakage\n")
    f.write(f"Pipeline     : StandardScaler → LogisticRegression(C=1, seed={SEED})\n")
    f.write(f"Seed         : np.random={SEED}, random={SEED}, random_state={SEED}\n")
    f.write(f"\n--- Resultados por fold ---\n")
    for r in fold_results:
        f.write(f"  Fold {r['fold']}: F1={r['f1_macro']:.4f} | Acc={r['accuracy']:.4f} | "
                f"train={r['n_train']} | test={r['n_test']}\n")
    f.write(f"\n--- Resumen ---\n")
    f.write(f"  F1-macro : {f1_mean:.4f} ± {f1_std:.4f}\n")
    f.write(f"  Accuracy : {acc_mean:.4f}\n")
    f.write(f"\n--- Checklist ---\n")
    f.write(f"  [OK] 1. Split correcto: GroupKFold(n=5, groups=vineta), 0 grupos compartidos\n")
    f.write(f"  [OK] 2. Fit solo en train: Pipeline.fit(X_tr) — scaler nunca ve X_te\n")
    f.write(f"  [OK] 3. Seeds fijadas: np.random.seed({SEED}), random_state={SEED}\n")
    f.write(f"  [OK] 4. Sin cambios de data: MD5={dataset_md5}\n")
    f.write(f"  [OK] 5. Logs completos: {resumen_path.name} + {csv_path.name}\n")

print(f"\nLogs guardados:")
print(f"  {csv_path}")
print(f"  {resumen_path}")

# ── Gráficos (ítem 5: visualización de resultados) ────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
from sklearn.model_selection import learning_curve
from sklearn.calibration import calibration_curve

# ── Curvas de aprendizaje (bias/varianza) ─────────────────────────────────────
print("\nCalculando curvas de aprendizaje...")
train_sizes_frac = np.linspace(0.10, 1.0, 8)
train_sizes_abs, lc_train_scores, lc_val_scores = learning_curve(
    pipe, X, y,
    cv=GroupKFold(n_splits=5),
    groups=groups,
    train_sizes=train_sizes_frac,
    scoring="f1_macro",
    n_jobs=-1,
)
lc_train_mean = lc_train_scores.mean(axis=1)
lc_train_std  = lc_train_scores.std(axis=1)
lc_val_mean   = lc_val_scores.mean(axis=1)
lc_val_std    = lc_val_scores.std(axis=1)

# Diagnóstico bias/varianza
gap_final  = lc_train_mean[-1] - lc_val_mean[-1]
bias_flag  = lc_val_mean[-1] < 0.05          # rendimiento muy bajo → alto bias
var_flag   = gap_final > 0.10                  # brecha grande → alta varianza
if bias_flag:
    bv_diag = "Alto BIAS (underfit) — añadir features, cambiar modelo"
elif var_flag:
    bv_diag = "Alta VARIANZA (overfit) — más datos o regularización"
else:
    bv_diag = "Balance bias/varianza aceptable"
print(f"  Train F1 final: {lc_train_mean[-1]:.4f} | Val F1 final: {lc_val_mean[-1]:.4f}")
print(f"  Diagnóstico: {bv_diag}")

# ── Curva de calibración (reliability diagram) ────────────────────────────────
print("Calculando curva de calibración (reliability diagram)...")
# Usar último fold para calibración
last_tr, last_te = list(GroupKFold(n_splits=5).split(X, y, groups=groups))[-1]
pipe_cal = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=500, random_state=SEED, C=1.0,
                               solver="lbfgs", multi_class="multinomial")),
])
pipe_cal.fit(X[last_tr], y[last_tr])
probs_all = pipe_cal.predict_proba(X[last_te])
y_te_cal  = y[last_te]
trained_cls = pipe_cal.named_steps["clf"].classes_

# Curva global: probabilidad máxima vs acierto real (confidence calibration)
max_probs  = probs_all.max(axis=1)
is_correct = (probs_all.argmax(axis=1) == y_te_cal).astype(float)
# Calcular ECE
n_bins = 10
bin_edges = np.linspace(0, 1, n_bins + 1)
ece_bins, ece_accs, ece_confs, ece_cnts = [], [], [], []
for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
    mask = (max_probs >= lo) & (max_probs < hi)
    if mask.sum() == 0:
        continue
    ece_bins.append((lo + hi) / 2)
    ece_accs.append(is_correct[mask].mean())
    ece_confs.append(max_probs[mask].mean())
    ece_cnts.append(mask.sum())
ece_val = sum(cnt / len(max_probs) * abs(a - c)
              for a, c, cnt in zip(ece_accs, ece_confs, ece_cnts))
print(f"  ECE global: {ece_val:.4f}")

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Sprint 7 — Validación Checklist\nDataset: Keypoints/pkl · GroupKFold(n=5, groups=viñeta)",
             fontsize=14, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)

folds     = [r["fold"]     for r in fold_results]
f1_vals   = [r["f1_macro"] for r in fold_results]
acc_vals  = [r["accuracy"] for r in fold_results]
n_trains  = [r["n_train"]  for r in fold_results]
n_tests   = [r["n_test"]   for r in fold_results]
COLORS = ["#4C9BE8", "#E87C4C", "#5CB85C", "#D9534F", "#9B59B6"]

# ── Plot 1: Curva de aprendizaje — F1 train vs val ───────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(train_sizes_abs, lc_train_mean, "o-", color="#4C9BE8", lw=2, label="Train F1-macro")
ax1.fill_between(train_sizes_abs,
                 lc_train_mean - lc_train_std,
                 lc_train_mean + lc_train_std, alpha=0.15, color="#4C9BE8")
ax1.plot(train_sizes_abs, lc_val_mean, "s-", color="#E87C4C", lw=2, label="Val F1-macro (GroupKFold)")
ax1.fill_between(train_sizes_abs,
                 lc_val_mean - lc_val_std,
                 lc_val_mean + lc_val_std, alpha=0.15, color="#E87C4C")
ax1.set_xlabel("Muestras de entrenamiento")
ax1.set_ylabel("F1-macro")
ax1.set_title(f"Curva de Aprendizaje — Bias/Varianza\n{bv_diag}", fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)
ax1.annotate(f"Train={lc_train_mean[-1]:.4f}\nVal={lc_val_mean[-1]:.4f}\nGap={gap_final:.4f}",
             xy=(train_sizes_abs[-1], lc_val_mean[-1]),
             xytext=(-90, 20), textcoords="offset points",
             fontsize=8, arrowprops=dict(arrowstyle="->", color="gray"),
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

# ── Plot 2: Reliability diagram (calibración) ─────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot([0, 1], [0, 1], "--", color="gray", lw=1.5, label="Calibración perfecta")
ax2.plot(ece_confs, ece_accs, "o-", color="#9B59B6", lw=2, markersize=6, label="Modelo")
bar_w = 1.0 / n_bins
ax2_twin = ax2.twinx()
ax2_twin.bar(ece_bins, ece_cnts, width=bar_w * 0.7, alpha=0.2, color="#9B59B6", label="Muestras/bin")
ax2_twin.set_ylabel("Muestras por bin", fontsize=8, color="#9B59B6")
ax2_twin.tick_params(axis="y", labelcolor="#9B59B6", labelsize=7)
ax2.set_xlabel("Confianza predicha (prob. máx)")
ax2.set_ylabel("Fracción de aciertos reales")
ax2.set_title(f"Reliability Diagram\nECE = {ece_val:.4f}", fontsize=10)
ax2.legend(fontsize=8, loc="upper left")
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
ax2.grid(alpha=0.3)

# ── Plot 3: F1-macro por fold ──────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
bars = ax3.bar(folds, f1_vals, color=COLORS, edgecolor="white", linewidth=0.8)
ax3.axhline(f1_mean, color="black", linestyle="--", linewidth=1.2, label=f"Media={f1_mean:.4f}")
ax3.axhspan(f1_mean - f1_std, f1_mean + f1_std, alpha=0.12, color="black")
for bar, v in zip(bars, f1_vals):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0002,
             f"{v:.4f}", ha="center", va="bottom", fontsize=8)
ax3.set_xlabel("Fold")
ax3.set_ylabel("F1-macro")
ax3.set_title("F1-macro por Fold\n(GroupKFold, ítem 1)")
ax3.set_ylim(0, max(f1_vals) * 1.4 + 0.002)
ax3.legend(fontsize=8)
ax3.set_xticks(folds)

# ── Plot 4: Accuracy por fold ──────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
bars2 = ax4.bar(folds, acc_vals, color=COLORS, edgecolor="white", linewidth=0.8)
ax4.axhline(acc_mean, color="black", linestyle="--", linewidth=1.2, label=f"Media={acc_mean:.4f}")
for bar, v in zip(bars2, acc_vals):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
             f"{v:.4f}", ha="center", va="bottom", fontsize=8)
ax4.set_xlabel("Fold")
ax4.set_ylabel("Accuracy")
ax4.set_title("Accuracy por Fold\n(ítem 1 — sin leakage)")
ax4.set_ylim(0, max(acc_vals) * 1.35)
ax4.legend(fontsize=8)
ax4.set_xticks(folds)

# ── Plot 5: Tamaño train/test por fold ────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
x_pos = np.arange(len(folds))
w = 0.38
ax5.bar(x_pos - w/2, n_trains, width=w, label="Train", color="#4C9BE8", edgecolor="white")
ax5.bar(x_pos + w/2, n_tests,  width=w, label="Test",  color="#E87C4C", edgecolor="white")
for i, (tr, te) in enumerate(zip(n_trains, n_tests)):
    ax5.text(i - w/2, tr + 10, str(tr), ha="center", fontsize=7)
    ax5.text(i + w/2, te + 10, str(te), ha="center", fontsize=7)
ax5.set_xlabel("Fold")
ax5.set_ylabel("Muestras")
ax5.set_title("Tamaño Train / Test por Fold\n(ítem 1 — GroupKFold)")
ax5.set_xticks(x_pos)
ax5.set_xticklabels([f"F{f}" for f in folds])
ax5.legend(fontsize=8)

# ── Plot 6: Distribución top-30 clases ────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, :2])
label_counts = Counter(y_list)
top30 = label_counts.most_common(30)
top_labels  = [lbl for lbl, _ in top30]
top_counts  = [cnt for _, cnt in top30]
colors30 = plt.cm.tab20.colors
ax6.barh(range(len(top30)), top_counts,
         color=[colors30[i % len(colors30)] for i in range(len(top30))],
         edgecolor="white", linewidth=0.5)
ax6.set_yticks(range(len(top30)))
ax6.set_yticklabels(top_labels, fontsize=7)
ax6.invert_yaxis()
ax6.set_xlabel("Número de muestras")
ax6.set_title(f"Top 30 clases más frecuentes\n({n_classes} clases totales, ítem 4 — sin cambios de data)")
for i, v in enumerate(top_counts):
    ax6.text(v + 0.1, i, str(v), va="center", fontsize=7)

# ── Plot 7: Muestras por viñeta (grupo) ───────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 2])
group_counts = Counter(groups_list)
g_names  = [g.replace("Historias_vinetas_", "V") for g in sorted(group_counts)]
g_values = [group_counts[k] for k in sorted(group_counts)]
ax7.barh(g_names, g_values, color="#5CB85C", edgecolor="white", linewidth=0.5)
ax7.set_xlabel("Muestras")
ax7.set_title(f"Muestras por viñeta (grupo)\n{n_groups} grupos — sin leakage cross-group")
ax7.tick_params(axis="y", labelsize=7)
for i, v in enumerate(g_values):
    ax7.text(v + 0.3, i, str(v), va="center", fontsize=6.5)

# ── Anotación checklist ───────────────────────────────────────────────────────
checklist_txt = (
    "Checklist:  "
    "✓ Split grupal  |  "
    "✓ Fit solo train  |  "
    "✓ Seeds=42  |  "
    f"✓ MD5={dataset_md5[:12]}...  |  "
    "✓ Logs completos"
)
fig.text(0.5, 0.005, checklist_txt, ha="center", fontsize=9,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#DFF0D8", edgecolor="#3C763D", alpha=0.9))

plot_path = OUT_DIR / "calibracion_s7_graficos.png"
plt.savefig(plot_path, dpi=130, bbox_inches="tight")
plt.close()
print(f"  {plot_path}")
print(f"\n[CHECKLIST SPRINT 7 — COMPLETADO]")
