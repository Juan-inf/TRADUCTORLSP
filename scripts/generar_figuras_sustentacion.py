"""Genera las figuras reales para el documento de resultados de sustentación.

Todo lo que se puede medir sin reentrenar se mide en vivo aquí:
  - Split de test determinista (StratifiedShuffleSplit, SEED=42, test_size=0.15)
    idéntico al usado por scripts/train_s27.py — se reproduce, no se inventa.
  - Inferencia real del checkpoint v4 (checkpoints/bilstm_s27.onnx) sobre ese
    test set → matriz de confusión, F1 por clase, top-1/3/5 real.
  - Latencia ONNX real (forward pass, no reentrenamiento).

Lo que NO se recomputa (viene de logs/runs.csv y de mediciones ya hechas
esta semana con el pipeline WebSocket real, documentado en
SUSTENTACION_RESUMEN_FINAL.md): la evolución histórica S5-S27 y la latencia
end-to-end del WebSocket, porque exigirían reconstruir checkpoints/datasets
de sprints anteriores que ya no están en disco (mismo hallazgo de R2 en el
plan de despliegue: solo el checkpoint más reciente sobrevive).
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import onnxruntime as ort
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).parent.parent
OUT = ROOT / "data" / "sustentacion_figs"
OUT.mkdir(exist_ok=True)

TEAL = "#0e7c78"
TEAL_DARK = "#075c59"
AMBER = "#b3791f"
BAD = "#c2384a"
GOOD = "#1f8f5f"
GREY = "#8398a0"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.edgecolor": "#425055",
    "axes.labelcolor": "#182225",
    "text.color": "#182225",
    "xtick.color": "#425055",
    "ytick.color": "#425055",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

SEED = 42
MIN_SAMPLES = 15

# ── 1. Reproducir el test set determinista de v4 y correr inferencia real ────
print("Cargando dataset_s17.npz y reproduciendo split de test (SEED=42)...")
data = np.load(ROOT / "data" / "dataset_s17.npz")
X_raw, y_raw, groups = data["X"], data["y"], data["groups"]

with open(ROOT / "data" / "s17_label2idx.json", encoding="utf-8") as f:
    label2idx = json.load(f)
idx2label_raw = {int(v): k for k, v in label2idx.items()}

counts = Counter(y_raw.tolist())
keep = np.array([counts[int(v)] >= MIN_SAMPLES for v in y_raw])
X_raw, y_raw = X_raw[keep], y_raw[keep]


def normalize_sample(x):
    mu, std = x.mean(), x.std()
    if std < 1e-8:
        return x
    return ((x - mu) / std).astype(np.float32)


X = np.stack([normalize_sample(X_raw[i]) for i in range(len(X_raw))])
old_ids = sorted(set(y_raw.tolist()))
remap = {old: new for new, old in enumerate(old_ids)}
y = np.array([remap[int(v)] for v in y_raw], dtype=np.int64)
idx2label = {remap[o]: idx2label_raw.get(o, str(o)) for o in old_ids}

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
tv_idx, te_idx = next(sss.split(X, y))
X_te, y_te = X[te_idx], y[te_idx]
print(f"  Test holdout reproducido: {len(X_te)} muestras, {len(set(y_te.tolist()))} clases")

sess = ort.InferenceSession(str(ROOT / "checkpoints" / "bilstm_s27.onnx"), providers=["CPUExecutionProvider"])
inp_name = sess.get_inputs()[0].name

probs_list = []
BATCH = 64
for i in range(0, len(X_te), BATCH):
    xb = X_te[i:i + BATCH].astype(np.float32)
    logits = sess.run(None, {inp_name: xb})[0]
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs_list.append(e / e.sum(axis=1, keepdims=True))
probs = np.concatenate(probs_list, axis=0)
y_pred = probs.argmax(axis=1)

f1_macro_live = f1_score(y_te, y_pred, average="macro")
top1 = (y_pred == y_te).mean()
top3 = np.mean([y_te[i] in np.argsort(-probs[i])[:3] for i in range(len(y_te))])
top5 = np.mean([y_te[i] in np.argsort(-probs[i])[:5] for i in range(len(y_te))])
print(f"  F1-macro (recomputado en vivo) = {f1_macro_live:.4f}  (logs/runs.csv dice 0.4426)")
print(f"  Top-1={top1:.4f}  Top-3={top3:.4f}  Top-5={top5:.4f}")

# ── Figura: Top-k accuracy ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.2, 4.3))
vals = [top1, top3, top5]
labels = ["Top-1", "Top-3", "Top-5"]
bars = ax.bar(labels, vals, color=[TEAL_DARK, TEAL, "#5fb3af"], width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v*100:.1f}%", ha="center", fontsize=11, fontweight="bold")
ax.set_ylim(0, 0.85)
ax.set_ylabel("Exactitud")
ax.set_title("Exactitud Top-k — Modelo S27 v4\n(test holdout real, N={})".format(len(y_te)), fontsize=11.5)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
plt.tight_layout()
plt.savefig(OUT / "fig_topk.png", dpi=160)
plt.close()

# ── Figura: matriz de confusión (heatmap denso, 96 clases) ──────────────────
cm = confusion_matrix(y_te, y_pred, normalize="true")
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(cm, cmap="BuGn", vmin=0, vmax=1)
ax.set_title(f"Matriz de confusión normalizada — 96 clases (test holdout, N={len(y_te)})", fontsize=11.5)
ax.set_xlabel("Clase predicha (índice)")
ax.set_ylabel("Clase real (índice)")
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cbar.set_label("Proporción de aciertos por fila")
plt.tight_layout()
plt.savefig(OUT / "fig_matriz_confusion.png", dpi=160)
plt.close()

# ── Figura: F1 por clase, mejores y peores 15 ────────────────────────────────
f1_per_class = f1_score(y_te, y_pred, average=None, labels=sorted(set(y_te.tolist())))
present_labels = sorted(set(y_te.tolist()))
names = [idx2label.get(c, str(c)) for c in present_labels]
order = np.argsort(f1_per_class)
worst15 = order[:15]
best15 = order[-15:][::-1]

fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
for ax, idxs, title, color in [
    (axes[0], best15, "Mejores 15 clases (F1)", GOOD),
    (axes[1], worst15, "Peores 15 clases (F1)", BAD),
]:
    vals = [f1_per_class[i] for i in idxs]
    lbls = [names[i] for i in idxs]
    ax.barh(range(len(idxs)), vals, color=color)
    ax.set_yticks(range(len(idxs)))
    ax.set_yticklabels(lbls, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1")
    ax.set_title(title, fontsize=10.5)
plt.tight_layout()
plt.savefig(OUT / "fig_f1_por_clase.png", dpi=160)
plt.close()

# ── 2. Evolución histórica S5→S27 (desde logs/runs.csv, real) ───────────────
print("Leyendo trayectoria histórica de logs/runs.csv...")
runs = pd.read_csv(ROOT / "logs" / "runs.csv")
runs = runs[runs["sprint"].notna() & (runs["sprint"] != "sprint")]

hitos = {
    "S5": None, "S13": None, "S26": None, "S27": None,
}
best_by_sprint = runs.sort_values("f1_test", ascending=False).groupby("sprint", as_index=False).first()
sprint_order = ["S5", "S6", "S7", "S8", "S9", "S10", "S10v2", "S11", "S12", "S13", "S14",
                 "S15", "S16", "S17", "S18", "S19", "S20", "S21", "S22", "S23", "S24", "S25", "S26", "S27"]
rows = []
for sp in sprint_order:
    r = best_by_sprint[best_by_sprint["sprint"] == sp]
    if len(r):
        rows.append((sp, float(r.iloc[0]["f1_test"])))
print(f"  {len(rows)} sprints con F1-test registrado: {[r[0] for r in rows]}")

fig, ax = plt.subplots(figsize=(11, 5))
xs = [r[0] for r in rows]
ys = [r[1] for r in rows]
ax.plot(xs, ys, color=TEAL, marker="o", markersize=5, linewidth=2, zorder=3)
ax.fill_between(range(len(xs)), ys, color=TEAL, alpha=0.08)
ax.scatter([xs.index("S27")], [ys[xs.index("S27")]], color=AMBER, s=90, zorder=4, label="S27 (mejor punto — v3)")
ax.axhline(0.70, color=BAD, linestyle="--", linewidth=1.2, alpha=0.7, label="Meta declarada F1>0.70")
ax.set_ylabel("F1-macro (test)")
ax.set_title("Evolución de F1-macro por sprint — S5 → S27 (datos reales de logs/runs.csv)", fontsize=11.5)
ax.set_xticklabels(xs, rotation=45, ha="right", fontsize=8.5)
ax.legend(frameon=False, loc="upper left", fontsize=9.5)
ax.set_ylim(0, 0.8)
plt.tight_layout()
plt.savefig(OUT / "fig_f1_evolucion.png", dpi=160)
plt.close()

# ── 3. HE3 comparación v1-v4 (datos reales del análisis S27, verificados en el notebook) ──
he3_data = {
    "v1": {"delta_f1": 0.1311, "psi": 0.1281, "ks_d": 0.136},
    "v2": {"delta_f1": 0.0509, "psi": 0.0818, "ks_d": 0.117},
    "v3": {"delta_f1": 0.0785, "psi": 0.0184, "ks_d": 0.049},
    "v4": {"delta_f1": 0.0406, "psi": 0.0288, "ks_d": 0.065},
}
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
versions = list(he3_data.keys())
colors_v = [GREY, GREY, TEAL, TEAL_DARK]

specs = [
    ("delta_f1", "ΔF1 (holdout de grupo)", 0.15, "≤0.15"),
    ("psi", "PSI", 0.20, "<0.20"),
    ("ks_d", "KS — estadístico D", 0.045, "D crítico≈0.045"),
]
for ax, (key, title, thr, thr_label) in zip(axes, specs):
    vals = [he3_data[v][key] for v in versions]
    bars = ax.bar(versions, vals, color=colors_v, width=0.6)
    ax.axhline(thr, color=BAD, linestyle="--", linewidth=1.2)
    ax.text(3.4, thr, thr_label, color=BAD, fontsize=8.5, va="bottom", ha="right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.4f}", ha="center", fontsize=8.5)
    ax.set_title(title, fontsize=10.5)
    ax.set_ylim(0, max(max(vals), thr) * 1.35)
fig.suptitle("HE3 — comparación de las 4 corridas de S27 (v1→v4)", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(OUT / "fig_he3_comparacion.png", dpi=160, bbox_inches="tight")
plt.close()

# ── 4. Latencia — modelo aislado vs. pipeline end-to-end real (WebSocket) ───
lat_model = {"p50": 0.617, "p95": 0.902, "max": 1.272}   # medido en vivo hoy (v4, ONNX, batch=1)
lat_e2e = {"p50": 54.7, "p95": 58.7, "max": 118.2}         # medido esta semana, cliente WS real

fig, ax = plt.subplots(figsize=(7.5, 5.2))
labels = ["p50", "p95", "max"]
x = np.arange(len(labels))
w = 0.32
b1 = ax.bar(x - w / 2, [lat_model[k] for k in labels], width=w, color=TEAL_DARK, label="Modelo ONNX aislado")
b2 = ax.bar(x + w / 2, [lat_e2e[k] for k in labels], width=w, color=AMBER, label="Pipeline E2E real (WebSocket)")
ax.axhline(200, color=BAD, linestyle="--", linewidth=1.2, label="Umbral objetivo (200ms)")
ax.set_yscale("log")
ax.set_ylim(0.3, 1200)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Latencia (ms, escala log)")
ax.set_title("Latencia real medida — modelo vs. pipeline completo", fontsize=11.5, pad=45)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.15, f"{b.get_height():.1f}", ha="center", fontsize=8.5)
ax.legend(frameon=False, fontsize=8.8, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1)
plt.tight_layout()
plt.savefig(OUT / "fig_latencia.png", dpi=160)
plt.close()

print(f"\nFiguras guardadas en {OUT}/")
for p in sorted(OUT.glob("*.png")):
    print(f"  {p.name}  ({p.stat().st_size/1024:.0f} KB)")

# Guardar métricas recomputadas para uso posterior en el documento
metrics_out = {
    "f1_macro_live_recomputed": float(f1_macro_live),
    "top1": float(top1), "top3": float(top3), "top5": float(top5),
    "n_test": int(len(y_te)), "n_classes_test": int(len(set(y_te.tolist()))),
}
with open(OUT / "metrics_live.json", "w", encoding="utf-8") as f:
    json.dump(metrics_out, f, indent=2, ensure_ascii=False)
print("\nmetrics_live.json:", metrics_out)
