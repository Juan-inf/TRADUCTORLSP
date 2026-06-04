"""
semana8_hpo.py — Sprint 8: Búsqueda de Hiperparámetros
Random Search vs Bayesian Optimization (Optuna) con pruning/early stopping
Entregables: logs + artefactos, tabla top-k, gráfico evolución, resumen ganador.
"""

import os, re, random, hashlib, datetime, pathlib, pickle, csv, warnings
import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

# ── Constantes ────────────────────────────────────────────────────────────────

SEED      = 42
N_TRIALS  = 10          # presupuesto por método (Random y Bayes)
N_SPLITS  = 5
TOP_K     = 5

np.random.seed(SEED)
random.seed(SEED)

ROOT     = pathlib.Path(__file__).parent.parent
PKL_ROOT = ROOT / "data" / "Keypoints" / "pkl"
OUT_DIR  = ROOT / "data"
OUT_DIR.mkdir(exist_ok=True)

TIMESTAMP = datetime.datetime.now().isoformat(timespec="seconds")

# ── Extracción de features (idéntica a S7) ────────────────────────────────────

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
        X_list.append(feat)
        y_list.append(label_from_filename(pkl_path.name))
        groups_list.append(group_id)

X      = np.array(X_list)
groups = np.array(groups_list)
classes   = sorted(set(y_list))
label2id  = {c: i for i, c in enumerate(classes)}
y         = np.array([label2id[l] for l in y_list])
n_classes = len(classes)
n_samples = len(X)

pkl_paths_sorted = sorted(
    str(p) for v in sorted(PKL_ROOT.iterdir()) if v.is_dir()
    for p in v.glob("*.pkl")
)
dataset_md5 = hashlib.md5("\n".join(pkl_paths_sorted).encode()).hexdigest()

print(f"  Muestras: {n_samples} | Clases: {n_classes} | Features: {X.shape[1]}")
print(f"  Dataset MD5: {dataset_md5}")

# ── Espacio de búsqueda ───────────────────────────────────────────────────────
#
# Dos modelos de árbol (escalan bien con 1086 clases — O(n_samples), no O(n_classes)):
#   rf — RandomForestClassifier (best splits, baja varianza, mayor sesgo)
#   et — ExtraTreesClassifier   (splits aleatorios, mayor varianza, más rápido que RF)

SPACE = {
    "model":                ["rf", "et"],
    # rf — RandomForestClassifier (best splits, O(n_samples) por árbol, sin escalar con n_classes)
    "rf__n_estimators":     (10, 60, "int"),
    "rf__max_depth":        (5, 20, "int"),
    "rf__min_samples_leaf": (1, 8, "int"),
    # et — ExtraTreesClassifier (splits aleatorios, más rápido que RF, misma escala)
    "et__n_estimators":     (10, 60, "int"),
    "et__max_depth":        (5, 20, "int"),
    "et__min_samples_leaf": (1, 8, "int"),
}


def build_pipeline(params):
    """Construye un Pipeline a partir del dict de hiperparámetros."""
    m = params["model"]
    if m == "rf":
        clf = RandomForestClassifier(
            n_estimators=params["rf__n_estimators"],
            max_depth=params["rf__max_depth"],
            min_samples_leaf=params["rf__min_samples_leaf"],
            random_state=SEED,
            n_jobs=4,
        )
    else:  # et
        clf = ExtraTreesClassifier(
            n_estimators=params["et__n_estimators"],
            max_depth=params["et__max_depth"],
            min_samples_leaf=params["et__min_samples_leaf"],
            random_state=SEED,
            n_jobs=4,
        )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def evaluate_pipeline(pipe, X, y, groups, trial=None):
    """Evalúa con GroupKFold. Si trial != None reporta valores intermedios para pruning."""
    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_scores = []
    for fold_i, (tr, te) in enumerate(gkf.split(X, y, groups=groups)):
        pipe.fit(X[tr], y[tr])
        f1 = f1_score(y[te], pipe.predict(X[te]), average="macro", zero_division=0)
        fold_scores.append(f1)

        if trial is not None:
            trial.report(float(np.mean(fold_scores)), fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return float(np.mean(fold_scores)), float(np.std(fold_scores))


# ── Random Search ─────────────────────────────────────────────────────────────

def sample_random_params(rng):
    m = rng.choice(SPACE["model"])
    p = {"model": m}
    for key in ("rf__n_estimators", "rf__max_depth", "rf__min_samples_leaf",
                "et__n_estimators", "et__max_depth", "et__min_samples_leaf"):
        lo, hi, _ = SPACE[key]
        p[key] = int(rng.integers(lo, hi + 1))
    return p


print(f"\n{'='*60}")
print(f"RANDOM SEARCH  (n_trials={N_TRIALS})")
print(f"{'='*60}")

rng = np.random.default_rng(SEED)
rand_trials = []
rand_best_so_far = []

for i in range(N_TRIALS):
    params = sample_random_params(rng)
    pipe   = build_pipeline(params)
    try:
        f1_mean, f1_std = evaluate_pipeline(pipe, X, y, groups)
        status = "ok"
    except Exception as e:
        f1_mean, f1_std, status = 0.0, 0.0, f"error:{e}"

    record = {
        "trial": i + 1, "method": "random",
        "model": params["model"],
        "f1_macro_mean": round(f1_mean, 6),
        "f1_macro_std":  round(f1_std,  6),
        "status": status,
        **{k: v for k, v in params.items() if k != "model"},
    }
    rand_trials.append(record)
    best = max(r["f1_macro_mean"] for r in rand_trials)
    rand_best_so_far.append(best)
    print(f"  Trial {i+1:02d}/{N_TRIALS} | {params['model']:3s} | "
          f"F1={f1_mean:.4f} | best_so_far={best:.4f}")

rand_best_idx  = np.argmax([r["f1_macro_mean"] for r in rand_trials])
rand_best      = rand_trials[rand_best_idx]

# ── Bayesian Optimization (Optuna + MedianPruner) ─────────────────────────────

print(f"\n{'='*60}")
print(f"BAYESIAN SEARCH  (n_trials={N_TRIALS}, pruner=MedianPruner)")
print(f"{'='*60}")

def optuna_objective(trial):
    m = trial.suggest_categorical("model", SPACE["model"])
    params = {"model": m}
    for key in ("rf__n_estimators", "rf__max_depth", "rf__min_samples_leaf",
                "et__n_estimators", "et__max_depth", "et__min_samples_leaf"):
        lo, hi, _ = SPACE[key]
        params[key] = trial.suggest_int(key, lo, hi)

    pipe = build_pipeline(params)
    f1_mean, _ = evaluate_pipeline(pipe, X, y, groups, trial=trial)
    return f1_mean


study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
    sampler=optuna.samplers.TPESampler(seed=SEED),
)

bayes_trials    = []
bayes_best_so_far = []

def _bayes_callback(study, trial_obj):
    if trial_obj.state == optuna.trial.TrialState.COMPLETE:
        f1_mean = trial_obj.value
        f1_std  = 0.0
        status  = "ok"
    elif trial_obj.state == optuna.trial.TrialState.PRUNED:
        f1_mean, f1_std, status = 0.0, 0.0, "pruned"
    else:
        f1_mean, f1_std, status = 0.0, 0.0, "failed"

    params = trial_obj.params
    record = {
        "trial":  trial_obj.number + 1,
        "method": "bayes",
        "model":  params.get("model", "?"),
        "f1_macro_mean": round(f1_mean, 6),
        "f1_macro_std":  round(f1_std, 6),
        "status": status,
        **{k: v for k, v in params.items() if k != "model"},
    }
    bayes_trials.append(record)
    best_complete = [t.value for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
    best = max(best_complete) if best_complete else 0.0
    bayes_best_so_far.append(best)
    pruned_marker = " [PRUNED]" if status == "pruned" else ""
    print(f"  Trial {trial_obj.number+1:02d}/{N_TRIALS} | {params.get('model','?'):3s} | "
          f"F1={f1_mean:.4f} | best_so_far={best:.4f}{pruned_marker}")


study.optimize(optuna_objective, n_trials=N_TRIALS, callbacks=[_bayes_callback],
               catch=(Exception,))

bayes_best_trial = study.best_trial
bayes_best = {
    "trial": bayes_best_trial.number + 1,
    "method": "bayes",
    "model":  bayes_best_trial.params.get("model"),
    "f1_macro_mean": round(bayes_best_trial.value, 6),
    "f1_macro_std":  0.0,
    "status": "ok",
    **{k: v for k, v in bayes_best_trial.params.items() if k != "model"},
}

# ── Top-k de ambos métodos ─────────────────────────────────────────────────────

all_trials = rand_trials + [t for t in bayes_trials if t["status"] == "ok"]
all_trials_sorted = sorted(all_trials, key=lambda r: r["f1_macro_mean"], reverse=True)
top_k_rows = all_trials_sorted[:TOP_K]

print(f"\n{'='*60}")
print(f"TOP-{TOP_K} configuraciones (combinado Random + Bayes)")
print(f"{'='*60}")
for rank, r in enumerate(top_k_rows, 1):
    print(f"  #{rank}  [{r['method']:6s}] model={r['model']:3s} | F1={r['f1_macro_mean']:.4f}")

overall_winner = top_k_rows[0]
print(f"\n  CONFIG GANADORA: {overall_winner['method'].upper()} | "
      f"model={overall_winner['model']} | F1={overall_winner['f1_macro_mean']:.4f}")

# ── Guardar artefactos ─────────────────────────────────────────────────────────

# CSV Random
rand_csv = OUT_DIR / "semana8_random_trials.csv"
all_keys = ["trial", "method", "model", "f1_macro_mean", "f1_macro_std", "status",
            "rf__n_estimators", "rf__max_depth", "rf__min_samples_leaf",
            "et__n_estimators", "et__max_depth", "et__min_samples_leaf"]
with open(rand_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
    w.writeheader(); w.writerows(rand_trials)

# CSV Bayes
bayes_csv = OUT_DIR / "semana8_bayes_trials.csv"
with open(bayes_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
    w.writeheader(); w.writerows(bayes_trials)

# TXT Resumen
resumen_path = OUT_DIR / "semana8_resumen.txt"
n_pruned = sum(1 for t in bayes_trials if t["status"] == "pruned")
with open(resumen_path, "w") as f:
    f.write("=== Sprint 8 — Búsqueda de Hiperparámetros ===\n")
    f.write(f"Timestamp  : {TIMESTAMP}\n")
    f.write(f"Dataset    : data/Keypoints/pkl  ({n_samples} muestras, {n_classes} clases)\n")
    f.write(f"MD5        : {dataset_md5}\n")
    f.write(f"Features   : pose(33×2) + right_hand(21×2) = {X.shape[1]} dims\n")
    f.write(f"Split      : GroupKFold(n_splits={N_SPLITS}, groups=vineta)\n")
    f.write(f"Seed       : {SEED}\n\n")

    f.write("--- Espacio de búsqueda ---\n")
    f.write(f"  Modelos      : {SPACE['model']}\n")
    f.write(f"  rf__n_est    : int {SPACE['rf__n_estimators'][:2]}\n")
    f.write(f"  rf__depth    : int {SPACE['rf__max_depth'][:2]}\n")
    f.write(f"  rf__leaf     : int {SPACE['rf__min_samples_leaf'][:2]}\n")
    f.write(f"  et__n_est    : int {SPACE['et__n_estimators'][:2]}\n")
    f.write(f"  et__depth    : int {SPACE['et__max_depth'][:2]}\n")
    f.write(f"  et__leaf     : int {SPACE['et__min_samples_leaf'][:2]}\n\n")

    f.write("--- Presupuesto ---\n")
    f.write(f"  Random Search : {N_TRIALS} trials\n")
    f.write(f"  Bayes (TPE)   : {N_TRIALS} trials  "
            f"(MedianPruner n_warmup=3 — early stopping a nivel trial) | Podados: {n_pruned}\n\n")

    f.write(f"--- Random Search — best F1: {rand_best['f1_macro_mean']:.4f} ---\n")
    for k, v in rand_best.items():
        if k not in ("method", "status"):
            f.write(f"    {k}: {v}\n")

    f.write(f"\n--- Bayesian (Optuna TPE) — best F1: {bayes_best['f1_macro_mean']:.4f} ---\n")
    for k, v in bayes_best.items():
        if k not in ("method", "status"):
            f.write(f"    {k}: {v}\n")

    f.write(f"\n--- Top-{TOP_K} global ---\n")
    for rank, r in enumerate(top_k_rows, 1):
        f.write(f"  #{rank}  [{r['method']:6s}] model={r['model']:3s} | "
                f"F1={r['f1_macro_mean']:.4f} ± {r['f1_macro_std']:.4f}\n")

    f.write(f"\n--- CONFIG GANADORA ---\n")
    for k, v in overall_winner.items():
        if k != "status":
            f.write(f"  {k}: {v}\n")

print(f"\nArtefactos guardados:")
print(f"  {rand_csv}")
print(f"  {bayes_csv}")
print(f"  {resumen_path}")

# ── Gráficos ──────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(18, 14))
fig.suptitle(
    "Sprint 8 — HPO: Random Search vs Bayesian Optimization\n"
    f"Dataset: {n_samples} muestras, {n_classes} clases · GroupKFold(n=5) · budget={N_TRIALS} trials/método",
    fontsize=13, fontweight="bold", y=0.99
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.35)

C_RAND  = "#4C9BE8"
C_BAYES = "#E87C4C"
C_WIN   = "#5CB85C"

# ── Plot 1: Evolución best F1 (curva de convergencia) ────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(range(1, N_TRIALS + 1), rand_best_so_far, "o-",
         color=C_RAND,  lw=2, markersize=5, label="Random Search")
ax1.plot(range(1, len(bayes_best_so_far) + 1), bayes_best_so_far, "s-",
         color=C_BAYES, lw=2, markersize=5, label="Bayesian (TPE + MedianPruner)")
ax1.axhline(rand_best["f1_macro_mean"],  color=C_RAND,  linestyle="--", lw=1, alpha=0.6)
ax1.axhline(bayes_best["f1_macro_mean"], color=C_BAYES, linestyle="--", lw=1, alpha=0.6)
ax1.set_xlabel("Trial #")
ax1.set_ylabel("Mejor F1-macro acumulado")
ax1.set_title("Evolución de la mejor configuración\n(convergencia por método)")
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)
ax1.annotate(f"Rand best={rand_best['f1_macro_mean']:.4f}",
             xy=(N_TRIALS, rand_best["f1_macro_mean"]),
             xytext=(-70, 12), textcoords="offset points",
             fontsize=8, color=C_RAND,
             arrowprops=dict(arrowstyle="->", color=C_RAND, lw=0.8))
ax1.annotate(f"Bayes best={bayes_best['f1_macro_mean']:.4f}",
             xy=(len(bayes_best_so_far), bayes_best["f1_macro_mean"]),
             xytext=(-70, -18), textcoords="offset points",
             fontsize=8, color=C_BAYES,
             arrowprops=dict(arrowstyle="->", color=C_BAYES, lw=0.8))

# ── Plot 2: Top-k horizontal bar ──────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
labels_topk = [f"#{i+1} [{r['method'][:4]}] {r['model']}" for i, r in enumerate(top_k_rows)]
vals_topk   = [r["f1_macro_mean"] for r in top_k_rows]
colors_topk = [C_RAND if r["method"] == "random" else C_BAYES for r in top_k_rows]
colors_topk[0] = C_WIN  # ganador en verde
bars = ax2.barh(labels_topk[::-1], vals_topk[::-1],
                color=colors_topk[::-1], edgecolor="white", linewidth=0.8)
for bar, v in zip(bars, vals_topk[::-1]):
    ax2.text(v + 0.0001, bar.get_y() + bar.get_height() / 2,
             f"{v:.4f}", va="center", fontsize=8)
ax2.set_xlabel("F1-macro")
ax2.set_title(f"Top-{TOP_K} configuraciones\n(verde=ganadora)")
ax2.grid(axis="x", alpha=0.3)

# ── Plot 3: Distribución de F1 por método (boxplot) ───────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
rand_scores  = [r["f1_macro_mean"] for r in rand_trials  if r["status"] == "ok"]
bayes_scores = [r["f1_macro_mean"] for r in bayes_trials if r["status"] == "ok"]
bp = ax3.boxplot([rand_scores, bayes_scores],
                 patch_artist=True, widths=0.5,
                 medianprops=dict(color="black", lw=2))
bp["boxes"][0].set_facecolor(C_RAND  + "88")
bp["boxes"][1].set_facecolor(C_BAYES + "88")
ax3.scatter([1] * len(rand_scores),  rand_scores,  alpha=0.6, color=C_RAND,  s=25, zorder=3)
ax3.scatter([2] * len(bayes_scores), bayes_scores, alpha=0.6, color=C_BAYES, s=25, zorder=3)
ax3.set_xticks([1, 2])
ax3.set_xticklabels(["Random\nSearch", "Bayesian\n(TPE)"])
ax3.set_ylabel("F1-macro")
ax3.set_title("Distribución de F1\npor método (trials completos)")
ax3.grid(axis="y", alpha=0.3)
ax3.annotate(f"n_pruned={n_pruned}", xy=(1.5, min(bayes_scores + [0.001])),
             ha="center", fontsize=8, color="gray")

# ── Plot 4: F1 por modelo explorado ───────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
from collections import defaultdict
model_scores = defaultdict(list)
for r in all_trials:
    if r["status"] == "ok":
        model_scores[r["model"]].append(r["f1_macro_mean"])
models_present = sorted(model_scores.keys())
bdata = [model_scores[m] for m in models_present]
bp2 = ax4.boxplot(bdata, patch_artist=True, widths=0.45,
                  medianprops=dict(color="black", lw=2))
mc = ["#1ABC9C", "#E74C3C", "#9B59B6"]
for patch, c in zip(bp2["boxes"], mc):
    patch.set_facecolor(c + "88")
for i, (m, vals) in enumerate(zip(models_present, bdata), 1):
    ax4.scatter([i] * len(vals), vals, alpha=0.6, color=mc[i-1], s=25, zorder=3)
    ax4.text(i, max(vals) + 0.0003, f"n={len(vals)}", ha="center", fontsize=8)
ax4.set_xticks(range(1, len(models_present) + 1))
ax4.set_xticklabels([m.upper() for m in models_present])
ax4.set_ylabel("F1-macro")
ax4.set_title("F1 por tipo de modelo\n(Random + Bayes combinados)")
ax4.grid(axis="y", alpha=0.3)

# ── Plot 5: Comparación presupuesto / trials podados ─────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
methods = ["Random", "Bayesian\n(TPE)"]
completed = [len(rand_scores), len(bayes_scores)]
pruned_    = [0, n_pruned]
x_pos = np.arange(len(methods))
w = 0.35
ax5.bar(x_pos - w/2, completed, width=w, label="Completados", color=[C_RAND, C_BAYES], alpha=0.85)
ax5.bar(x_pos + w/2, pruned_,   width=w, label="Podados",     color="gray", alpha=0.6)
for i, (c, p) in enumerate(zip(completed, pruned_)):
    ax5.text(i - w/2, c + 0.2, str(c), ha="center", fontsize=9)
    if p:
        ax5.text(i + w/2, p + 0.2, str(p), ha="center", fontsize=9)
ax5.set_xticks(x_pos)
ax5.set_xticklabels(methods)
ax5.set_ylabel("Trials")
ax5.set_title(f"Presupuesto: {N_TRIALS} trials/método\nCompletos vs. Podados (Bayes)")
ax5.legend(fontsize=9)
ax5.grid(axis="y", alpha=0.3)

# ── Footer ────────────────────────────────────────────────────────────────────
winner_txt = (
    f"CONFIG GANADORA:  método={overall_winner['method'].upper()}  |  "
    f"modelo={overall_winner['model'].upper()}  |  "
    f"F1-macro={overall_winner['f1_macro_mean']:.4f}  |  "
    f"presupuesto={N_TRIALS*2} trials totales  |  MD5={dataset_md5[:12]}..."
)
fig.text(0.5, 0.005, winner_txt, ha="center", fontsize=9,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#DFF0D8", edgecolor="#3C763D", alpha=0.9))

plot_path = OUT_DIR / "semana8_graficos.png"
plt.savefig(plot_path, dpi=130, bbox_inches="tight")
plt.close()
print(f"  {plot_path}")
print(f"\n[SPRINT 8 — COMPLETADO]")
print(f"  Random best  : {rand_best['f1_macro_mean']:.4f}  (model={rand_best['model']})")
print(f"  Bayes  best  : {bayes_best['f1_macro_mean']:.4f}  (model={bayes_best['model']})")
print(f"  GANADOR      : {overall_winner['method'].upper()} | {overall_winner['model'].upper()} "
      f"| F1={overall_winner['f1_macro_mean']:.4f}")
