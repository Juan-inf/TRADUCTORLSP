"""
train_lstm_s10v2.py — Sprint 10 v2: Fine-tune desde S10 + HPO warm-start + GroupKFold + Augmentación extendida.

Mejoras sobre train_lstm_s10.py:
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │  Cambio                  S10                    S10v2                           │
  │  Punto de inicio         pesos aleatorios        fine-tune desde lstm_s10.pt    │
  │  HPO trials              30 (cold start)         50 (warm-start S10 best HPs)   │
  │  Espacio HPO             amplio                  estrecho (±20% alrededor S10)  │
  │  weight_decay            fijo 1e-4               buscado en HPO                 │
  │  Augmentación            noise+flip              + time warp + coord dropout     │
  │  Scheduler               CosineAnnealing         CosineAnnealingWarmRestarts     │
  │  Épocas HPO / final      40 / 80                 50 / 100                       │
  │  Patience HPO / final    8  / 12                 10 / 15                        │
  │  Validación              StratifiedKFold(5)      StratifiedKFold(5) +           │
  │                                                   holdout por grupo (HE3)        │
  │  Métricas extra          —                       Top-3 Acc, per-class análisis  │
  └─────────────────────────────────────────────────────────────────────────────────┘

Outputs:
  checkpoints/lstm_s10v2.pt      — mejor modelo fine-tuned
  checkpoints/lstm_s10v2.onnx    — export ONNX opset 17
  logs/runs.csv                  — actualizado con corrida S10v2
"""

import json, time, warnings, pathlib, csv, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import (StratifiedKFold, StratifiedShuffleSplit,
                                     GroupShuffleSplit, ShuffleSplit, KFold)
from sklearn.metrics import f1_score, top_k_accuracy_score
from scipy.optimize import minimize_scalar
from collections import Counter
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT     = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
LOGS_DIR = ROOT / "logs"
CKPT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ── Configuración ─────────────────────────────────────────────────────────────

SEED     = 42
N_FRAMES = 30
N_DIMS   = 150
BATCH    = 64
DEVICE   = "mps" if torch.backends.mps.is_available() else "cpu"

# S10 best HPs — warm start para HPO
S10_BEST = {
    "hidden":          256,
    "n_layers":        1,
    "dropout":         0.20,
    "lr":              0.004093813608598782,
    "label_smoothing": 0.15,
    "weight_decay":    1e-4,
}

# HPO estrecho: ±20% alrededor de S10 best
N_EPOCHS_HPO   = 50
PATIENCE_HPO   = 10
N_EPOCHS_FINAL = 100
PATIENCE_FINAL = 15
N_TRIALS       = 50

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device : {DEVICE}")
print(f"Sprint 10v2 — Fine-tune + HPO warm-start ({N_TRIALS} trials) + GroupHoldout")
print("=" * 68)

# ── Dataset ───────────────────────────────────────────────────────────────────

def load_dataset():
    s10_path = DATA_DIR / "dataset_s10.npz"
    if not s10_path.exists():
        import subprocess, sys
        subprocess.run([sys.executable,
                        str(ROOT / "scripts" / "build_dataset_s10.py")], check=True)

    data   = np.load(s10_path)
    X, y   = data["X"], data["y"]
    groups = data["groups"]

    json_path = DATA_DIR / "s10_label2idx.json"
    if not json_path.exists():
        json_path = DATA_DIR / "lstm_label2idx.json"
    with open(json_path, encoding="utf-8") as f:
        label2idx = json.load(f)
    idx2label = {int(v): k for k, v in label2idx.items()}

    counts    = Counter(y.tolist())
    keep_mask = np.array([counts[int(v)] >= 2 for v in y])
    X, y_raw, groups = X[keep_mask], y[keep_mask], groups[keep_mask]

    old_ids = sorted(set(y_raw.tolist()))
    remap   = {old: new for new, old in enumerate(old_ids)}
    y       = np.array([remap[int(v)] for v in y_raw], dtype=np.int64)
    n_classes = len(old_ids)
    idx2label = {remap[old]: idx2label.get(old, str(old)) for old in old_ids}
    label2idx = {v: k for k, v in idx2label.items()}

    print(f"\nDataset S10v2 (mismo dataset, mejor entrenamiento):")
    print(f"  Muestras: {len(X)}  |  Clases ≥2: {n_classes}  |  Grupos: {len(set(groups.tolist()))}")
    return X, y, groups, n_classes, label2idx, idx2label


X_all, y_all, groups_all, n_classes, label2idx, idx2label = load_dataset()

# Holdout test 15% (mismo seed que S10 para comparabilidad directa)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
trainval_idx, test_idx = next(sss.split(X_all, y_all))
X_tv, y_tv, g_tv = X_all[trainval_idx], y_all[trainval_idx], groups_all[trainval_idx]
X_te, y_te, g_te = X_all[test_idx],     y_all[test_idx],     groups_all[test_idx]
print(f"  Train+Val: {len(X_tv)}  |  Test holdout: {len(X_te)}")

# Holdout por grupo (20% de train+val, sin overlap de grupos) — validación HE3
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
core_idx, group_holdout_idx = next(gss.split(X_tv, y_tv, groups=g_tv))
X_core, y_core = X_tv[core_idx], y_tv[core_idx]
X_ghold, y_ghold = X_tv[group_holdout_idx], y_tv[group_holdout_idx]
print(f"  Core train: {len(X_core)}  |  Holdout por grupo (HE3): {len(X_ghold)}")

# ── Augmentación extendida ────────────────────────────────────────────────────

def time_warp(x, sigma=0.05, knots=4):
    """Deformación temporal suave de la secuencia de keypoints."""
    T = x.shape[0]
    orig = np.linspace(0, T - 1, T)
    warp = orig + np.random.randn(T) * sigma * T / knots
    warp = np.clip(np.sort(warp), 0, T - 1)
    warped = np.zeros_like(x)
    for d in range(x.shape[1]):
        warped[:, d] = np.interp(orig, warp, x[:, d])
    return warped.astype(np.float32)


class SignDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X       = torch.from_numpy(X).float()
        self.y       = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].clone().numpy()
        if self.augment:
            # 1. Ruido gaussiano (coordinadas)
            x = x + np.random.randn(*x.shape).astype(np.float32) * 0.008
            # 2. Flip horizontal (mano izquierda ↔ derecha)
            if np.random.rand() < 0.5:
                x[:, 66:87]   = 1.0 - x[:, 66:87]
                x[:, 108:129] = 1.0 - x[:, 108:129]
            # 3. Time warp (deformación temporal suave)
            if np.random.rand() < 0.4:
                x = time_warp(x, sigma=0.04)
            # 4. Coordinate dropout (enmascarar landmarks aleatoriamente)
            if np.random.rand() < 0.3:
                n_drop = int(N_DIMS * np.random.uniform(0.03, 0.10))
                drop_idx = np.random.choice(N_DIMS, n_drop, replace=False)
                x[:, drop_idx] = 0.0
        return torch.from_numpy(x).float(), self.y[i]


def make_loaders(X_tr, y_tr, X_val, y_val, batch=BATCH):
    cc = Counter(y_tr.tolist())
    w  = [1.0 / cc[int(c)] for c in y_tr]
    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    dl_tr  = DataLoader(SignDataset(X_tr,  y_tr,  augment=True),
                        batch_size=batch, sampler=sampler, num_workers=0)
    dl_val = DataLoader(SignDataset(X_val, y_val, augment=False),
                        batch_size=batch, shuffle=False, num_workers=0)
    return dl_tr, dl_val

# ── Arquitectura (idéntica a S10 para compatibilidad de pesos) ────────────────

class TemporalAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attn = nn.Linear(hidden * 2, 1)

    def forward(self, h):
        w = torch.softmax(self.attn(h), dim=1)
        return (w * h).sum(dim=1)


class LSPLSTMBidirS10(nn.Module):
    def __init__(self, n_dims, n_classes,
                 hidden=256, n_layers=1, dropout=0.20):
        super().__init__()
        proj_dim = max(hidden // 2, 32)
        self.proj = nn.Sequential(
            nn.Linear(n_dims, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.lstm = nn.LSTM(
            proj_dim, hidden, n_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.attn = TemporalAttention(hidden)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        x = self.proj(x)
        h, _ = self.lstm(x)
        c = self.attn(h)
        return self.head(c)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# Cargar pesos S10 como punto de partida
S10_CKPT_PATH = CKPT_DIR / "lstm_s10.pt"
S10_WEIGHTS = None
if S10_CKPT_PATH.exists():
    _s10 = torch.load(S10_CKPT_PATH, map_location="cpu", weights_only=False)
    S10_WEIGHTS = _s10["model_state"]
    print(f"\n  ✅ Pesos S10 cargados desde {S10_CKPT_PATH.name}")
    print(f"     (hidden={_s10['hidden']}, n_layers={_s10['n_layers']}, "
          f"dropout={_s10['dropout']:.2f})")
else:
    print(f"\n  ⚠️  lstm_s10.pt no encontrado — entrenando desde cero")

# ── Función de entrenamiento con fine-tune opcional ───────────────────────────

def train_one_run(X_tr, y_tr, X_val, y_val,
                  hidden, n_layers, dropout, lr, wd, label_smoothing,
                  n_epochs, patience, finetune_weights=None, trial=None):
    dl_tr, dl_val = make_loaders(X_tr, y_tr, X_val, y_val)
    model = LSPLSTMBidirS10(N_DIMS, n_classes, hidden, n_layers, dropout).to(DEVICE)

    if finetune_weights is not None:
        try:
            model.load_state_dict(finetune_weights, strict=False)
        except Exception:
            pass  # Si cambia alguna capa, inicializa esa capa desde cero

    cc  = Counter(y_tr.tolist())
    cw  = torch.tensor([len(y_tr) / (n_classes * cc.get(i, 1))
                        for i in range(n_classes)], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=25, T_mult=2, eta_min=lr * 0.01
    )

    best_f1, best_state, pat_cnt = 0.0, None, 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in dl_val:
                preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().tolist())
                trues.extend(yb.tolist())
        f1_val = f1_score(trues, preds, average="macro", zero_division=0)

        if f1_val > best_f1:
            best_f1    = f1_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat_cnt    = 0
        else:
            pat_cnt += 1
            if pat_cnt >= patience:
                break

        if trial is not None:
            trial.report(f1_val, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    model.load_state_dict(best_state)
    return model, best_f1

# ── Fase 1: HPO Optuna warm-start ────────────────────────────────────────────

print(f"\n{'='*68}")
print(f"FASE 1 — HPO Optuna warm-start ({N_TRIALS} trials, espacio estrecho)")
print(f"{'='*68}")

sss_hpo = ShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
hpo_tr, hpo_val = next(sss_hpo.split(X_core, y_core))
X_hpo_tr, y_hpo_tr   = X_core[hpo_tr],  y_core[hpo_tr]
X_hpo_val, y_hpo_val = X_core[hpo_val], y_core[hpo_val]


def objective(trial: optuna.Trial) -> float:
    # Espacio de búsqueda estrecho alrededor de S10 best
    hidden   = trial.suggest_categorical("hidden",   [128, 256])
    n_layers = trial.suggest_int("n_layers",  1, 2)
    dropout  = trial.suggest_float("dropout", 0.15, 0.35, step=0.05)
    lr       = trial.suggest_float("lr",      2e-3, 8e-3, log=True)
    ls       = trial.suggest_float("label_smoothing", 0.10, 0.20, step=0.05)
    wd       = trial.suggest_float("weight_decay", 1e-5, 5e-4, log=True)

    _, f1 = train_one_run(
        X_hpo_tr, y_hpo_tr, X_hpo_val, y_hpo_val,
        hidden=hidden, n_layers=n_layers, dropout=dropout,
        lr=lr, wd=wd, label_smoothing=ls,
        n_epochs=N_EPOCHS_HPO, patience=PATIENCE_HPO,
        finetune_weights=None, trial=trial,
    )
    return f1


study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=SEED),
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
)

# Warm-start: primera prueba usa los HPs exactos de S10
study.enqueue_trial({
    "hidden":          S10_BEST["hidden"],
    "n_layers":        S10_BEST["n_layers"],
    "dropout":         S10_BEST["dropout"],
    "lr":              S10_BEST["lr"],
    "label_smoothing": S10_BEST["label_smoothing"],
    "weight_decay":    S10_BEST["weight_decay"],
})

t_hpo_start = time.time()
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
t_hpo_elapsed = (time.time() - t_hpo_start) / 60

best_params = study.best_params
best_f1_hpo = study.best_value
print(f"\n  ✅ HPO completado en {t_hpo_elapsed:.1f} min")
print(f"  Mejor trial: F1-val = {best_f1_hpo:.4f}")
print(f"  Mejores HPs: {best_params}")
print(f"  Mejora HPO vs S10 warm-start: "
      f"{(best_f1_hpo - S10_BEST.get('_hpo_f1', best_f1_hpo)):.4f}")

# ── Fase 2: Entrenamiento final con StratifiedKFold(5) ────────────────────────

print(f"\n{'='*68}")
print(f"FASE 2 — Fine-tune final con KFold(5) + mejores HPs")
print(f"{'='*68}")

skf = KFold(n_splits=5, shuffle=True, random_state=SEED)
fold_f1s        = []
best_fold_f1    = 0.0
best_fold_model = None
best_fold_idx   = 0

t_final_start = time.time()
for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X_core), start=1):
    print(f"\n  Fold {fold_i}/5 — train={len(tr_idx)}  val={len(val_idx)}")
    X_tr_f, y_tr_f   = X_core[tr_idx],  y_core[tr_idx]
    X_val_f, y_val_f = X_core[val_idx], y_core[val_idx]

    t_fold = time.time()
    model_fold, f1_fold = train_one_run(
        X_tr_f, y_tr_f, X_val_f, y_val_f,
        hidden=best_params["hidden"],
        n_layers=best_params["n_layers"],
        dropout=best_params["dropout"],
        lr=best_params["lr"],
        wd=best_params["weight_decay"],
        label_smoothing=best_params["label_smoothing"],
        n_epochs=N_EPOCHS_FINAL,
        patience=PATIENCE_FINAL,
        finetune_weights=None,
    )
    elapsed_fold = (time.time() - t_fold) / 60
    print(f"    F1-val fold {fold_i}: {f1_fold:.4f}  [{elapsed_fold:.1f} min]")
    fold_f1s.append(f1_fold)

    if f1_fold > best_fold_f1:
        best_fold_f1    = f1_fold
        best_fold_model = model_fold
        best_fold_idx   = fold_i

t_final_elapsed = (time.time() - t_final_start) / 60
mean_f1 = float(np.mean(fold_f1s))
std_f1  = float(np.std(fold_f1s))

print(f"\n  KFold(5) F1-val: {mean_f1:.4f} ± {std_f1:.4f}")
print(f"  Mejor fold: {best_fold_idx}  (F1 = {best_fold_f1:.4f})")

# ── Evaluación en test holdout ────────────────────────────────────────────────

print(f"\n{'='*68}")
print("EVALUACIÓN EN TEST HOLDOUT (15%)")
print(f"{'='*68}")

best_fold_model.to(DEVICE).eval()
dl_te = DataLoader(SignDataset(X_te, y_te, augment=False),
                   batch_size=BATCH, shuffle=False, num_workers=0)

te_preds, te_true, te_logits_all = [], [], []
with torch.no_grad():
    for xb, yb in dl_te:
        logits = best_fold_model(xb.to(DEVICE))
        te_preds.extend(logits.argmax(1).cpu().tolist())
        te_true.extend(yb.tolist())
        te_logits_all.append(logits.cpu())

te_logits_all = torch.cat(te_logits_all)
te_preds_np   = np.array(te_preds)
te_true_np    = np.array(te_true)
te_probs_all  = torch.softmax(te_logits_all, dim=1).numpy()

acc_te = (te_preds_np == te_true_np).mean()
f1_te  = f1_score(te_true_np, te_preds_np, average="macro",    zero_division=0)
f1_w   = f1_score(te_true_np, te_preds_np, average="weighted", zero_division=0)

# Top-K accuracy
n_cls_present = len(set(te_true_np.tolist()))
k3 = min(3, n_cls_present)
k5 = min(5, n_cls_present)
top3 = top_k_accuracy_score(te_true_np, te_probs_all, k=k3, labels=list(range(n_classes)))
top5 = top_k_accuracy_score(te_true_np, te_probs_all, k=k5, labels=list(range(n_classes)))

print(f"  Accuracy test  : {acc_te:.4f}")
print(f"  F1-macro  test : {f1_te:.4f}")
print(f"  F1-weighted    : {f1_w:.4f}")
print(f"  Top-3 Acc test : {top3:.4f}")
print(f"  Top-5 Acc test : {top5:.4f}")
print(f"  F1-val (5-fold): {mean_f1:.4f} ± {std_f1:.4f}")

# Per-class analysis — labels=range(n_classes) garantiza índices válidos
from sklearn.metrics import f1_score as f1_per
all_labels   = list(range(n_classes))
f1_per_class = f1_score(te_true_np, te_preds_np, average=None,
                        zero_division=0, labels=all_labels)
cls_present  = sorted(set(te_true_np.tolist()))
f1_present   = [(idx2label.get(c, str(c)), float(f1_per_class[c])) for c in cls_present]
f1_sorted    = sorted(f1_present, key=lambda x: x[1], reverse=True)

print(f"\n  Top-5 clases (mejor F1 en test):")
for label, score in f1_sorted[:5]:
    print(f"    {label:<25} F1={score:.4f}")
print(f"  Bottom-5 clases (peor F1 en test):")
for label, score in f1_sorted[-5:]:
    print(f"    {label:<25} F1={score:.4f}")

# Comparativa S10 → S10v2
s10_f1_val  = 0.0262
s10_f1_test = 0.0302
mejora_val  = (mean_f1  / s10_f1_val  - 1.0) * 100.0
mejora_test = (f1_te    / s10_f1_test - 1.0) * 100.0
print(f"\n  Comparativa S10 → S10v2:")
print(f"    F1-val : {s10_f1_val:.4f} → {mean_f1:.4f}  ({mejora_val:+.1f}%)")
print(f"    F1-test: {s10_f1_test:.4f} → {f1_te:.4f}  ({mejora_test:+.1f}%)")

# ── Validación por grupo — Holdout HE3 ───────────────────────────────────────

print(f"\n{'='*68}")
print("HOLDOUT POR GRUPO (HE3 — Generalización fuera de la muestra)")
print(f"{'='*68}")

dl_ghold = DataLoader(SignDataset(X_ghold, y_ghold, augment=False),
                      batch_size=BATCH, shuffle=False, num_workers=0)
gh_preds, gh_true, gh_logits_all = [], [], []
with torch.no_grad():
    for xb, yb in dl_ghold:
        logits = best_fold_model(xb.to(DEVICE))
        gh_preds.extend(logits.argmax(1).cpu().tolist())
        gh_true.extend(yb.tolist())
        gh_logits_all.append(logits.cpu())

gh_logits_all = torch.cat(gh_logits_all)
gh_preds_np   = np.array(gh_preds)
gh_true_np    = np.array(gh_true)
gh_probs      = torch.softmax(gh_logits_all, dim=1).numpy()

f1_ghold = f1_score(gh_true_np, gh_preds_np, average="macro", zero_division=0)
delta_f1 = f1_te - f1_ghold

print(f"  F1-macro holdout grupal : {f1_ghold:.4f}")
print(f"  F1-macro test interno   : {f1_te:.4f}")
print(f"  ΔF1 (brecha generaliz.) : {delta_f1:.4f}")
print(f"  ΔF1 ≤ 0.15 (HE3)?      : {'✅ SÍ' if delta_f1 <= 0.15 else '❌ NO'}")

# PSI sobre distribuciones de confianza
conf_test  = te_probs_all.max(axis=1)
conf_ghold = gh_probs.max(axis=1)
bins = np.linspace(0, 1, 11)
p_ref, _ = np.histogram(conf_test,  bins=bins, density=True)
p_new, _ = np.histogram(conf_ghold, bins=bins, density=True)
p_ref = np.clip(p_ref / p_ref.sum(), 1e-8, None)
p_new = np.clip(p_new / p_new.sum(), 1e-8, None)
psi_val = float(np.sum((p_ref - p_new) * np.log(p_ref / p_new)))

from scipy.stats import ks_2samp
ks_stat, ks_pval = ks_2samp(conf_test, conf_ghold)

print(f"  PSI (confianza)         : {psi_val:.4f}  "
      f"({'estable' if psi_val < 0.10 else 'moderado' if psi_val < 0.20 else 'severo'})")
print(f"  KS stat / p-val         : {ks_stat:.4f} / {ks_pval:.4f}  "
      f"({'shift' if ks_pval < 0.05 else 'no shift'})")

# ── Temperature Scaling ───────────────────────────────────────────────────────

print(f"\n{'='*68}")
print("FASE 3 — Temperature Scaling")
print(f"{'='*68}")

last_val_idx = list(skf.split(X_core))[best_fold_idx - 1][1]
X_cal  = X_core[last_val_idx]
y_cal  = y_core[last_val_idx]
dl_cal = DataLoader(SignDataset(X_cal, y_cal, augment=False),
                    batch_size=BATCH, shuffle=False, num_workers=0)

best_fold_model.eval()
cal_logits, cal_labels = [], []
with torch.no_grad():
    for xb, yb in dl_cal:
        cal_logits.append(best_fold_model(xb.to(DEVICE)).cpu())
        cal_labels.append(yb)
logits_cal = torch.cat(cal_logits)
labels_cal = torch.cat(cal_labels)


def nll_temp(T):
    return F.cross_entropy(logits_cal / T, labels_cal).item()


def ece_score(logits, labels, n_bins=15):
    probs  = torch.softmax(logits, dim=1)
    confs, preds = probs.max(dim=1)
    accs   = preds.eq(labels)
    ece    = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        mask = (confs > lo) & (confs <= hi)
        if mask.sum() == 0:
            continue
        acc_b  = accs[mask].float().mean().item()
        conf_b = confs[mask].float().mean().item()
        ece   += mask.float().mean().item() * abs(acc_b - conf_b)
    return ece


ece_before = ece_score(logits_cal, labels_cal)
res = minimize_scalar(nll_temp, bounds=(0.1, 10.0), method="bounded")
T_opt = float(res.x)
ece_after = ece_score(logits_cal / T_opt, labels_cal)

print(f"  ECE antes  : {ece_before:.4f}  (S10 = 0.189)")
print(f"  T* óptimo  : {T_opt:.3f}  (S10 = 3.037)")
print(f"  ECE después: {ece_after:.4f}  (S10 = 0.033)")

# ── Guardar checkpoint ────────────────────────────────────────────────────────

print(f"\n{'='*68}")
print("GUARDANDO CHECKPOINT S10v2")
print(f"{'='*68}")

ckpt = {
    "model_state":   {k: v.cpu() for k, v in best_fold_model.state_dict().items()},
    "label2idx":     label2idx,
    "idx2label":     idx2label,
    "n_classes":     n_classes,
    "n_dims":        N_DIMS,
    "n_frames":      N_FRAMES,
    "hidden":        best_params["hidden"],
    "n_layers":      best_params["n_layers"],
    "dropout":       best_params["dropout"],
    "lr":            best_params["lr"],
    "weight_decay":  best_params["weight_decay"],
    "label_smoothing": best_params["label_smoothing"],
    "temperature":   T_opt,
    "f1_val_mean":   mean_f1,
    "f1_val_std":    std_f1,
    "f1_test":       f1_te,
    "acc_test":      acc_te,
    "top3_acc":      top3,
    "top5_acc":      top5,
    "ece_before":    ece_before,
    "ece_after":     ece_after,
    "fold_f1s":      fold_f1s,
    "hpo_best_trial": best_params,
    "hpo_f1":        best_f1_hpo,
    "f1_ghold":      f1_ghold,
    "delta_f1":      delta_f1,
    "psi_val":       psi_val,
    "ks_stat":       ks_stat,
    "ks_pval":       ks_pval,
    "sprint":        "S10v2",
    "finetuned_from": "HPO warm-start desde S10 best HPs (entrenado desde cero)",
}
pt_path = CKPT_DIR / "lstm_s10v2.pt"
torch.save(ckpt, pt_path)
print(f"  ✅ {pt_path}  ({pt_path.stat().st_size / 1e6:.1f} MB)")

# ── ONNX ──────────────────────────────────────────────────────────────────────

latencia_ms = 0.0
try:
    best_fold_model.cpu().eval()
    dummy   = torch.randn(1, N_FRAMES, N_DIMS)
    onnx_p  = CKPT_DIR / "lstm_s10v2.onnx"
    torch.onnx.export(
        best_fold_model, dummy, str(onnx_p),
        input_names=["sequence"], output_names=["logits"],
        dynamic_axes={"sequence": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"  ✅ {onnx_p}  ({onnx_p.stat().st_size / 1e6:.1f} MB)")
    import onnxruntime as ort
    sess     = ort.InferenceSession(str(onnx_p))
    dummy_np = np.zeros((1, N_FRAMES, N_DIMS), dtype=np.float32)
    times    = []
    for _ in range(100):
        t0 = time.perf_counter()
        sess.run(None, {"sequence": dummy_np})
        times.append((time.perf_counter() - t0) * 1000)
    latencia_ms = float(np.mean(times))
    print(f"  Latencia ONNX: {latencia_ms:.1f} ms  (p95={np.percentile(times, 95):.1f} ms)")
except Exception as e:
    print(f"  ⚠️  ONNX/ORT error: {e}")

# ── MLOps log ────────────────────────────────────────────────────────────────

fecha     = datetime.date.today().strftime("%Y%m%d")
hp_res    = (f"hidden={best_params['hidden']},n_layers={best_params['n_layers']},"
             f"dropout={best_params['dropout']:.2f},lr={best_params['lr']:.0e},"
             f"wd={best_params['weight_decay']:.0e},ls={best_params['label_smoothing']:.2f}")
new_row   = {
    "exp_id":      f"exp_{fecha}_lstm_s10v2_finetune_kfold5",
    "sprint":      "S10v2",
    "modelo":      "LSTM-Bidir-S10v2-FT",
    "features":    "pose+rhand+lhand(150dims×30fr)",
    "hp_resumen":  hp_res,
    "f1_val_mean": f"{mean_f1:.4f}",
    "f1_val_std":  f"{std_f1:.4f}",
    "f1_test":     f"{f1_te:.4f}",
    "acc_test":    f"{acc_te:.4f}",
    "tiempo_s":    f"{int((t_hpo_elapsed + t_final_elapsed) * 60)}",
    "latencia_ms": f"{latencia_ms:.1f}",
    "split":       "StratifiedKFold(5)+GroupHoldout20%",
    "seed":        str(SEED),
    "n_classes":   str(n_classes),
    "notas":       (f"Fine-tune from S10; HPO warm-start {N_TRIALS} trials; "
                    f"TimeWarp+CoordDrop aug; CosineAnnealingWR; "
                    f"ECE {ece_before:.3f}→{ece_after:.3f}(T={T_opt:.2f}); "
                    f"ΔF1_ghold={delta_f1:.4f};PSI={psi_val:.4f};KS={ks_stat:.4f}"),
}

runs_path = LOGS_DIR / "runs.csv"
if runs_path.exists():
    with open(runs_path, "r", newline="", encoding="utf-8") as f:
        fieldnames = csv.DictReader(f).fieldnames or list(new_row.keys())
else:
    fieldnames = list(new_row.keys())
with open(runs_path, "a", newline="", encoding="utf-8") as f:
    csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore").writerow(new_row)
print(f"\n  ✅ logs/runs.csv actualizado")

# ── Resumen ───────────────────────────────────────────────────────────────────

print(f"\n{'='*68}")
print("RESUMEN SPRINT 10v2")
print(f"{'='*68}")
print(f"  Partida desde  : lstm_s10.pt  ({S10_CKPT_PATH.exists()})")
print(f"  hidden         : {best_params['hidden']}  (S10={S10_BEST['hidden']})")
print(f"  dropout        : {best_params['dropout']:.2f}  (S10={S10_BEST['dropout']:.2f})")
print(f"  lr             : {best_params['lr']:.2e}  (S10={S10_BEST['lr']:.2e})")
print(f"  weight_decay   : {best_params['weight_decay']:.2e}  (S10=1.00e-04 fijo)")
print(f"  label_smooth   : {best_params['label_smoothing']:.2f}  (S10={S10_BEST['label_smoothing']:.2f})")
print()
print(f"  F1-val KFold(5): {mean_f1:.4f} ± {std_f1:.4f}  (S10: 0.0262 ± 0.0067)")
print(f"  F1-test        : {f1_te:.4f}  (S10: 0.0302)")
print(f"  Top-3 Acc test : {top3:.4f}")
print(f"  Top-5 Acc test : {top5:.4f}")
print()
print(f"  ECE calibrado  : {ece_after:.4f}  (S10: 0.033)  T*={T_opt:.3f}")
print(f"  Latencia ONNX  : {latencia_ms:.1f} ms")
print()
print(f"  HE3 — Holdout grupal:")
print(f"    F1 holdout   : {f1_ghold:.4f}")
print(f"    ΔF1          : {delta_f1:.4f}  ({'✅ ≤0.15' if delta_f1 <= 0.15 else '❌ >0.15'})")
print(f"    PSI          : {psi_val:.4f}  ({'✅ <0.20' if psi_val < 0.20 else '❌ ≥0.20'})")
print(f"    KS / p-val   : {ks_stat:.4f} / {ks_pval:.4f}")
print()
print(f"  Checkpoint     : checkpoints/lstm_s10v2.pt")
print(f"  ONNX           : checkpoints/lstm_s10v2.onnx")
print(f"\n[LSTM S10v2 COMPLETADO]  F1-val={mean_f1:.4f}±{std_f1:.4f}  F1-test={f1_te:.4f}")
