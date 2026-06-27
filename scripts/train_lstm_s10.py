"""
train_lstm_s10.py — Sprint 10: LSTM Bidir REDUCIDO + HPO Optuna + KFold(5) + Temperature Scaling.

Mejoras vs S9 (train_lstm_signs.py):
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Cambio               S9              S10                               │
  │  hidden               256             128  (-50% params)                │
  │  dropout              0.35            0.50  (+43% regularización)       │
  │  label_smoothing      0.0             0.10  (nuevo: ECE 0.427 → <0.20) │
  │  validación           SSS(70/15/15)   StratifiedKFold(5) (CV estable)  │
  │  HPO                  ninguno         Optuna 30 trials TPE              │
  │  calibración          ninguna         Temperature Scaling               │
  └─────────────────────────────────────────────────────────────────────────┘

Outputs:
  checkpoints/lstm_s10.pt      — mejor modelo (state + HPs + historial)
  checkpoints/lstm_s10.onnx    — export ONNX opset 17
  logs/runs.csv                — actualizado con corrida S10
"""

import json, time, warnings, pathlib, csv, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import f1_score
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

# ── Hiperparámetros ───────────────────────────────────────────────────────────

SEED     = 42
N_FRAMES = 30
N_DIMS   = 150
BATCH    = 64
DEVICE   = "mps" if torch.backends.mps.is_available() else "cpu"

# Defaults S10 (mejores por diseño antes del HPO)
HIDDEN_DEF  = 128
NLAYERS_DEF = 2
DROP_DEF    = 0.50
LR_DEF      = 1e-3
WD_DEF      = 1e-4
LS_DEF      = 0.10  # label_smoothing

# Entrenamiento
N_EPOCHS_HPO   = 40   # épocas máx por trial HPO (rápido)
PATIENCE_HPO   = 8
N_EPOCHS_FINAL = 80   # épocas finales (mismo que S9)
PATIENCE_FINAL = 12
N_TRIALS       = 30   # trials Optuna
N_FOLDS        = 5    # StratifiedKFold CV final

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device: {DEVICE}")
print(f"Sprint 10 — LSTM Bidir Reducido + HPO Optuna {N_TRIALS} trials + KFold({N_FOLDS})")
print("=" * 65)

# ── Cargar dataset S10 ────────────────────────────────────────────────────────

def load_dataset():
    s10_path = DATA_DIR / "dataset_s10.npz"
    if not s10_path.exists():
        print("  dataset_s10.npz no existe → construyendo con build_dataset_s10.py...")
        import subprocess, sys
        subprocess.run([sys.executable,
                        str(ROOT / "scripts" / "build_dataset_s10.py")],
                       check=True)

    data   = np.load(s10_path)
    X, y   = data["X"], data["y"]
    groups = data["groups"]

    json_path = DATA_DIR / "s10_label2idx.json"
    if not json_path.exists():
        json_path = DATA_DIR / "lstm_label2idx.json"

    with open(json_path, encoding="utf-8") as f:
        label2idx = json.load(f)
    idx2label = {int(v): k for k, v in label2idx.items()}

    # Filtrar clases con < 2 muestras y re-mapear
    counts    = Counter(y.tolist())
    keep_mask = np.array([counts[int(v)] >= 2 for v in y])
    X, y_raw, groups = X[keep_mask], y[keep_mask], groups[keep_mask]

    old_ids   = sorted(set(y_raw.tolist()))
    remap     = {old: new for new, old in enumerate(old_ids)}
    y         = np.array([remap[int(v)] for v in y_raw], dtype=np.int64)
    n_classes = len(old_ids)
    idx2label = {remap[old]: idx2label.get(old, str(old)) for old in old_ids}
    label2idx = {v: k for k, v in idx2label.items()}

    print(f"\nDataset S10:")
    print(f"  Muestras: {len(X)}  |  Clases ≥2: {n_classes}")
    return X, y, groups, n_classes, label2idx, idx2label


X_all, y_all, groups_all, n_classes, label2idx, idx2label = load_dataset()

# Holdout test fijo 15% (mismo seed que S9 para comparabilidad)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
trainval_idx, test_idx = next(sss.split(X_all, y_all))
X_tv, y_tv = X_all[trainval_idx], y_all[trainval_idx]
X_te, y_te = X_all[test_idx],     y_all[test_idx]
print(f"  Train+Val: {len(X_tv)}  |  Test holdout: {len(X_te)}")

# ── Dataset PyTorch ───────────────────────────────────────────────────────────

class SignDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X       = torch.from_numpy(X).float()
        self.y       = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].clone()
        if self.augment:
            x = x + torch.randn_like(x) * 0.008
            if torch.rand(1) < 0.5:
                x[:, 66:87]   = 1.0 - x[:, 66:87]
                x[:, 108:129] = 1.0 - x[:, 108:129]
        return x, self.y[i]


def make_loaders(X_tr, y_tr, X_val, y_val, batch=BATCH):
    cc = Counter(y_tr.tolist())
    w  = [1.0 / cc[int(c)] for c in y_tr]
    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    dl_tr  = DataLoader(SignDataset(X_tr,  y_tr,  augment=True),
                        batch_size=batch, sampler=sampler, num_workers=0)
    dl_val = DataLoader(SignDataset(X_val, y_val, augment=False),
                        batch_size=batch, shuffle=False,  num_workers=0)
    return dl_tr, dl_val

# ── Modelo ────────────────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attn = nn.Linear(hidden * 2, 1)

    def forward(self, h):
        w = torch.softmax(self.attn(h), dim=1)
        return (w * h).sum(dim=1)


class LSPLSTMBidirS10(nn.Module):
    """LSTM Bidir reducido para Sprint 10."""

    def __init__(self, n_dims, n_classes,
                 hidden=HIDDEN_DEF, n_layers=NLAYERS_DEF, dropout=DROP_DEF):
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

    def get_logits(self, x):
        return self.forward(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ── Función de entrenamiento ──────────────────────────────────────────────────

def train_one_run(X_tr, y_tr, X_val, y_val, hidden, n_layers, dropout,
                  lr, wd, label_smoothing, n_epochs, patience, trial=None):
    dl_tr, dl_val = make_loaders(X_tr, y_tr, X_val, y_val)

    model = LSPLSTMBidirS10(N_DIMS, n_classes, hidden, n_layers, dropout).to(DEVICE)
    cc    = Counter(y_tr.tolist())
    cw    = torch.tensor([len(y_tr) / (n_classes * cc.get(i, 1))
                          for i in range(n_classes)], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

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
            best_f1   = f1_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat_cnt   = 0
        else:
            pat_cnt += 1
            if pat_cnt >= patience:
                break

        # Optuna pruning
        if trial is not None:
            trial.report(f1_val, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    model.load_state_dict(best_state)
    return model, best_f1

# ── Fase 1: HPO Optuna ────────────────────────────────────────────────────────

print(f"\n{'='*65}")
print(f"FASE 1 — HPO Optuna ({N_TRIALS} trials TPE + MedianPruner)")
print(f"{'='*65}")

# Split interno para HPO: 70/30 sobre train+val (rápido, no CV)
sss_hpo = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
hpo_tr, hpo_val = next(sss_hpo.split(X_tv, y_tv))
X_hpo_tr, y_hpo_tr  = X_tv[hpo_tr],  y_tv[hpo_tr]
X_hpo_val, y_hpo_val = X_tv[hpo_val], y_tv[hpo_val]


def objective(trial: optuna.Trial) -> float:
    hidden   = trial.suggest_categorical("hidden",   [64, 128, 256])
    n_layers = trial.suggest_int("n_layers",  1, 3)
    dropout  = trial.suggest_float("dropout", 0.20, 0.60, step=0.05)
    lr       = trial.suggest_float("lr",      1e-4, 5e-3, log=True)
    ls       = trial.suggest_float("label_smoothing", 0.0, 0.15, step=0.05)

    _, f1 = train_one_run(
        X_hpo_tr, y_hpo_tr, X_hpo_val, y_hpo_val,
        hidden=hidden, n_layers=n_layers, dropout=dropout,
        lr=lr, wd=WD_DEF, label_smoothing=ls,
        n_epochs=N_EPOCHS_HPO, patience=PATIENCE_HPO, trial=trial,
    )
    return f1


t_hpo_start = time.time()
study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=SEED),
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
t_hpo_elapsed = (time.time() - t_hpo_start) / 60

best_params = study.best_params
best_f1_hpo = study.best_value
print(f"\n  ✅ HPO completado en {t_hpo_elapsed:.1f} min")
print(f"  Mejor trial: F1-val = {best_f1_hpo:.4f}")
print(f"  Mejores HPs: {best_params}")

# ── Fase 2: Entrenamiento final con StratifiedKFold(5) ────────────────────────

print(f"\n{'='*65}")
print(f"FASE 2 — Entrenamiento final con KFold({N_FOLDS}) + mejores HPs")
print(f"{'='*65}")
print(f"  hidden={best_params['hidden']}  n_layers={best_params['n_layers']}  "
      f"dropout={best_params['dropout']:.2f}")
print(f"  lr={best_params['lr']:.2e}  label_smoothing={best_params['label_smoothing']:.2f}")

_model_tmp = LSPLSTMBidirS10(N_DIMS, n_classes,
                              best_params['hidden'], best_params['n_layers'],
                              best_params['dropout'])
print(f"  Parámetros del modelo: {count_params(_model_tmp):,}")
print(f"  Ratio params/datos  : {count_params(_model_tmp) / len(X_tv):.0f}:1")
del _model_tmp

skf  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_f1s = []
best_fold_f1   = 0.0
best_fold_model = None
best_fold_idx   = 0
hist_all = []

t_final_start = time.time()
for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X_tv, y_tv), start=1):
    print(f"\n  Fold {fold_i}/{N_FOLDS} — train={len(tr_idx)}  val={len(val_idx)}")
    X_tr_f, y_tr_f = X_tv[tr_idx], y_tv[tr_idx]
    X_val_f, y_val_f = X_tv[val_idx], y_tv[val_idx]

    t_fold = time.time()
    model_fold, f1_fold = train_one_run(
        X_tr_f, y_tr_f, X_val_f, y_val_f,
        hidden=best_params["hidden"],
        n_layers=best_params["n_layers"],
        dropout=best_params["dropout"],
        lr=best_params["lr"],
        wd=WD_DEF,
        label_smoothing=best_params["label_smoothing"],
        n_epochs=N_EPOCHS_FINAL,
        patience=PATIENCE_FINAL,
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

print(f"\n  KFold({N_FOLDS}) F1-val: {mean_f1:.4f} ± {std_f1:.4f}")
print(f"  Mejor fold: {best_fold_idx}  (F1 = {best_fold_f1:.4f})")
print(f"  Tiempo total: {t_final_elapsed:.1f} min")

# ── Evaluación en test holdout ────────────────────────────────────────────────

print(f"\n{'='*65}")
print("EVALUACIÓN EN TEST HOLDOUT (15%)")
print(f"{'='*65}")

best_fold_model.to(DEVICE).eval()
dl_te = DataLoader(SignDataset(X_te, y_te, augment=False),
                   batch_size=BATCH, shuffle=False, num_workers=0)
te_preds, te_true = [], []
with torch.no_grad():
    for xb, yb in dl_te:
        te_preds.extend(best_fold_model(xb.to(DEVICE)).argmax(1).cpu().tolist())
        te_true.extend(yb.tolist())

acc_te = (np.array(te_preds) == np.array(te_true)).mean()
f1_te  = f1_score(te_true, te_preds, average="macro",    zero_division=0)
f1_w   = f1_score(te_true, te_preds, average="weighted", zero_division=0)

# Gap de overfitting
f1_tr_best = max(fold_f1s)
gap = (f1_tr_best - mean_f1) / f1_tr_best if f1_tr_best > 0 else 1.0
s9_f1_val = 0.0365
mejora = (mean_f1 / s9_f1_val - 1.0) * 100.0

print(f"  Accuracy  test : {acc_te:.4f}")
print(f"  F1-macro  test : {f1_te:.4f}")
print(f"  F1-weighted    : {f1_w:.4f}")
print(f"  F1-val (5-fold): {mean_f1:.4f} ± {std_f1:.4f}")
print(f"  Mejora vs S9   : {mejora:+.1f}%  (S9={s9_f1_val})")

# ── Fase 3: Temperature Scaling ───────────────────────────────────────────────

print(f"\n{'='*65}")
print("FASE 3 — Temperature Scaling (calibración ECE)")
print(f"{'='*65}")

# Recolectar logits del val del último fold para calibración
last_fold_val_idx = list(skf.split(X_tv, y_tv))[best_fold_idx - 1][1]
X_cal = X_tv[last_fold_val_idx]
y_cal = y_tv[last_fold_val_idx]

dl_cal = DataLoader(SignDataset(X_cal, y_cal, augment=False),
                    batch_size=BATCH, shuffle=False, num_workers=0)
best_fold_model.eval()
all_logits, all_labels_cal = [], []
with torch.no_grad():
    for xb, yb in dl_cal:
        logits = best_fold_model(xb.to(DEVICE)).cpu()
        all_logits.append(logits)
        all_labels_cal.append(yb)
logits_cal = torch.cat(all_logits)
labels_cal = torch.cat(all_labels_cal)


def nll_with_temp(T):
    scaled = logits_cal / T
    return F.cross_entropy(scaled, labels_cal).item()


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
res = minimize_scalar(nll_with_temp, bounds=(0.1, 10.0), method="bounded")
T_opt = float(res.x)
ece_after = ece_score(logits_cal / T_opt, labels_cal)

print(f"  ECE antes de calibración  : {ece_before:.4f}")
print(f"  Temperatura óptima (T*)   : {T_opt:.3f}")
print(f"  ECE tras calibración      : {ece_after:.4f}")
print(f"  Mejora ECE                : {(ece_before - ece_after):.4f}")

# ── Guardar checkpoint ────────────────────────────────────────────────────────

print(f"\n{'='*65}")
print("GUARDANDO CHECKPOINT")
print(f"{'='*65}")

tiempo_total_s = (t_hpo_elapsed + t_final_elapsed) * 60

ckpt = {
    "model_state": {k: v.cpu() for k, v in best_fold_model.state_dict().items()},
    "label2idx":   label2idx,
    "idx2label":   idx2label,
    "n_classes":   n_classes,
    "n_dims":      N_DIMS,
    "n_frames":    N_FRAMES,
    "hidden":      best_params["hidden"],
    "n_layers":    best_params["n_layers"],
    "dropout":     best_params["dropout"],
    "lr":          best_params["lr"],
    "label_smoothing": best_params["label_smoothing"],
    "temperature": T_opt,
    "f1_val_mean": mean_f1,
    "f1_val_std":  std_f1,
    "f1_test":     f1_te,
    "acc_test":    acc_te,
    "ece_before":  ece_before,
    "ece_after":   ece_after,
    "fold_f1s":    fold_f1s,
    "hpo_best_trial": best_params,
    "hpo_f1":      best_f1_hpo,
    "sprint":      "S10",
}
pt_path = CKPT_DIR / "lstm_s10.pt"
torch.save(ckpt, pt_path)
print(f"  ✅ {pt_path}  ({pt_path.stat().st_size / 1e6:.1f} MB)")

# ── Exportar ONNX ─────────────────────────────────────────────────────────────

try:
    best_fold_model.cpu().eval()
    dummy  = torch.randn(1, N_FRAMES, N_DIMS)
    onnx_p = CKPT_DIR / "lstm_s10.onnx"
    torch.onnx.export(
        best_fold_model, dummy, str(onnx_p),
        input_names=["sequence"], output_names=["logits"],
        dynamic_axes={"sequence": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"  ✅ {onnx_p}  ({onnx_p.stat().st_size / 1e6:.1f} MB)")
except Exception as e:
    print(f"  ⚠️  ONNX export error: {e}")

# ── Latencia ONNX ─────────────────────────────────────────────────────────────

latencia_ms = 0.0
try:
    import onnxruntime as ort
    sess   = ort.InferenceSession(str(onnx_p))
    dummy_np = np.zeros((1, N_FRAMES, N_DIMS), dtype=np.float32)
    times  = []
    for _ in range(100):
        t0 = time.perf_counter()
        sess.run(None, {"sequence": dummy_np})
        times.append((time.perf_counter() - t0) * 1000)
    latencia_ms = float(np.mean(times))
    print(f"  Latencia ONNX: {latencia_ms:.1f} ms  "
          f"(p95={np.percentile(times, 95):.1f} ms)")
except Exception as e:
    print(f"  ⚠️  Latencia no medida: {e}")

# ── Actualizar logs/runs.csv ──────────────────────────────────────────────────

fecha = datetime.date.today().strftime("%Y%m%d")
hp_resumen = (f"hidden={best_params['hidden']},n_layers={best_params['n_layers']},"
              f"dropout={best_params['dropout']:.2f},lr={best_params['lr']:.0e},"
              f"ls={best_params['label_smoothing']:.2f}")
new_row = {
    "exp_id":       f"exp_{fecha}_lstm_s10_150dims_kfold5",
    "sprint":       "S10",
    "modelo":       "LSTM-Bidir-S10",
    "features":     "pose+rhand+lhand(150dims×30fr)",
    "hp_resumen":   hp_resumen,
    "f1_val_mean":  f"{mean_f1:.4f}",
    "f1_val_std":   f"{std_f1:.4f}",
    "f1_test":      f"{f1_te:.4f}",
    "acc_test":     f"{acc_te:.4f}",
    "tiempo_s":     f"{int(tiempo_total_s)}",
    "latencia_ms":  f"{latencia_ms:.1f}",
    "split":        f"StratifiedKFold({N_FOLDS})",
    "seed":         str(SEED),
    "n_classes":    str(n_classes),
    "notas":        (f"HPO Optuna best: hidden={best_params['hidden']},n_layers={best_params['n_layers']},"
                     f"drop={best_params['dropout']:.2f},lr={best_params['lr']:.2e},"
                     f"ls={best_params['label_smoothing']:.2f}; KFold({N_FOLDS}); "
                     f"ECE {ece_before:.3f}→{ece_after:.3f}(T={T_opt:.2f}); "
                     f"F1-test +{(f1_te/0.0109-1)*100:.0f}% vs S9; lat {latencia_ms:.1f}ms"),
}

runs_path = LOGS_DIR / "runs.csv"
fieldnames = list(new_row.keys())

# Leer encabezado existente para respetar columnas originales
if runs_path.exists():
    with open(runs_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or fieldnames

with open(runs_path, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writerow(new_row)

print(f"\n  ✅ logs/runs.csv actualizado")

# ── Resumen final ─────────────────────────────────────────────────────────────

print(f"\n{'='*65}")
print("RESUMEN SPRINT 10")
print(f"{'='*65}")
print(f"  Modelo         : LSPLSTMBidirS10  "
      f"({count_params(best_fold_model):,} params)")
print(f"  hidden         : {best_params['hidden']}  "
      f"(S9=256 → S10={best_params['hidden']})")
print(f"  dropout        : {best_params['dropout']:.2f}  "
      f"(S9=0.35 → S10={best_params['dropout']:.2f})")
print(f"  label_smooth   : {best_params['label_smoothing']:.2f}")
print()
print(f"  F1-val KFold(5): {mean_f1:.4f} ± {std_f1:.4f}")
print(f"  F1-test        : {f1_te:.4f}")
print(f"  Accuracy test  : {acc_te:.4f}")
print(f"  Mejora vs S9   : {mejora:+.1f}%  (S9 F1-val=0.0365)")
print()
print(f"  ECE antes      : {ece_before:.4f}  (S9=0.427)")
print(f"  ECE tras T={T_opt:.2f} : {ece_after:.4f}")
print(f"  Latencia ONNX  : {latencia_ms:.1f} ms")
print()
print(f"  Checkpoint     : checkpoints/lstm_s10.pt")
print(f"  ONNX           : checkpoints/lstm_s10.onnx")
print(f"  HPO trials     : {N_TRIALS}  |  KFolds: {N_FOLDS}")
print(f"  Tiempo total   : {t_hpo_elapsed:.1f} min (HPO) + {t_final_elapsed:.1f} min (CV)")

print(f"\n[LSTM S10 COMPLETADO]  F1-val={mean_f1:.4f}±{std_f1:.4f}  F1-test={f1_te:.4f}")
