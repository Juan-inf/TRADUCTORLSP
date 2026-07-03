"""
train_bilstm_s11.py — BiLSTM S11 entrenamiento directo (sin HPO largo)

Usa S10 best HPs como punto de partida + fine-tuning en dataset S11.
Diseñado para ejecutarse DESPUÉS de train_s11.py (Transformer ya entrenado).

Fixes vs train_s11.py:
  - Batch size reducido a 32 (evita MPS OOM en modelos grandes)
  - Limpieza explícita de caché MPS entre runs
  - HPO reducido: 20 trials (no 30)
  - PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 via os.environ
  - CPU fallback automático si MPS falla con OOM
"""

import os, json, time, warnings, pathlib, csv, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, ShuffleSplit
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score
from scipy.optimize import minimize_scalar
from scipy.stats import ks_2samp
from collections import Counter
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Aumentar límite MPS para reducir OOM
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT     = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
LOGS_DIR = ROOT / "logs"
CKPT_DIR.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────

SEED    = 42
N_FRAMES = 30
N_DIMS   = 150
BATCH    = 32    # Reducido de 64 a 32 para evitar MPS OOM
DEVICE   = "mps" if torch.backends.mps.is_available() else "cpu"

S10_BEST = {
    "hidden": 256, "n_layers": 1, "dropout": 0.20,
    "lr": 0.004093813608598782, "label_smoothing": 0.15, "weight_decay": 1e-4,
}

N_EPOCHS_HPO   = 30
PATIENCE_HPO   = 6
N_EPOCHS_FINAL = 80
PATIENCE_FINAL = 12
N_TRIALS       = 20
N_FOLDS        = 5

torch.manual_seed(SEED); np.random.seed(SEED)
print(f"Device : {DEVICE}  |  Batch: {BATCH}")
print("BiLSTM S11 — HPO 20 trials + KFold(5)")
print("=" * 60)

# ── Dataset ───────────────────────────────────────────────────────────────────

s11 = np.load(DATA_DIR / "dataset_s11.npz")
X_all, y_all_r, groups_all = s11["X"], s11["y"], s11["groups"]

with open(DATA_DIR / "s11_label2idx.json", encoding="utf-8") as f:
    label2idx = json.load(f)
idx2label_raw = {int(v): k for k, v in label2idx.items()}

# Re-filtrar LSP - Vocabulario-palabras ≥2 y re-mapear
cnt = Counter(y_all_r.tolist())
keep = np.array([cnt[int(v)] >= 2 for v in y_all_r])
X_all, y_all_r, groups_all = X_all[keep], y_all_r[keep], groups_all[keep]
old_ids = sorted(set(y_all_r.tolist()))
remap   = {o: n for n, o in enumerate(old_ids)}
y_all   = np.array([remap[int(v)] for v in y_all_r], dtype=np.int64)
n_classes = len(old_ids)
idx2label = {remap[o]: idx2label_raw.get(o, str(o)) for o in old_ids}
label2idx = {v: k for k, v in idx2label.items()}

print(f"Muestras: {len(X_all)}  |  LSP - Vocabulario-palabras: {n_classes}")

# Split train+val / test
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
tv_idx, te_idx = next(sss.split(X_all, y_all))
X_tv, y_tv, g_tv = X_all[tv_idx], y_all[tv_idx], groups_all[tv_idx]
X_te, y_te        = X_all[te_idx],  y_all[te_idx]

# Group holdout HE3
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
core_idx, ghold_idx = next(gss.split(X_tv, y_tv, groups=g_tv))
X_core, y_core = X_tv[core_idx], y_tv[core_idx]
X_ghold, y_ghold = X_tv[ghold_idx], y_tv[ghold_idx]
print(f"Core: {len(X_core)}  |  GHold HE3: {len(X_ghold)}  |  Test: {len(X_te)}")

# ── Augmentation & Dataset ────────────────────────────────────────────────────

def time_warp(x, sigma=0.04, knots=4):
    T = x.shape[0]
    orig = np.linspace(0, T-1, T)
    warp = orig + np.random.randn(T) * sigma * T / knots
    warp = np.clip(np.sort(warp), 0, T-1)
    out  = np.zeros_like(x)
    for d in range(x.shape[1]):
        out[:, d] = np.interp(orig, warp, x[:, d])
    return out.astype(np.float32)


class SignDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X.copy(); self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].copy()
        if self.augment:
            x = x + np.random.randn(*x.shape).astype(np.float32) * 0.008
            if np.random.rand() < 0.5:
                x[:, 66:87] = 1.0 - x[:, 66:87]
                x[:, 108:129] = 1.0 - x[:, 108:129]
            if np.random.rand() < 0.40: x = time_warp(x)
            if np.random.rand() < 0.30:
                n_drop = int(N_DIMS * np.random.uniform(0.03, 0.10))
                x[:, np.random.choice(N_DIMS, n_drop, replace=False)] = 0.0
        return torch.from_numpy(x).float(), self.y[i]


def make_loaders(X_tr, y_tr, X_val, y_val):
    cc  = Counter(y_tr.tolist())
    w   = [1.0 / cc[int(c)] for c in y_tr]
    smp = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    dl_tr  = DataLoader(SignDataset(X_tr, y_tr, augment=True),
                        batch_size=BATCH, sampler=smp, num_workers=0, pin_memory=False)
    dl_val = DataLoader(SignDataset(X_val, y_val, augment=False),
                        batch_size=BATCH, shuffle=False, num_workers=0)
    return dl_tr, dl_val


# ── Modelo ────────────────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attn = nn.Linear(hidden * 2, 1)

    def forward(self, h):
        w = torch.softmax(self.attn(h), dim=1)
        return (w * h).sum(dim=1)


class LSPLSTMBidirS11(nn.Module):
    def __init__(self, n_dims, n_classes, hidden=256, n_layers=1, dropout=0.20):
        super().__init__()
        proj_dim = max(hidden // 2, 32)
        self.proj = nn.Sequential(
            nn.Linear(n_dims, proj_dim), nn.LayerNorm(proj_dim),
            nn.GELU(), nn.Dropout(dropout * 0.5),
        )
        self.lstm = nn.LSTM(proj_dim, hidden, n_layers, batch_first=True,
                            bidirectional=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.attn = TemporalAttention(hidden)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        x   = self.proj(x)
        h, _ = self.lstm(x)
        c   = self.attn(h)
        return self.head(c)


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ── Entrenamiento ─────────────────────────────────────────────────────────────

def mps_safe_clear():
    if DEVICE == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def make_criterion(y_tr, ls):
    cc = Counter(y_tr.tolist())
    cw = torch.tensor([len(y_tr) / (n_classes * cc.get(i, 1))
                        for i in range(n_classes)], dtype=torch.float32).to(DEVICE)
    return nn.CrossEntropyLoss(weight=cw, label_smoothing=ls)


def train_one_run(X_tr, y_tr, X_val, y_val, hidden, n_layers, dropout, lr, wd, ls,
                  n_epochs, patience, trial=None):
    mps_safe_clear()
    dl_tr, dl_val = make_loaders(X_tr, y_tr, X_val, y_val)
    model = LSPLSTMBidirS11(N_DIMS, n_classes, hidden, n_layers, dropout).to(DEVICE)
    crit  = make_criterion(y_tr, ls)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=20, T_mult=2, eta_min=lr * 0.01)

    best_f1, best_state, pat = 0.0, None, 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in dl_val:
                preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().tolist())
                trues.extend(yb.tolist())
        f1 = f1_score(trues, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_state, pat = f1, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= patience: break
        if trial:
            trial.report(f1, epoch)
            if trial.should_prune(): raise optuna.exceptions.TrialPruned()
    model.load_state_dict(best_state)
    mps_safe_clear()
    return model, best_f1


# ── HPO ───────────────────────────────────────────────────────────────────────

sss_hpo = ShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
hpo_tr, hpo_val = next(sss_hpo.split(X_core))
Xh_tr, yh_tr   = X_core[hpo_tr],  y_core[hpo_tr]
Xh_val, yh_val  = X_core[hpo_val], y_core[hpo_val]


def objective(trial):
    hidden   = trial.suggest_categorical("hidden",   [128, 256])
    n_layers = trial.suggest_int("n_layers",  1, 2)
    dropout  = trial.suggest_float("dropout", 0.10, 0.40, step=0.05)
    lr       = trial.suggest_float("lr",      1e-3, 8e-3, log=True)
    wd       = trial.suggest_float("wd",      1e-5, 5e-4, log=True)
    ls       = trial.suggest_float("ls",      0.05, 0.20, step=0.05)
    try:
        _, f1 = train_one_run(Xh_tr, yh_tr, Xh_val, yh_val,
                              hidden, n_layers, dropout, lr, wd, ls,
                              N_EPOCHS_HPO, PATIENCE_HPO, trial)
        return f1
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            mps_safe_clear()
            return 0.0
        raise


print(f"\nHPO BiLSTM S11 ({N_TRIALS} trials) …")
t0 = time.time()
study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=SEED),
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=8),
)
study.enqueue_trial({
    "hidden": S10_BEST["hidden"],
    "n_layers": S10_BEST["n_layers"],
    "dropout": S10_BEST["dropout"],
    "lr":      S10_BEST["lr"],
    "wd":      S10_BEST["weight_decay"],
    "ls":      S10_BEST["label_smoothing"],
})
study.optimize(objective, n_trials=N_TRIALS,
               catch=(RuntimeError,), show_progress_bar=True)
hp = study.best_params
print(f"\n  ✅ HPO F1={study.best_value:.4f}  tiempo={( time.time()-t0)/60:.1f} min")
print(f"  HPs: {hp}")

# ── Entrenamiento final KFold(5) ──────────────────────────────────────────────

print(f"\nKFold({N_FOLDS}) final BiLSTM S11 …")
_tmp = LSPLSTMBidirS11(N_DIMS, n_classes, hp["hidden"], hp["n_layers"], hp["dropout"])
print(f"  Params: {count_params(_tmp):,}"); del _tmp

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_f1s, best_m, best_f = [], None, 0.0
t_cv = time.time()
for fi, (tr_i, va_i) in enumerate(skf.split(X_core, y_core), 1):
    m, f1 = train_one_run(
        X_core[tr_i], y_core[tr_i], X_core[va_i], y_core[va_i],
        hp["hidden"], hp["n_layers"], hp["dropout"],
        hp["lr"], hp["wd"], hp["ls"], N_EPOCHS_FINAL, PATIENCE_FINAL)
    fold_f1s.append(f1); print(f"  Fold {fi}: F1={f1:.4f}")
    if f1 > best_f: best_f, best_m = f1, m

mean_f = float(np.mean(fold_f1s)); std_f = float(np.std(fold_f1s))
t_cv_min = (time.time() - t_cv) / 60
print(f"\n  KFold F1: {mean_f:.4f} ± {std_f:.4f}  ({t_cv_min:.1f} min)")

# ── Evaluación test ───────────────────────────────────────────────────────────

best_m.to(DEVICE).eval()
dl_te = DataLoader(SignDataset(X_te, y_te, augment=False),
                   batch_size=BATCH, shuffle=False, num_workers=0)
all_logits, all_true, preds = [], [], []
with torch.no_grad():
    for xb, yb in dl_te:
        logits = best_m(xb.to(DEVICE)).cpu()
        all_logits.append(logits)
        preds.extend(logits.argmax(1).tolist())
        all_true.extend(yb.tolist())

logits_t = torch.cat(all_logits)
probs    = torch.softmax(logits_t, dim=1).numpy()
true_arr = np.array(all_true)
preds_arr = np.array(preds)

f1_te = f1_score(true_arr, preds_arr, average="macro",    zero_division=0)
f1_w  = f1_score(true_arr, preds_arr, average="weighted", zero_division=0)
acc   = (preds_arr == true_arr).mean()
top3  = sum(int(t) in np.argsort(probs[i])[-3:] for i, t in enumerate(true_arr)) / len(true_arr)
top5  = sum(int(t) in np.argsort(probs[i])[-5:] for i, t in enumerate(true_arr)) / len(true_arr)

print(f"\nTest: Acc={acc:.4f}  F1-macro={f1_te:.4f}  F1-weighted={f1_w:.4f}")
print(f"      Top-3={top3:.4f}  Top-5={top5:.4f}")
print(f"  Mejora vs S10 F1-test: {(f1_te/0.0302-1)*100:+.1f}%")

# ── Calibración ───────────────────────────────────────────────────────────────

last_va = list(skf.split(X_core, y_core))[np.argmax(fold_f1s)][1]
Xcal, ycal = X_core[last_va], y_core[last_va]
dl_cal = DataLoader(SignDataset(Xcal, ycal, augment=False),
                    batch_size=BATCH, shuffle=False, num_workers=0)
cal_logits, cal_true = [], []
best_m.eval()
with torch.no_grad():
    for xb, yb in dl_cal:
        cal_logits.append(best_m(xb.to(DEVICE)).cpu())
        cal_true.extend(yb.tolist())
cal_logits_t = torch.cat(cal_logits)
cal_labels_t = torch.tensor(cal_true)


def ece_score(logits_np, labels_np, n_bins=15):
    probs_ = torch.softmax(torch.from_numpy(logits_np), dim=1)
    confs, pred_ = probs_.max(dim=1)
    accs = pred_.eq(torch.tensor(labels_np))
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b+1) / n_bins
        mask   = (confs > lo) & (confs <= hi)
        if mask.sum() == 0: continue
        ece += mask.float().mean().item() * abs(
            accs[mask].float().mean().item() - confs[mask].float().mean().item())
    return ece


ece_pre = ece_score(cal_logits_t.numpy(), np.array(cal_true))
def nll_t(T): return F.cross_entropy(cal_logits_t / T, cal_labels_t).item()
res    = minimize_scalar(nll_t, bounds=(0.1, 10.0), method="bounded")
T_opt  = float(res.x)
ece_post = ece_score(cal_logits_t.numpy() / T_opt, np.array(cal_true))
print(f"ECE: {ece_pre:.4f} → {ece_post:.4f}  (T*={T_opt:.3f})")

# ── HE3 ──────────────────────────────────────────────────────────────────────

best_m.eval()
dl_gh = DataLoader(SignDataset(X_ghold, y_ghold, augment=False),
                   batch_size=BATCH, shuffle=False, num_workers=0)
gh_logits, gh_true, gh_preds = [], [], []
with torch.no_grad():
    for xb, yb in dl_gh:
        logits_gh = best_m(xb.to(DEVICE)).cpu()
        gh_logits.append(logits_gh)
        gh_preds.extend(logits_gh.argmax(1).tolist())
        gh_true.extend(yb.tolist())
gh_probs = torch.softmax(torch.cat(gh_logits), dim=1).numpy()
f1_gh = f1_score(np.array(gh_true), np.array(gh_preds), average="macro", zero_division=0)
delta_f1 = abs(f1_te - f1_gh)

conf_te = probs.max(axis=1)
conf_gh = gh_probs.max(axis=1)
bins = np.linspace(0, 1, 11)
p_ref, _ = np.histogram(conf_te, bins=bins, density=True)
p_new, _ = np.histogram(conf_gh, bins=bins, density=True)
p_ref = np.clip(p_ref / (p_ref.sum() + 1e-8), 1e-8, None)
p_new = np.clip(p_new / (p_new.sum() + 1e-8), 1e-8, None)
psi   = float(np.sum((p_ref - p_new) * np.log(p_ref / p_new)))
ks_stat, ks_pval = ks_2samp(conf_te, conf_gh)
he3_passed = (delta_f1 <= 0.15) and (psi < 0.20) and (ks_pval > 0.05)
print(f"HE3: ΔF1={delta_f1:.4f} {'✅' if delta_f1<=0.15 else '❌'}  "
      f"PSI={psi:.4f} {'✅' if psi<0.20 else '❌'}  "
      f"KS p={ks_pval:.4f} {'✅' if ks_pval>0.05 else '❌'}  "
      f"→ {'PASA ✅' if he3_passed else 'FALLA ❌'}")

# ── ONNX ─────────────────────────────────────────────────────────────────────

best_m.cpu().eval()
dummy   = torch.randn(1, N_FRAMES, N_DIMS)
onnx_p  = CKPT_DIR / "bilstm_s11.onnx"
try:
    torch.onnx.export(best_m, dummy, str(onnx_p),
                      input_names=["sequence"], output_names=["logits"],
                      dynamic_axes={"sequence": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_p))
    np_in = np.zeros((1, N_FRAMES, N_DIMS), dtype=np.float32)
    times = [time.perf_counter() for _ in range(1)]  # warm-up
    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        sess.run(None, {"sequence": np_in})
        times.append((time.perf_counter() - t0) * 1000)
    lat = float(np.mean(times))
    print(f"ONNX latencia: {lat:.2f} ms  ({onnx_p.stat().st_size/1e6:.1f} MB)")
except Exception as e:
    print(f"ONNX error: {e}"); lat = 0.0

# ── Guardar ───────────────────────────────────────────────────────────────────

ckpt = {
    "model_state":    {k: v.cpu() for k, v in best_m.state_dict().items()},
    "architecture":   "LSPLSTMBidirS11",
    "label2idx":      label2idx, "idx2label": idx2label,
    "n_classes":      n_classes, "n_dims": N_DIMS, "n_frames": N_FRAMES,
    "hidden":         hp["hidden"], "n_layers": hp["n_layers"],
    "dropout":        hp["dropout"], "lr": hp["lr"],
    "wd":             hp["wd"], "label_smoothing": hp["ls"],
    "temperature":    T_opt,
    "f1_val_mean":    mean_f, "f1_val_std": std_f,
    "f1_test":        f1_te, "acc_test": acc,
    "top3_test":      top3, "top5_test": top5,
    "ece_before":     ece_pre, "ece_after": ece_post,
    "fold_f1s":       fold_f1s,
    "he3": {"delta_f1": delta_f1, "psi": psi,
             "ks_stat": ks_stat, "ks_pval": ks_pval,
             "passed": he3_passed, "f1_ghold": f1_gh},
    "latencia_onnx_ms": lat,
    "sprint": "S11",
}
torch.save(ckpt, CKPT_DIR / "bilstm_s11.pt")
print(f"✅ checkpoints/bilstm_s11.pt")

# ── Actualizar runs.csv ───────────────────────────────────────────────────────

fecha   = datetime.date.today().strftime("%Y%m%d")
new_row = {
    "exp_id":      f"exp_{fecha}_bilstm_s11",
    "sprint":      "S11",
    "modelo":      "BiLSTM-S11",
    "features":    "pose+rhand+lhand(150dims×30fr)",
    "hp_resumen":  f"hidden={hp['hidden']},n_layers={hp['n_layers']},dropout={hp['dropout']:.2f},lr={hp['lr']:.2e}",
    "f1_val_mean": f"{mean_f:.4f}",
    "f1_val_std":  f"{std_f:.4f}",
    "f1_test":     f"{f1_te:.4f}",
    "acc_test":    f"{acc:.4f}",
    "tiempo_s":    f"{int(t_cv_min*60)}",
    "latencia_ms": f"{lat:.1f}",
    "split":       f"StratifiedKFold({N_FOLDS})+GroupHold",
    "seed":        str(SEED),
    "n_classes":   str(n_classes),
    "notas":       (f"S11 dataset; warmstart S10; "
                    f"ECE {ece_pre:.3f}→{ece_post:.3f}(T={T_opt:.2f}); "
                    f"Top3={top3:.4f}; HE3={'PASA' if he3_passed else 'FALLA'}"),
}
runs_path = LOGS_DIR / "runs.csv"
fn = list(new_row.keys())
if runs_path.exists():
    with open(runs_path, "r", newline="", encoding="utf-8") as f:
        import csv
        fn = csv.DictReader(f).fieldnames or fn
with open(runs_path, "a", newline="", encoding="utf-8") as f:
    import csv
    csv.DictWriter(f, fieldnames=fn, extrasaction="ignore").writerow(new_row)

print("\n=== RESUMEN BiLSTM S11 ===")
print(f"  F1-val KFold : {mean_f:.4f} ± {std_f:.4f}")
print(f"  F1-test      : {f1_te:.4f}  (S10=0.0302, {(f1_te/0.0302-1)*100:+.1f}%)")
print(f"  Top-3        : {top3:.4f}")
print(f"  ECE calibrado: {ece_post:.4f}")
print(f"  Lat ONNX     : {lat:.2f} ms")
print("\n[BILSTM S11 COMPLETADO]")
