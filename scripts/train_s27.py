"""
train_s27.py — Sprint 27: BiLSTM sobre dataset_s17 (grupos cross-source corregidos)

Diferencias vs S26:
  Dataset    : dataset_s17.npz — igual a S16 pero con grupos unificados para
               clases que aparecen en múltiples fuentes (dgi156 + vineta).
               Fix root-cause del HE3: HISTORIAS_VINETAS_XX ya no tiene
               cross-source contamination (F1≈0 con 234 samples en holdout).
  HPs        : S13-best warm-start (idéntico a S26)
  Augment    : agresivo (idéntico a S26)
  Objetivo   : F1-test ≥ 0.35 Y HE3 PASA (ΔF1 = 0.1663 en S26, umbral 0.15)

Checkpoints : bilstm_s27.pt / bilstm_s27.onnx
"""

import json, time, warnings, pathlib, csv, datetime, gc, os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def _free_mps():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.metrics import f1_score
from scipy.optimize import minimize_scalar
from scipy.stats import ks_2samp
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

SEED     = 42
N_FRAMES = 30
N_DIMS   = 150
BATCH    = 64
DEVICE   = "mps" if torch.backends.mps.is_available() else "cpu"

S13_BEST = {
    "hidden": 256, "n_layers": 1, "dropout": 0.20,
    "lr": 1.709e-3, "wd": 1.296e-4, "ls": 0.15,
}

N_EPOCHS_HPO   = 35
PATIENCE_HPO   = 8
N_EPOCHS_FINAL = 100
PATIENCE_FINAL = 15
N_TRIALS       = 20
N_FOLDS        = 5

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--min-muestras", type=int, default=15)
_ap.add_argument("--skip-hpo", action="store_true",
                 help="Usar HPs S13-best directamente, sin HPO")
_args = _ap.parse_args()
MIN_SAMPLES = _args.min_muestras
SKIP_HPO    = _args.skip_hpo

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device  : {DEVICE}")
print(f"Sprint 27 — BiLSTM S27 | dataset_s17 (grupos cross-source fix) | min={MIN_SAMPLES}")
print("=" * 65)


def normalize_sample(x: np.ndarray) -> np.ndarray:
    mu  = x.mean()
    std = x.std()
    if std < 1e-8:
        return x
    return ((x - mu) / std).astype(np.float32)


def normalize_batch(X: np.ndarray) -> np.ndarray:
    return np.stack([normalize_sample(X[i]) for i in range(len(X))])


def load_dataset():
    npz_path = DATA_DIR / "dataset_s17.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            "dataset_s17.npz no encontrado.\n"
            "  python3 scripts/build_dataset_s17.py"
        )

    data   = np.load(npz_path)
    X_raw  = data["X"]
    y_raw  = data["y"]
    groups = data["groups"]

    with open(DATA_DIR / "s17_label2idx.json", encoding="utf-8") as f:
        label2idx = json.load(f)
    idx2label = {int(v): k for k, v in label2idx.items()}

    counts = Counter(y_raw.tolist())
    keep   = np.array([counts[int(v)] >= MIN_SAMPLES for v in y_raw])
    X_raw, y_raw, groups = X_raw[keep], y_raw[keep], groups[keep]

    print("  Aplicando normalización por muestra (z-score)...")
    X = normalize_batch(X_raw)

    old_ids   = sorted(set(y_raw.tolist()))
    remap     = {old: new for new, old in enumerate(old_ids)}
    y         = np.array([remap[int(v)] for v in y_raw], dtype=np.int64)
    n_cl      = len(old_ids)
    idx2label = {remap[o]: idx2label.get(o, str(o)) for o in old_ids}
    label2idx = {v: k for k, v in idx2label.items()}

    print(f"\nDataset S27 (S17 — grupos cross-source fix):")
    print(f"  Muestras  : {len(X)}  |  Clases activas: {n_cl}")
    cnt = Counter(y.tolist())
    v   = sorted(cnt.values())
    print(f"  Samples/cls: min={v[0]} max={v[-1]} mean={np.mean(v):.1f} median={np.median(v):.0f}")
    return X, y, groups, n_cl, label2idx, idx2label


X_all, y_all, groups_all, n_classes, label2idx, idx2label = load_dataset()

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
tv_idx, te_idx = next(sss.split(X_all, y_all))
X_tv, y_tv, g_tv = X_all[tv_idx], y_all[tv_idx], groups_all[tv_idx]
X_te, y_te        = X_all[te_idx],  y_all[te_idx]
print(f"  Train+Val : {len(X_tv)}  |  Test holdout: {len(X_te)}")

gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
core_idx, group_hold_idx = next(gss.split(X_tv, y_tv, groups=g_tv))
X_core,  y_core  = X_tv[core_idx],       y_tv[core_idx]
g_core           = g_tv[core_idx]
X_ghold, y_ghold = X_tv[group_hold_idx], y_tv[group_hold_idx]
print(f"  Core train: {len(X_core)}  |  Group holdout HE3: {len(X_ghold)}")

clases_core  = set(y_core.tolist())
clases_ghold = set(y_ghold.tolist())
sin_training = clases_ghold - clases_core
print(f"  Clases holdout sin training: {len(sin_training)}")
if sin_training:
    nombres_st = [idx2label.get(c, str(c)) for c in sorted(sin_training)]
    print(f"  → {nombres_st}")


def time_warp(x, sigma=0.04, knots=4):
    T    = x.shape[0]
    orig = np.linspace(0, T - 1, T)
    warp = np.clip(np.sort(orig + np.random.randn(T) * sigma * T / knots), 0, T - 1)
    out  = np.zeros_like(x)
    for d in range(x.shape[1]):
        out[:, d] = np.interp(orig, warp, x[:, d])
    return out.astype(np.float32)


class SignDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X       = X.copy()
        self.y       = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].copy()
        if self.augment:
            x = x + np.random.randn(*x.shape).astype(np.float32) * 0.020
            if np.random.rand() < 0.60:
                x = x * np.random.uniform(0.85, 1.15)
            if np.random.rand() < 0.70:
                x[:, 66:87]   = -x[:, 66:87]
                x[:, 108:129] = -x[:, 108:129]
            if np.random.rand() < 0.60:
                x = time_warp(x, sigma=0.06)
            if np.random.rand() < 0.50:
                n_drop = int(N_DIMS * np.random.uniform(0.05, 0.15))
                x[:, np.random.choice(N_DIMS, n_drop, replace=False)] = 0.0
            x = normalize_sample(x)
        return torch.from_numpy(x).float(), self.y[i]


def make_loaders(X_tr, y_tr, X_val, y_val, batch=BATCH):
    cc  = Counter(y_tr.tolist())
    w   = [1.0 / cc[int(c)] for c in y_tr]
    smp = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    dl_tr  = DataLoader(SignDataset(X_tr, y_tr,  augment=True),
                        batch_size=batch, sampler=smp, num_workers=0)
    dl_val = DataLoader(SignDataset(X_val, y_val, augment=False),
                        batch_size=batch, shuffle=False, num_workers=0)
    return dl_tr, dl_val


def make_criterion(y_tr, label_smoothing):
    cc = Counter(y_tr.tolist())
    cw = torch.tensor(
        [len(y_tr) / (n_classes * cc.get(i, 1)) for i in range(n_classes)],
        dtype=torch.float32,
    ).to(DEVICE)
    return nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)


class TemporalAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attn = nn.Linear(hidden * 2, 1)

    def forward(self, h):
        w = torch.softmax(self.attn(h), dim=1)
        return (w * h).sum(dim=1)


class LSPLSTMBidirS27(nn.Module):
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
            nn.Dropout(dropout), nn.Linear(hidden * 2, hidden),
            nn.GELU(), nn.Dropout(dropout * 0.5), nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        x    = self.proj(x)
        h, _ = self.lstm(x)
        return self.head(self.attn(h))


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def train_one_run(X_tr, y_tr, X_val, y_val, model_fn,
                  lr, wd, label_smoothing, n_epochs, patience, trial=None):
    dl_tr, dl_val = make_loaders(X_tr, y_tr, X_val, y_val)
    model     = model_fn().to(DEVICE)
    criterion = make_criterion(y_tr, label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=25, T_mult=2, eta_min=lr * 0.01)

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


@torch.no_grad()
def full_eval(model, X_eval, y_eval, label="eval"):
    model.to(DEVICE).eval()
    dl = DataLoader(SignDataset(X_eval, y_eval, augment=False),
                    batch_size=BATCH, shuffle=False, num_workers=0)
    all_logits, all_true = [], []
    for xb, yb in dl:
        all_logits.append(model(xb.to(DEVICE)).cpu())
        all_true.extend(yb.tolist())

    logits_t = torch.cat(all_logits)
    probs    = torch.softmax(logits_t, dim=1).numpy()
    preds    = np.argmax(probs, axis=1)
    true_arr = np.array(all_true)

    acc  = (preds == true_arr).mean()
    f1_m = f1_score(true_arr, preds, average="macro", zero_division=0)
    top3 = sum(int(t) in np.argsort(probs[i])[-3:] for i, t in enumerate(true_arr)) / len(true_arr)
    top5 = sum(int(t) in np.argsort(probs[i])[-5:] for i, t in enumerate(true_arr)) / len(true_arr)

    print(f"  [{label}] Acc={acc:.4f}  F1-macro={f1_m:.4f}  Top-3={top3:.4f}  Top-5={top5:.4f}")
    return {"f1_macro": f1_m, "acc": acc, "top3": top3, "top5": top5,
            "probs": probs, "preds": preds, "true": true_arr, "logits": logits_t.numpy()}


def per_class_f1_holdout(res, idx2label, top_n_worst=20):
    from sklearn.metrics import f1_score as _f1
    classes = sorted(set(res["true"].tolist()))
    f1s     = _f1(res["true"], res["preds"], labels=classes,
                  average=None, zero_division=0)
    pairs   = sorted(zip(f1s, classes), key=lambda x: x[0])
    print(f"  Peores {min(top_n_worst, len(pairs))} clases en holdout:")
    for score, cls in pairs[:top_n_worst]:
        nombre = idx2label.get(cls, str(cls))
        cnt    = int((res["true"] == cls).sum())
        print(f"    F1={score:.3f}  n={cnt:3d}  {nombre}")
    return {idx2label.get(c, str(c)): float(s) for s, c in pairs}


def ece_score(logits_np, labels_np, n_bins=15):
    logits = torch.from_numpy(logits_np)
    labels = torch.from_numpy(labels_np).long()
    probs  = torch.softmax(logits, dim=1)
    confs, preds = probs.max(dim=1)
    accs   = preds.eq(labels)
    ece    = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        mask   = (confs > lo) & (confs <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.float().mean().item() * abs(
            accs[mask].float().mean().item() - confs[mask].float().mean().item())
    return ece


def calibrate(logits_np, labels_np):
    logits_t = torch.from_numpy(logits_np)
    labels_t = torch.from_numpy(labels_np).long()
    ece_pre  = ece_score(logits_np, labels_np)

    def nll_t(T):
        return F.cross_entropy(logits_t / T, labels_t).item()

    res   = minimize_scalar(nll_t, bounds=(0.1, 10.0), method="bounded")
    T_opt = float(res.x)
    return T_opt, ece_pre, ece_score(logits_np / T_opt, labels_np)


def he3_eval(model, res_test, X_ghold, y_ghold, T_opt):
    res_gh   = full_eval(model, X_ghold, y_ghold, label="group-holdout")
    delta_f1 = abs(res_test["f1_macro"] - res_gh["f1_macro"])

    conf_te = res_test["probs"].max(axis=1)
    conf_gh = res_gh["probs"].max(axis=1)

    bins  = np.linspace(0, 1, 11)
    p_ref, _ = np.histogram(conf_te, bins=bins, density=True)
    p_new, _ = np.histogram(conf_gh, bins=bins, density=True)
    p_ref = np.clip(p_ref / (p_ref.sum() + 1e-8), 1e-8, None)
    p_new = np.clip(p_new / (p_new.sum() + 1e-8), 1e-8, None)
    psi   = float(np.sum((p_ref - p_new) * np.log(p_ref / p_new)))

    ks_stat, ks_pval = ks_2samp(conf_te, conf_gh)

    ok_f1  = delta_f1 <= 0.15
    ok_psi = psi       < 0.20
    ok_ks  = ks_pval   > 0.05
    passed = ok_f1 and ok_psi and ok_ks

    print(f"  HE3 → ΔF1={delta_f1:.4f} {'✅' if ok_f1 else '❌'}  "
          f"PSI={psi:.4f} {'✅' if ok_psi else '❌'}  "
          f"KS p={ks_pval:.4f} {'✅' if ok_ks else '❌'}  "
          f"→ {'PASA ✅' if passed else 'FALLA ❌'}")
    print()
    pcf1 = per_class_f1_holdout(res_gh, idx2label, top_n_worst=20)

    return {"delta_f1": delta_f1, "psi": psi, "ks_stat": ks_stat,
            "ks_pval": ks_pval, "passed": passed, "f1_ghold": res_gh["f1_macro"],
            "per_class_f1_ghold": pcf1}


def export_onnx(model, name):
    model.cpu().eval()
    dummy  = torch.randn(1, N_FRAMES, N_DIMS)
    onnx_p = CKPT_DIR / name
    try:
        torch.onnx.export(
            model, dummy, str(onnx_p),
            input_names=["sequence"], output_names=["logits"],
            dynamic_axes={"sequence": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )
        import onnxruntime as ort
        sess  = ort.InferenceSession(str(onnx_p))
        np_in = np.zeros((1, N_FRAMES, N_DIMS), dtype=np.float32)
        times = []
        for _ in range(200):
            t0 = time.perf_counter()
            sess.run(None, {"sequence": np_in})
            times.append((time.perf_counter() - t0) * 1000)
        lat = float(np.mean(times))
        print(f"  ONNX → {onnx_p.name}  ({onnx_p.stat().st_size/1e6:.1f} MB)  lat={lat:.2f} ms")
        return lat
    except Exception as e:
        print(f"  ONNX error: {e}")
        return 0.0


# ════════════════════════════════════════════════════════════════════════════
# BLOQUE A — HPO
# ════════════════════════════════════════════════════════════════════════════

if SKIP_HPO:
    hp      = {k: S13_BEST[k] for k in ("hidden", "n_layers", "dropout", "lr", "wd", "ls")}
    f1_hpo  = 0.0
    hpo_min = 0.0
    print(f"\n{'='*65}")
    print("BLOQUE A — HPO omitido (--skip-hpo): usando parámetros S13-best")
    print(f"{'='*65}")
    print(f"  HPs: {hp}")
else:
    print(f"\n{'='*65}")
    print("BLOQUE A — HPO BiLSTM S27  (warm-start S13-best, 20 trials)")
    print(f"{'='*65}")

    from sklearn.model_selection import ShuffleSplit
    sss_hpo = ShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    hpo_tr, hpo_val = next(sss_hpo.split(X_core))
    Xh_tr, yh_tr    = X_core[hpo_tr],  y_core[hpo_tr]
    Xh_val, yh_val  = X_core[hpo_val], y_core[hpo_val]

    def objective(trial):
        hidden   = trial.suggest_categorical("hidden",   [128, 256])
        n_layers = trial.suggest_int("n_layers", 1, 3)
        dropout  = trial.suggest_float("dropout", 0.10, 0.45, step=0.05)
        lr       = trial.suggest_float("lr",      5e-4, 8e-3, log=True)
        wd       = trial.suggest_float("wd",      1e-5, 5e-4, log=True)
        ls       = trial.suggest_float("ls",      0.05, 0.25, step=0.05)

        def make():
            return LSPLSTMBidirS27(N_DIMS, n_classes,
                                   hidden=hidden, n_layers=n_layers, dropout=dropout)

        m_trial, f1 = train_one_run(Xh_tr, yh_tr, Xh_val, yh_val, make,
                              lr=lr, wd=wd, label_smoothing=ls,
                              n_epochs=N_EPOCHS_HPO, patience=PATIENCE_HPO, trial=trial)
        del m_trial
        _free_mps()
        return f1

    t0    = time.time()
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=SEED),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    study.enqueue_trial({
        "hidden":   S13_BEST["hidden"],   "n_layers": S13_BEST["n_layers"],
        "dropout":  S13_BEST["dropout"],  "lr":       S13_BEST["lr"],
        "wd":       S13_BEST["wd"],       "ls":       S13_BEST["ls"],
    })
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    hp      = study.best_params
    f1_hpo  = study.best_value
    hpo_min = (time.time() - t0) / 60
    print(f"\n  ✅ HPO → F1-hpo={f1_hpo:.4f}  tiempo={hpo_min:.1f} min")
    print(f"  Mejores HPs: {hp}")


# ════════════════════════════════════════════════════════════════════════════
# BLOQUE B — KFold(5) final
# ════════════════════════════════════════════════════════════════════════════

print(f"\nB. KFold({N_FOLDS}) final BiLSTM S27")


def make_best():
    return LSPLSTMBidirS27(N_DIMS, n_classes,
                           hidden=hp["hidden"], n_layers=hp["n_layers"],
                           dropout=hp["dropout"])


print(f"  Params: {count_params(make_best()):,}")

skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_f1s, best_f1, best_model = [], 0.0, None
t_final = time.time()

for fi, (tr_i, va_i) in enumerate(skf.split(X_core, y_core), 1):
    m, f1 = train_one_run(
        X_core[tr_i], y_core[tr_i], X_core[va_i], y_core[va_i],
        model_fn=make_best,
        lr=hp["lr"], wd=hp["wd"], label_smoothing=hp["ls"],
        n_epochs=N_EPOCHS_FINAL, patience=PATIENCE_FINAL,
    )
    fold_f1s.append(f1)
    print(f"  Fold {fi}: F1-val={f1:.4f}")
    if f1 > best_f1:
        best_f1, best_model = f1, m
    elif m is not best_model:
        del m
    _free_mps()

total_min = (time.time() - t_final) / 60
mean_f1   = float(np.mean(fold_f1s))
std_f1    = float(np.std(fold_f1s))
print(f"\n  KFold F1: {mean_f1:.4f} ± {std_f1:.4f}  ({total_min:.1f} min)")


# ── Evaluación ────────────────────────────────────────────────────────────────

print("\nC. Evaluación en test holdout:")
res_te = full_eval(best_model, X_te, y_te, "BiLSTM-S27-test")

best_fold_idx = int(np.argmax(fold_f1s))
last_va = list(skf.split(X_core, y_core))[best_fold_idx][1]
res_cal = full_eval(best_model, X_core[last_va], y_core[last_va], "BiLSTM-S27-cal")
T_opt, ece_pre, ece_post = calibrate(res_cal["logits"], res_cal["true"])
print(f"  ECE: {ece_pre:.4f} → {ece_post:.4f}  (T*={T_opt:.3f})")

print("\nD. HE3 Generalización:")
he3 = he3_eval(best_model, res_te, X_ghold, y_ghold, T_opt)

lat = export_onnx(best_model, "bilstm_s27.onnx")

ckpt = {
    "model_state":      {k: v.cpu() for k, v in best_model.state_dict().items()},
    "architecture":     "LSPLSTMBidirS27",
    "normalize_input":  True,
    "label2idx":        label2idx,
    "idx2label":        idx2label,
    "n_classes":        n_classes,
    "n_dims":           N_DIMS,
    "n_frames":         N_FRAMES,
    "hidden":           hp["hidden"],
    "n_layers":         hp["n_layers"],
    "dropout":          hp["dropout"],
    "temperature":      T_opt,
    "f1_val_mean":      mean_f1,
    "f1_val_std":       std_f1,
    "f1_test":          res_te["f1_macro"],
    "acc_test":         res_te["acc"],
    "top3_test":        res_te["top3"],
    "top5_test":        res_te["top5"],
    "ece_before":       ece_pre,
    "ece_after":        ece_post,
    "fold_f1s":         fold_f1s,
    "he3":              he3,
    "latencia_onnx_ms": lat,
    "sprint":           "S27",
}
torch.save(ckpt, CKPT_DIR / "bilstm_s27.pt")
print(f"  ✅ checkpoints/bilstm_s27.pt")

with open(DATA_DIR / "s27_label2idx.json", "w", encoding="utf-8") as f:
    json.dump(label2idx, f, ensure_ascii=False, indent=2)
with open(DATA_DIR / "s27_idx2label.json", "w", encoding="utf-8") as f:
    json.dump({str(k): v for k, v in idx2label.items()}, f, ensure_ascii=False, indent=2)
print(f"  ✅ data/s27_label2idx.json  ({n_classes} clases)")

fecha = datetime.date.today().strftime("%Y%m%d")
row = {
    "exp_id":      f"exp_{fecha}_bilstm_s27",
    "sprint":      "S27",
    "modelo":      "BiLSTM-S27",
    "features":    "pose+rhand+lhand(150dims×30fr)+per-sample-zscore+s17-grupos-fix",
    "hp_resumen":  (f"hidden={hp['hidden']},n_layers={hp['n_layers']},"
                    f"dropout={hp['dropout']:.2f},lr={hp['lr']:.2e}"),
    "f1_val_mean": f"{mean_f1:.4f}",
    "f1_val_std":  f"{std_f1:.4f}",
    "f1_test":     f"{res_te['f1_macro']:.4f}",
    "acc_test":    f"{res_te['acc']:.4f}",
    "tiempo_s":    f"{int(total_min*60)}",
    "latencia_ms": f"{lat:.1f}",
    "split":       f"StratifiedKFold({N_FOLDS})+GroupHold",
    "seed":        str(SEED),
    "n_classes":   str(n_classes),
    "notas":       (f"Dataset S17 cross-source-fix {len(X_all)} muestras {n_classes} clases "
                    f"per-sample-zscore aug-agresivo; warmstart S13; "
                    f"ECE {ece_pre:.3f}→{ece_post:.3f}(T={T_opt:.2f}); "
                    f"Top3={res_te['top3']:.4f} Top5={res_te['top5']:.4f}; "
                    f"HE3={'PASA' if he3['passed'] else 'FALLA'}; "
                    f"ΔF1={he3['delta_f1']:.4f} PSI={he3['psi']:.4f}"),
}

runs_path = LOGS_DIR / "runs.csv"
fieldnames = list(row.keys())
if runs_path.exists():
    with open(runs_path, "r", newline="", encoding="utf-8") as f:
        fieldnames = csv.DictReader(f).fieldnames or fieldnames

with open(runs_path, "a", newline="", encoding="utf-8") as f:
    csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore").writerow(row)
print(f"  ✅ logs/runs.csv actualizado")

print(f"\n{'='*65}")
print("[TRAIN S27 COMPLETADO]")
print(f"{'='*65}")
print(f"  BiLSTM S27  F1-test : {res_te['f1_macro']:.4f}")
print(f"  BiLSTM S26  F1-test : 0.4098  (S16, HE3 FALLA ΔF1=0.1663)")
print(f"  BiLSTM S13  F1-test : 0.3696  (mejor histórico — con lsa64)")
print(f"  Objetivo            : F1-test ≥ 0.35  Y  HE3 PASA")
objetivo_f1  = res_te['f1_macro'] >= 0.35
objetivo_he3 = he3['passed']
print(f"  F1 ≥ 0.35           : {'✅' if objetivo_f1  else '❌'}  ({res_te['f1_macro']:.4f})")
print(f"  HE3                 : {'✅ PASA' if objetivo_he3 else '❌ FALLA'}")
print(f"  OBJETIVO LOGRADO    : {'✅ SÍ' if (objetivo_f1 and objetivo_he3) else '❌ NO'}")
print(f"  Top-3 test          : {res_te['top3']:.4f}")
print(f"  Top-5 test          : {res_te['top5']:.4f}")
print(f"  ΔF1 (HE3)           : {he3['delta_f1']:.4f}  (umbral ≤ 0.15)")
print(f"  PSI                 : {he3['psi']:.4f}  (umbral < 0.20)")
print(f"  KS p-value          : {he3['ks_pval']:.4f}  (umbral > 0.05)")
