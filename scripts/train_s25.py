"""
train_s25.py — Sprint 25: Source-level DANN (6 dominios)

Análisis S24: el ratio ΔF1/F1-test ≈ 0.92 es CONSTANTE en S19-S24 para cualquier
representación (posición, velocidad, manos-only). Es una propiedad estructural:

  - 7 clases del holdout con core_n=0 → F1 holdout=0 siempre
  - 10 HISTORIAS_VINETAS: cross-source (dgi156↔vineta), F1 ≈ 0.044 actualmente
  - S24 Top-5 holdout=0.464 > Top-5 test=0.441 → el modelo SÍ coloca la clase
    correcta en top-5 para holdout, pero el ranking final falla

Estrategia S25: Source-level DANN con 6 dominios (dgi156, vineta, abecedario,
aec, pucp305, glosa). Diferencia clave vs S20 (DANN con 71 grupos):
  - S20: 71 grupos individuales → señal adversarial demasiado granular
  - S25: 6 fuentes → invarianza a nivel de FUENTE de grabación
  - Para HISTORIAS_VINETAS: si el encoder no puede distinguir dgi156 de vineta,
    debería generalizar cross-source en el holdout

Implementación (Ganin et al. 2016):
  - GradientReversalLayer en el context embedding
  - DomainClassifier(6 clases) entrenado con CE adversarial
  - LAMBDA_D=0.5, ALPHA ramp desde 0→1.0 después de WARMUP_EP=10 epochs
  - Base: misma arquitectura y features que S19 (global z-score, 150D)

Arquitectura:   proj(150→128) → LN → BiLSTM(128,256) → TempAttn → head(sign)
                                                                  ↗ GRL → domain_head(6)
Checkpoints:    bilstm_s25.pt / bilstm_s25.onnx
"""

import json, time, math, warnings, pathlib, csv, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# DANN hyperparams
N_DOMAINS  = 6
SRC_TO_DOM = {'dgi156': 0, 'vineta': 1, 'abecedario': 2,
              'aec': 3, 'pucp305': 4, 'glosa': 5}
LAMBDA_D   = 0.5   # peso del loss adversarial
ALPHA_MAX  = 1.0
WARMUP_EP  = 10

S13_BEST = {
    "hidden": 256, "n_layers": 1, "dropout": 0.20,
    "lr": 1.709e-3, "wd": 1.296e-4, "ls": 0.15,
}

N_EPOCHS_FINAL = 100
PATIENCE_FINAL = 15
N_FOLDS        = 5

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--min-muestras", type=int, default=15)
_ap.add_argument("--skip-hpo",     action="store_true")
_args   = _ap.parse_args()
MIN_SAMPLES = _args.min_muestras
SKIP_HPO    = _args.skip_hpo

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device  : {DEVICE}")
print(f"Sprint 25 — BiLSTM Source-DANN (6 dom) | LSP+AEC | min_muestras={MIN_SAMPLES}")
print("=" * 65)


# ── Normalización ─────────────────────────────────────────────────────────────

def normalize_sample(x: np.ndarray) -> np.ndarray:
    mu = x.mean(); std = x.std()
    if std < 1e-8: return x
    return ((x - mu) / std).astype(np.float32)

def normalize_batch(X: np.ndarray) -> np.ndarray:
    return np.stack([normalize_sample(X[i]) for i in range(len(X))])


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_dataset():
    s15_path = DATA_DIR / "dataset_s15.npz"
    if not s15_path.exists():
        raise FileNotFoundError("dataset_s15.npz no encontrado")
    data        = np.load(s15_path)
    X_raw       = data["X"]
    y_raw       = data["y"]
    groups      = data["groups"]
    sources_raw = np.load(DATA_DIR / "s15_sources.npy", allow_pickle=True)

    with open(DATA_DIR / "s15_label2idx.json", encoding="utf-8") as f:
        label2idx = json.load(f)
    idx2label = {int(v): k for k, v in label2idx.items()}

    counts = Counter(y_raw.tolist())
    keep   = np.array([counts[int(v)] >= MIN_SAMPLES for v in y_raw])
    X_raw, y_raw, groups = X_raw[keep], y_raw[keep], groups[keep]
    sources_raw = sources_raw[keep]

    # Domain labels (fuente → int)
    domains_raw = np.array([SRC_TO_DOM.get(str(s), 0) for s in sources_raw], dtype=np.int64)

    print("  Aplicando normalización por muestra (z-score global)...")
    X = normalize_batch(X_raw)

    old_ids  = sorted(set(y_raw.tolist()))
    remap    = {old: new for new, old in enumerate(old_ids)}
    y        = np.array([remap[int(v)] for v in y_raw], dtype=np.int64)
    n_cl     = len(old_ids)
    idx2label = {remap[o]: idx2label.get(o, str(o)) for o in old_ids}
    label2idx = {v: k for k, v in idx2label.items()}

    print(f"\nDataset S25 (LSP+AEC, min{MIN_SAMPLES}, source-DANN):")
    print(f"  Muestras  : {len(X)}  |  Clases activas: {n_cl}")
    cnt = Counter(y.tolist()); v = sorted(cnt.values())
    print(f"  Samples/cls: min={v[0]} max={v[-1]} mean={np.mean(v):.1f} median={np.median(v):.0f}")
    print(f"  Dominios  : {Counter(domains_raw.tolist())} (fuentes)")
    return X, y, domains_raw, groups, sources_raw, n_cl, label2idx, idx2label


X_all, y_all, dom_all, groups_all, sources_all, n_classes, label2idx, idx2label = load_dataset()

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
tv_idx, te_idx = next(sss.split(X_all, y_all))
X_tv, y_tv, g_tv = X_all[tv_idx], y_all[tv_idx], groups_all[tv_idx]
dom_tv           = dom_all[tv_idx]
X_te, y_te       = X_all[te_idx], y_all[te_idx]
print(f"  Train+Val : {len(X_tv)}  |  Test holdout: {len(X_te)}")

sources_tv = sources_all[tv_idx]
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
core_idx, gh_idx = next(gss.split(X_tv, y_tv, groups=g_tv))
X_core,  y_core  = X_tv[core_idx], y_tv[core_idx]
dom_core         = dom_tv[core_idx]
g_core           = g_tv[core_idx]
X_ghold, y_ghold = X_tv[gh_idx],   y_tv[gh_idx]
print(f"  Core train: {len(X_core)}  |  Group holdout HE3: {len(X_ghold)}")
print(f"  Fuentes en group-holdout: {Counter(sources_tv[gh_idx].tolist())}")


# ── Augmentation ─────────────────────────────────────────────────────────────

def time_warp(x, sigma=0.04, knots=4):
    T = x.shape[0]; orig = np.linspace(0, T - 1, T)
    warp = np.clip(np.sort(orig + np.random.randn(T) * sigma * T / knots), 0, T - 1)
    out = np.zeros_like(x)
    for d in range(x.shape[1]): out[:, d] = np.interp(orig, warp, x[:, d])
    return out.astype(np.float32)


class SignDataset(Dataset):
    def __init__(self, X, y, dom, augment=False):
        self.X       = X.copy()
        self.y       = torch.from_numpy(y).long()
        self.dom     = torch.from_numpy(dom).long()
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
        return torch.from_numpy(x).float(), self.y[i], self.dom[i]


def make_loaders(X_tr, y_tr, dom_tr, X_val, y_val, dom_val, batch=BATCH):
    cc  = Counter(y_tr.tolist())
    w   = [1.0 / cc[int(c)] for c in y_tr]
    smp = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    dl_tr  = DataLoader(SignDataset(X_tr, y_tr, dom_tr, augment=True),
                        batch_size=batch, sampler=smp, num_workers=0)
    dl_val = DataLoader(SignDataset(X_val, y_val, dom_val, augment=False),
                        batch_size=batch, shuffle=False, num_workers=0)
    return dl_tr, dl_val


def make_criterion(y_tr, label_smoothing):
    cc = Counter(y_tr.tolist())
    cw = torch.tensor(
        [len(y_tr) / (n_classes * cc.get(i, 1)) for i in range(n_classes)],
        dtype=torch.float32,
    ).to(DEVICE)
    return nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)


# ── Arquitectura DANN ────────────────────────────────────────────────────────

class GRL(torch.autograd.Function):
    """Gradient Reversal Layer (Ganin 2016)."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


class TemporalAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attn = nn.Linear(hidden * 2, 1)

    def forward(self, h):
        w = torch.softmax(self.attn(h), dim=1)
        return (w * h).sum(dim=1)


class LSPBiLSTM_DANN25(nn.Module):
    def __init__(self, n_dims, n_classes, hidden=256, n_layers=1, dropout=0.20,
                 n_domains=N_DOMAINS):
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
        # Domain classifier (6 fuentes: dgi156/vineta/abecedario/aec/pucp305/glosa)
        self.domain_head = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_domains),
        )

    def encode(self, x):
        x    = self.proj(x)
        h, _ = self.lstm(x)
        return self.attn(h)  # [B, hidden*2]

    def forward(self, x, alpha=None):
        ctx          = self.encode(x)
        cls_logits   = self.head(ctx)
        if alpha is not None:
            rev = GRL.apply(ctx, alpha)
            return cls_logits, self.domain_head(rev)
        return cls_logits


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def dann_alpha(epoch, n_epochs):
    """Rampa suave del factor alpha (Ganin 2016)."""
    if epoch <= WARMUP_EP:
        return 0.0
    p = (epoch - WARMUP_EP) / max(n_epochs - WARMUP_EP, 1)
    return ALPHA_MAX * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


# ── Training con DANN ────────────────────────────────────────────────────────

def train_one_run(X_tr, y_tr, dom_tr, X_val, y_val, dom_val, model_fn,
                  lr, wd, label_smoothing, n_epochs, patience, trial=None):
    dl_tr, dl_val = make_loaders(X_tr, y_tr, dom_tr, X_val, y_val, dom_val)
    model     = model_fn().to(DEVICE)
    criterion = make_criterion(y_tr, label_smoothing)
    dom_crit  = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=25, T_mult=2, eta_min=lr * 0.01)

    best_f1, best_state, pat_cnt = 0.0, None, 0
    for epoch in range(1, n_epochs + 1):
        alpha = dann_alpha(epoch, n_epochs)
        model.train()
        for xb, yb, db in dl_tr:
            xb, yb, db = xb.to(DEVICE), yb.to(DEVICE), db.to(DEVICE)
            optimizer.zero_grad()
            if alpha > 0:
                cls_log, dom_log = model(xb, alpha=alpha)
                loss = criterion(cls_log, yb) + LAMBDA_D * dom_crit(dom_log, db)
            else:
                loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb, db in dl_val:
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

    model.load_state_dict(best_state)
    return model, best_f1


# ── Evaluación ────────────────────────────────────────────────────────────────

@torch.no_grad()
def full_eval(model, X_eval, y_eval, dom_eval, label="eval"):
    model.to(DEVICE).eval()
    dl = DataLoader(SignDataset(X_eval, y_eval, dom_eval, augment=False),
                    batch_size=BATCH, shuffle=False, num_workers=0)
    all_logits, all_true = [], []
    for xb, yb, db in dl:
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


def ece_score(logits_np, labels_np, n_bins=15):
    logits = torch.from_numpy(logits_np); labels = torch.from_numpy(labels_np).long()
    probs  = torch.softmax(logits, dim=1); confs, preds = probs.max(dim=1)
    accs   = preds.eq(labels); ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins; mask = (confs > lo) & (confs <= hi)
        if mask.sum() == 0: continue
        ece += mask.float().mean().item() * abs(
            accs[mask].float().mean().item() - confs[mask].float().mean().item())
    return ece


def calibrate(logits_np, labels_np):
    logits_t = torch.from_numpy(logits_np); labels_t = torch.from_numpy(labels_np).long()
    ece_pre  = ece_score(logits_np, labels_np)
    def nll_t(T): return F.cross_entropy(logits_t / T, labels_t).item()
    res   = minimize_scalar(nll_t, bounds=(0.1, 10.0), method="bounded")
    T_opt = float(res.x)
    return T_opt, ece_pre, ece_score(logits_np / T_opt, labels_np)


def he3_eval(model, res_test, X_gh, y_gh, dom_gh, T_opt):
    res_gh   = full_eval(model, X_gh, y_gh, dom_gh, label="group-holdout")
    delta_f1 = abs(res_test["f1_macro"] - res_gh["f1_macro"])
    conf_te  = res_test["probs"].max(axis=1)
    conf_gh  = res_gh["probs"].max(axis=1)
    bins     = np.linspace(0, 1, 11)
    p_ref, _ = np.histogram(conf_te, bins=bins, density=True)
    p_new, _ = np.histogram(conf_gh, bins=bins, density=True)
    p_ref    = np.clip(p_ref / (p_ref.sum() + 1e-8), 1e-8, None)
    p_new    = np.clip(p_new / (p_new.sum() + 1e-8), 1e-8, None)
    psi      = float(np.sum((p_ref - p_new) * np.log(p_ref / p_new)))
    ks_stat, ks_pval = ks_2samp(conf_te, conf_gh)
    ok_f1  = delta_f1 <= 0.15; ok_psi = psi < 0.20; ok_ks = ks_pval > 0.05
    passed = ok_f1 and ok_psi and ok_ks
    print(f"  HE3 → ΔF1={delta_f1:.4f} {'✅' if ok_f1 else '❌'}  "
          f"PSI={psi:.4f} {'✅' if ok_psi else '❌'}  "
          f"KS p={ks_pval:.4f} {'✅' if ok_ks else '❌'}  "
          f"→ {'PASA ✅' if passed else 'FALLA ❌'}")
    return {"delta_f1": delta_f1, "psi": psi, "ks_stat": ks_stat,
            "ks_pval": ks_pval, "passed": passed, "f1_ghold": res_gh["f1_macro"]}


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
        print(f"  ONNX error: {e}"); return 0.0


# ════════════════════════════════════════════════════════════════════════════
# Entrenamiento KFold(5)
# ════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print(f"BLOQUE A — skip-hpo: usando S13-best + DANN(lambda={LAMBDA_D}, alpha_max={ALPHA_MAX})")
print(f"{'='*65}")
hp = {k: S13_BEST[k] for k in ("hidden", "n_layers", "dropout", "lr", "wd", "ls")}
print(f"  HPs: {hp}")

print(f"\nB. KFold({N_FOLDS}) final BiLSTM S25 — Source DANN (6 dominios)")


def make_best():
    return LSPBiLSTM_DANN25(N_DIMS, n_classes,
                            hidden=hp["hidden"], n_layers=hp["n_layers"],
                            dropout=hp["dropout"])


print(f"  Params: {count_params(make_best()):,}")

skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_f1s, best_f1, best_model = [], 0.0, None
t_final = time.time()

for fi, (tr_i, va_i) in enumerate(skf.split(X_core, y_core), 1):
    m, f1 = train_one_run(
        X_core[tr_i], y_core[tr_i], dom_core[tr_i],
        X_core[va_i], y_core[va_i], dom_core[va_i],
        model_fn=make_best,
        lr=hp["lr"], wd=hp["wd"], label_smoothing=hp["ls"],
        n_epochs=N_EPOCHS_FINAL, patience=PATIENCE_FINAL,
    )
    fold_f1s.append(f1)
    print(f"  Fold {fi}: F1-val={f1:.4f}")
    if f1 > best_f1:
        best_f1, best_model = f1, m

total_min = (time.time() - t_final) / 60
mean_f1   = float(np.mean(fold_f1s))
std_f1    = float(np.std(fold_f1s))
print(f"\n  KFold F1: {mean_f1:.4f} ± {std_f1:.4f}  ({total_min:.1f} min)")


# ── Test & HE3 ───────────────────────────────────────────────────────────────

dom_te  = dom_all[te_idx]
dom_gh  = dom_tv[gh_idx]

print("\nC. Evaluación en test holdout:")
res_te = full_eval(best_model, X_te, y_te, dom_te, "BiLSTM-S25-test")

best_fold_idx = int(np.argmax(fold_f1s))
last_va       = list(skf.split(X_core, y_core))[best_fold_idx][1]
res_cal = full_eval(best_model, X_core[last_va], y_core[last_va],
                    dom_core[last_va], "BiLSTM-S25-cal")
T_opt, ece_pre, ece_post = calibrate(res_cal["logits"], res_cal["true"])
print(f"  ECE: {ece_pre:.4f} → {ece_post:.4f}  (T*={T_opt:.3f})")

print("\nD. HE3 Generalización:")
he3 = he3_eval(best_model, res_te, X_ghold, y_ghold, dom_gh, T_opt)

lat = export_onnx(best_model, "bilstm_s25.onnx")

ckpt = {
    "model_state":      {k: v.cpu() for k, v in best_model.state_dict().items()},
    "architecture":     "LSPBiLSTM_DANN25",
    "normalize_input":  "global_zscore_150d",
    "label2idx":        label2idx,
    "idx2label":        idx2label,
    "n_classes":        n_classes,
    "n_dims":           N_DIMS,
    "n_frames":         N_FRAMES,
    "hidden":           hp["hidden"],
    "n_layers":         hp["n_layers"],
    "dropout":          hp["dropout"],
    "lr":               hp["lr"],
    "wd":               hp["wd"],
    "label_smoothing":  hp["ls"],
    "temperature":      T_opt,
    "lambda_d":         LAMBDA_D,
    "alpha_max":        ALPHA_MAX,
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
    "sprint":           "S25",
}
torch.save(ckpt, CKPT_DIR / "bilstm_s25.pt")
print(f"  ✅ checkpoints/bilstm_s25.pt")

with open(DATA_DIR / "s25_label2idx.json", "w", encoding="utf-8") as f:
    json.dump(label2idx, f, ensure_ascii=False, indent=2)
with open(DATA_DIR / "s25_idx2label.json", "w", encoding="utf-8") as f:
    json.dump({str(k): v for k, v in idx2label.items()}, f, ensure_ascii=False, indent=2)
print(f"  ✅ data/s25_label2idx.json  ({n_classes} clases LSP)")

fecha = datetime.date.today().strftime("%Y%m%d")
row = {
    "exp_id":      f"exp_{fecha}_bilstm_s25",
    "sprint":      "S25",
    "modelo":      "BiLSTM-S25-srcDANN",
    "features":    f"global-zscore(150d×30fr)+DANN(6src,lam={LAMBDA_D})",
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
    "notas":       (f"source-DANN 6dom lam={LAMBDA_D} alpha_max={ALPHA_MAX} min{MIN_SAMPLES}; "
                    f"HE3-holdout F1={he3['f1_ghold']:.4f}; "
                    f"ECE {ece_pre:.3f}→{ece_post:.3f}(T={T_opt:.2f}); "
                    f"Top3={res_te['top3']:.4f} Top5={res_te['top5']:.4f}; "
                    f"HE3={'PASA' if he3['passed'] else 'FALLA'}; "
                    f"ΔF1={he3['delta_f1']:.4f} PSI={he3['psi']:.4f}"),
}

runs_path  = LOGS_DIR / "runs.csv"
fieldnames = list(row.keys())
if runs_path.exists():
    with open(runs_path, "r", newline="", encoding="utf-8") as f:
        fieldnames = csv.DictReader(f).fieldnames or fieldnames
with open(runs_path, "a", newline="", encoding="utf-8") as f:
    csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore").writerow(row)
print(f"  ✅ logs/runs.csv actualizado")

print(f"\n{'='*65}")
print("[TRAIN S25 COMPLETADO]")
print(f"{'='*65}")
print(f"  BiLSTM S25 srcDANN   F1-test : {res_te['f1_macro']:.4f}")
print(f"  BiLSTM S24 hand-only F1-test : 0.2551")
print(f"  BiLSTM S19 global-z  F1-test : 0.3357  (mejor F1)")
print(f"  BiLSTM S13           F1-test : 0.3696  (mejor histórico)")
print(f"  Δ vs S19                     : {(res_te['f1_macro']/0.3357-1)*100:+.1f}%")
print(f"  Top-3 test                   : {res_te['top3']:.4f}")
print(f"  Top-5 test                   : {res_te['top5']:.4f}")
print(f"  HE3 holdout F1               : {he3['f1_ghold']:.4f}  (S19:0.026 S24:0.020 S23:0.012)")
print(f"  HE3 ΔF1                      : {he3['delta_f1']:.4f}  (umbral ≤0.15)")
print(f"  HE3                          : {'✅ PASA' if he3['passed'] else '❌ FALLA'}")
print(f"  KS p-value                   : {he3['ks_pval']:.4f}")
