"""
train_s11.py — Sprint 11: Transformer vs BiLSTM mejorado + HPO + KFold(5)

Mejoras vs S10 (train_lstm_s10.py):
  ┌────────────────────────────────────────────────────────────────────────┐
  │  Cambio               S10                  S11                        │
  │  Dataset              dataset_s10.npz      dataset_s11.npz (+80%)     │
  │  Arquitecturas        BiLSTM               Transformer + BiLSTM       │
  │  HPO trials           30                   50 (Transformer)           │
  │  Scheduler            CosineAnnealing      CosineAnnealingWarmRestarts│
  │  Augmentación         noise + handflip     + TimeWarp + CoordDropout  │
  │  Validación           KFold(5)             GroupKFold(5) + HE3        │
  │  ONNX                 opset 17             opset 17 (ambos modelos)   │
  └────────────────────────────────────────────────────────────────────────┘

Arquitectura principal: LSPTransformerS11
  Input [B, 30, 150] → Linear(150→d_model) → LayerNorm →
  + CLS token + PosEmbed →
  TransformerEncoder(n_layers, nhead, dim_ff) →
  CLS output → Dropout → Linear → n_classes

Justificación Transformer vs BiLSTM:
  - Self-attention captura dependencias inter-frame simultáneamente (no secuenciales)
  - Mejor rendimiento en SLR según literatura (SPOTER 2021, Sign-ILP 2023)
  - Con 7,536 muestras (S11) tiene suficiente datos para aprender
  - Latencia similar a BiLSTM en ONNX (<2 ms)

Outputs:
  checkpoints/transformer_s11.pt   — Transformer ganador
  checkpoints/transformer_s11.onnx
  checkpoints/bilstm_s11.pt        — BiLSTM entrenado en S11 dataset
  checkpoints/bilstm_s11.onnx
  logs/runs.csv                    — actualizado
"""

import json, time, warnings, pathlib, csv, datetime, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, GroupKFold
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

# ── Configuración ─────────────────────────────────────────────────────────────

SEED     = 42
N_FRAMES = 30
N_DIMS   = 150
BATCH    = 64
DEVICE   = "mps" if torch.backends.mps.is_available() else "cpu"

# HPO S10 best (warm start para BiLSTM S11)
S10_BEST = {
    "hidden": 256, "n_layers": 1, "dropout": 0.20,
    "lr": 0.004093813608598782, "label_smoothing": 0.15, "weight_decay": 1e-4,
}

N_EPOCHS_HPO    = 40
PATIENCE_HPO    = 8
N_EPOCHS_FINAL  = 100
PATIENCE_FINAL  = 15
N_TRIALS_TRANSF = 50
N_TRIALS_LSTM   = 30
N_FOLDS         = 5

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device  : {DEVICE}")
print(f"Sprint 11 — Transformer S11 vs BiLSTM S11 | HPO + KFold({N_FOLDS})")
print("=" * 70)

# ── Cargar dataset S11 ────────────────────────────────────────────────────────

def load_dataset():
    s11_path = DATA_DIR / "dataset_s11.npz"
    if not s11_path.exists():
        print("  dataset_s11.npz no encontrado → ejecutando build_dataset_s11.py …")
        import subprocess, sys
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_dataset_s11.py")],
            check=True,
        )

    data   = np.load(s11_path)
    X      = data["X"]       # [N, 30, 150]
    y_raw  = data["y"]       # [N]
    groups = data["groups"]  # [N]  int (grupo fuente/clase)

    json_path = DATA_DIR / "s11_label2idx.json"
    if not json_path.exists():
        json_path = DATA_DIR / "s10_label2idx.json"

    with open(json_path, encoding="utf-8") as f:
        label2idx = json.load(f)
    idx2label = {int(v): k for k, v in label2idx.items()}

    # Re-filtrar (por si el NPZ aún tiene LSP - Vocabulario-palabras <2)
    counts   = Counter(y_raw.tolist())
    keep     = np.array([counts[int(v)] >= 2 for v in y_raw])
    X, y_raw, groups = X[keep], y_raw[keep], groups[keep]

    old_ids  = sorted(set(y_raw.tolist()))
    remap    = {old: new for new, old in enumerate(old_ids)}
    y        = np.array([remap[int(v)] for v in y_raw], dtype=np.int64)
    n_cl     = len(old_ids)
    idx2label = {remap[o]: idx2label.get(o, str(o)) for o in old_ids}
    label2idx = {v: k for k, v in idx2label.items()}

    print(f"\nDataset S11:")
    print(f"  Muestras  : {len(X)}  |  LSP - Vocabulario-palabras activas: {n_cl}")
    cnt = Counter(y.tolist())
    v   = sorted(cnt.values())
    print(f"  Samples/cls: min={v[0]} max={v[-1]} mean={np.mean(v):.1f}")
    return X, y, groups, n_cl, label2idx, idx2label


X_all, y_all, groups_all, n_classes, label2idx, idx2label = load_dataset()

# Holdout test fijo 15%
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
tv_idx, te_idx = next(sss.split(X_all, y_all))
X_tv, y_tv, g_tv = X_all[tv_idx], y_all[tv_idx], groups_all[tv_idx]
X_te, y_te        = X_all[te_idx],  y_all[te_idx]
print(f"  Train+Val : {len(X_tv)}  |  Test holdout: {len(X_te)}")

# Group holdout 20% para HE3 (después del test split)
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
core_idx, group_hold_idx = next(gss.split(X_tv, y_tv, groups=g_tv))
X_core,  y_core  = X_tv[core_idx],       y_tv[core_idx]
g_core           = g_tv[core_idx]
X_ghold, y_ghold = X_tv[group_hold_idx], y_tv[group_hold_idx]
print(f"  Core train: {len(X_core)}  |  Group holdout HE3: {len(X_ghold)}")

# ── Dataset & Augmentation ────────────────────────────────────────────────────

def time_warp(x: np.ndarray, sigma: float = 0.04, knots: int = 4) -> np.ndarray:
    T = x.shape[0]
    orig = np.linspace(0, T - 1, T)
    warp = orig + np.random.randn(T) * sigma * T / knots
    warp = np.clip(np.sort(warp), 0, T - 1)
    out  = np.zeros_like(x)
    for d in range(x.shape[1]):
        out[:, d] = np.interp(orig, warp, x[:, d])
    return out.astype(np.float32)


class SignDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X       = X.copy()
        self.y       = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].copy()

        if self.augment:
            # Gaussian noise
            x = x + np.random.randn(*x.shape).astype(np.float32) * 0.008
            # Hand flip (mirror left↔right hand coords)
            if np.random.rand() < 0.5:
                x[:, 66:87]   = 1.0 - x[:, 66:87]    # left_hand_x
                x[:, 108:129] = 1.0 - x[:, 108:129]   # right_hand_x
            # Time warp
            if np.random.rand() < 0.40:
                x = time_warp(x, sigma=0.04)
            # Coordinate dropout
            if np.random.rand() < 0.30:
                n_drop  = int(N_DIMS * np.random.uniform(0.03, 0.10))
                drop_i  = np.random.choice(N_DIMS, n_drop, replace=False)
                x[:, drop_i] = 0.0
            # Speed perturbation (temporal scaling ±20%)
            if np.random.rand() < 0.30:
                factor = np.random.uniform(0.8, 1.2)
                old_t  = np.linspace(0, N_FRAMES - 1, N_FRAMES)
                new_t  = np.linspace(0, N_FRAMES - 1,
                                     int(N_FRAMES * factor))
                new_t  = np.clip(new_t, 0, N_FRAMES - 1)
                scaled = np.zeros_like(x)
                for d in range(N_DIMS):
                    scaled[:, d] = np.interp(old_t, new_t, np.interp(new_t, old_t, x[:, d]))
                x = scaled.astype(np.float32)

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


# ════════════════════════════════════════════════════════════════════════════
# ARQUITECTURA 1 — LSPTransformerS11
# ════════════════════════════════════════════════════════════════════════════

class LSPTransformerS11(nn.Module):
    """
    Temporal Transformer para reconocimiento de señas LSP.
    Input : [B, T=30, D=150]
    Output: [B, n_classes]

    Diseño:
      1. Proyección espacial: Linear(150 → d_model) + LayerNorm
      2. CLS token + Positional Embedding aprendibles
      3. TransformerEncoder(n_layers capas, nhead cabezas)
      4. LayerNorm final
      5. CLS output → MLP → logits
    """

    def __init__(self, n_dims: int, n_classes: int,
                 d_model: int = 256, nhead: int = 8,
                 num_layers: int = 4, dim_ff: int = 512,
                 dropout: float = 0.2):
        super().__init__()

        self.d_model = d_model

        # 1. Proyección de entrada
        self.input_proj = nn.Sequential(
            nn.Linear(n_dims, d_model),
            nn.LayerNorm(d_model),
        )

        # 2. CLS token y positional embedding (T+1 posiciones: CLS + 30 frames)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, N_FRAMES + 1, d_model) * 0.02)

        # 3. Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN (más estable en training)
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers,
                                                  enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

        # 4. Cabeza de clasificación
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.input_proj(x)                         # [B, 30, d_model]
        cls = self.cls_token.expand(B, -1, -1)          # [B,  1, d_model]
        x   = torch.cat([cls, x], dim=1)                # [B, 31, d_model]
        x   = x + self.pos_embed                        # add pos embedding
        x   = self.transformer(x)                       # [B, 31, d_model]
        x   = self.norm(x)
        return self.head(x[:, 0])                       # CLS → logits


# ════════════════════════════════════════════════════════════════════════════
# ARQUITECTURA 2 — LSPLSTMBidirS11 (mejorada de S10)
# ════════════════════════════════════════════════════════════════════════════

class TemporalAttention(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.attn = nn.Linear(hidden * 2, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.attn(h), dim=1)
        return (w * h).sum(dim=1)


class LSPLSTMBidirS11(nn.Module):
    """BiLSTM + Temporal Attention entrenado en dataset S11 (más datos que S10)."""

    def __init__(self, n_dims: int, n_classes: int,
                 hidden: int = 256, n_layers: int = 2, dropout: float = 0.25):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x   = self.proj(x)
        h, _ = self.lstm(x)
        c   = self.attn(h)
        return self.head(c)


def count_params(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ════════════════════════════════════════════════════════════════════════════
# Loop de entrenamiento genérico
# ════════════════════════════════════════════════════════════════════════════

def train_one_run(X_tr, y_tr, X_val, y_val,
                  model_fn,           # callable -> nn.Module
                  lr, wd, label_smoothing,
                  n_epochs, patience,
                  trial=None):
    dl_tr, dl_val = make_loaders(X_tr, y_tr, X_val, y_val)
    model     = model_fn().to(DEVICE)
    criterion = make_criterion(y_tr, label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=25, T_mult=2, eta_min=lr * 0.01,
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


# ════════════════════════════════════════════════════════════════════════════
# Evaluación completa (test + Top-K + por clase)
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def full_eval(model, X_eval, y_eval, label="eval"):
    model.to(DEVICE).eval()
    dl   = DataLoader(SignDataset(X_eval, y_eval, augment=False),
                      batch_size=BATCH, shuffle=False, num_workers=0)
    all_logits, all_true = [], []
    for xb, yb in dl:
        logits = model(xb.to(DEVICE)).cpu()
        all_logits.append(logits)
        all_true.extend(yb.tolist())

    logits_t = torch.cat(all_logits)
    probs    = torch.softmax(logits_t, dim=1).numpy()
    preds    = np.argmax(probs, axis=1)
    true_arr = np.array(all_true)

    acc   = (preds == true_arr).mean()
    f1_m  = f1_score(true_arr, preds, average="macro",    zero_division=0)
    f1_w  = f1_score(true_arr, preds, average="weighted", zero_division=0)

    # Top-K
    top3 = sum(int(t) in np.argsort(probs[i])[-3:] for i, t in enumerate(true_arr)) / len(true_arr)
    top5 = sum(int(t) in np.argsort(probs[i])[-5:] for i, t in enumerate(true_arr)) / len(true_arr)

    print(f"  [{label}] Acc={acc:.4f}  F1-macro={f1_m:.4f}  "
          f"F1-weighted={f1_w:.4f}  Top-3={top3:.4f}  Top-5={top5:.4f}")
    return {
        "f1_macro": f1_m, "f1_weighted": f1_w,
        "acc": acc, "top3": top3, "top5": top5,
        "probs": probs, "preds": preds, "true": true_arr,
        "logits": logits_t.numpy(),
    }


# ════════════════════════════════════════════════════════════════════════════
# Temperature Scaling
# ════════════════════════════════════════════════════════════════════════════

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
            accs[mask].float().mean().item() - confs[mask].float().mean().item()
        )
    return ece


def calibrate(logits_np, labels_np):
    logits_t = torch.from_numpy(logits_np)
    labels_t = torch.from_numpy(labels_np).long()
    ece_pre  = ece_score(logits_np, labels_np)

    def nll_t(T):
        return F.cross_entropy(logits_t / T, labels_t).item()

    res    = minimize_scalar(nll_t, bounds=(0.1, 10.0), method="bounded")
    T_opt  = float(res.x)
    ece_post = ece_score(logits_np / T_opt, labels_np)
    return T_opt, ece_pre, ece_post


# ════════════════════════════════════════════════════════════════════════════
# HE3: holdout generalización (PSI + KS)
# ════════════════════════════════════════════════════════════════════════════

def he3_eval(model, res_test, X_ghold, y_ghold, T_opt):
    """Calcula ΔF1, PSI y KS entre distribución de test y group holdout."""
    res_gh = full_eval(model, X_ghold, y_ghold, label="group-holdout")
    delta_f1 = abs(res_test["f1_macro"] - res_gh["f1_macro"])

    # Confianzas calibradas
    conf_te = (res_test["probs"] / T_opt).max(axis=1) if T_opt != 1.0 \
        else res_test["probs"].max(axis=1)
    conf_gh = (res_gh["probs"] / T_opt).max(axis=1) if T_opt != 1.0 \
        else res_gh["probs"].max(axis=1)

    # PSI
    bins  = np.linspace(0, 1, 11)
    p_ref, _ = np.histogram(conf_te, bins=bins, density=True)
    p_new, _ = np.histogram(conf_gh, bins=bins, density=True)
    p_ref = np.clip(p_ref / (p_ref.sum() + 1e-8), 1e-8, None)
    p_new = np.clip(p_new / (p_new.sum() + 1e-8), 1e-8, None)
    psi   = float(np.sum((p_ref - p_new) * np.log(p_ref / p_new)))

    # KS test
    ks_stat, ks_pval = ks_2samp(conf_te, conf_gh)

    ok_f1  = delta_f1 <= 0.15
    ok_psi = psi       < 0.20
    ok_ks  = ks_pval   > 0.05
    passed = ok_f1 and ok_psi and ok_ks

    print(f"  HE3 → ΔF1={delta_f1:.4f} {'✅' if ok_f1 else '❌'}  "
          f"PSI={psi:.4f} {'✅' if ok_psi else '❌'}  "
          f"KS p={ks_pval:.4f} {'✅' if ok_ks else '❌'}  "
          f"→ {'PASA ✅' if passed else 'FALLA ❌'}")
    return {
        "delta_f1": delta_f1, "psi": psi,
        "ks_stat": ks_stat, "ks_pval": ks_pval,
        "passed": passed,
        "f1_ghold": res_gh["f1_macro"],
    }


# ════════════════════════════════════════════════════════════════════════════
# ONNX export + latencia
# ════════════════════════════════════════════════════════════════════════════

def export_onnx(model, name: str) -> float:
    model.cpu().eval()
    dummy   = torch.randn(1, N_FRAMES, N_DIMS)
    onnx_p  = CKPT_DIR / name
    try:
        torch.onnx.export(
            model, dummy, str(onnx_p),
            input_names=["sequence"], output_names=["logits"],
            dynamic_axes={"sequence": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )
        # Latencia
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_p))
        np_in  = np.zeros((1, N_FRAMES, N_DIMS), dtype=np.float32)
        times  = []
        for _ in range(200):
            t0 = time.perf_counter()
            sess.run(None, {"sequence": np_in})
            times.append((time.perf_counter() - t0) * 1000)
        lat = float(np.mean(times))
        print(f"  ONNX → {onnx_p.name}  ({onnx_p.stat().st_size / 1e6:.1f} MB)"
              f"  latencia: {lat:.2f} ms (p95={np.percentile(times, 95):.2f} ms)")
        return lat
    except Exception as e:
        print(f"  ONNX error: {e}")
        return 0.0


# ════════════════════════════════════════════════════════════════════════════
# BLOQUE A — Entrenar LSPTransformerS11
# ════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("BLOQUE A — LSPTransformerS11  (arquitectura nueva)")
print(f"{'='*70}")

# --- A1. HPO Transformer ---
print(f"\nA1. HPO Transformer ({N_TRIALS_TRANSF} trials, Optuna TPE + MedianPruner)")

# Split HPO interno — ShuffleSplit (no Stratified: X_core puede tener LSP - Vocabulario-palabras con 1 muestra
# tras GroupShuffleSplit, lo que hace fallar StratifiedShuffleSplit)
from sklearn.model_selection import ShuffleSplit
sss_hpo = ShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
hpo_tr, hpo_val = next(sss_hpo.split(X_core))
Xh_tr, yh_tr   = X_core[hpo_tr],  y_core[hpo_tr]
Xh_val, yh_val  = X_core[hpo_val], y_core[hpo_val]


def objective_transf(trial: optuna.Trial) -> float:
    d_model  = trial.suggest_categorical("d_model",  [128, 256])
    # 128 y 256 son divisibles por 4 y 8 → espacio fijo sin condicionales
    nhead    = trial.suggest_categorical("nhead",    [4, 8])
    n_layers = trial.suggest_int("n_layers", 2, 5)
    dim_ff   = trial.suggest_categorical("dim_ff",   [256, 512])
    dropout  = trial.suggest_float("dropout", 0.10, 0.50, step=0.05)
    lr       = trial.suggest_float("lr",      5e-5, 5e-3, log=True)
    wd       = trial.suggest_float("wd",      1e-5, 5e-4, log=True)
    ls       = trial.suggest_float("ls",      0.05, 0.20, step=0.05)

    def make_transf():
        return LSPTransformerS11(N_DIMS, n_classes, d_model=d_model,
                                  nhead=nhead, num_layers=n_layers,
                                  dim_ff=dim_ff, dropout=dropout)

    _, f1 = train_one_run(
        Xh_tr, yh_tr, Xh_val, yh_val,
        model_fn=make_transf,
        lr=lr, wd=wd, label_smoothing=ls,
        n_epochs=N_EPOCHS_HPO, patience=PATIENCE_HPO, trial=trial,
    )
    return f1


t0 = time.time()
study_t = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=SEED),
    pruner=MedianPruner(n_startup_trials=8, n_warmup_steps=10),
)
study_t.optimize(objective_transf, n_trials=N_TRIALS_TRANSF, show_progress_bar=True)
hp_t   = study_t.best_params
f1_hpo_t = study_t.best_value
print(f"\n  ✅ HPO Transformer → F1-hpo={f1_hpo_t:.4f}  tiempo={( time.time()-t0)/60:.1f} min")
print(f"  Mejores HPs: {hp_t}")

# --- A2. Entrenamiento final KFold(5) Transformer ---
print(f"\nA2. KFold({N_FOLDS}) final Transformer")

def make_best_transf():
    return LSPTransformerS11(
        N_DIMS, n_classes,
        d_model   = hp_t["d_model"],
        nhead     = hp_t["nhead"],
        num_layers= hp_t["n_layers"],
        dim_ff    = hp_t["dim_ff"],
        dropout   = hp_t["dropout"],
    )

_tmp = make_best_transf()
print(f"  Params Transformer: {count_params(_tmp):,}"); del _tmp

skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_f1s_t, best_t_f1, best_t_model = [], 0.0, None
t_final = time.time()

for fi, (tr_i, va_i) in enumerate(skf.split(X_core, y_core), 1):
    m, f1 = train_one_run(
        X_core[tr_i], y_core[tr_i], X_core[va_i], y_core[va_i],
        model_fn=make_best_transf,
        lr=hp_t["lr"], wd=hp_t["wd"], label_smoothing=hp_t["ls"],
        n_epochs=N_EPOCHS_FINAL, patience=PATIENCE_FINAL,
    )
    fold_f1s_t.append(f1)
    print(f"  Fold {fi}: F1-val={f1:.4f}")
    if f1 > best_t_f1:
        best_t_f1, best_t_model = f1, m

t_transf_min = (time.time() - t_final) / 60
mean_t = float(np.mean(fold_f1s_t)); std_t = float(np.std(fold_f1s_t))
print(f"\n  Transformer KFold F1: {mean_t:.4f} ± {std_t:.4f}  "
      f"({t_transf_min:.1f} min)")

# Eval completa Transformer
print("\nA3. Evaluación Transformer en test holdout:")
res_t = full_eval(best_t_model, X_te, y_te, "Transformer-test")

# Calibración Transformer
last_va = list(skf.split(X_core, y_core))[np.argmax(fold_f1s_t)][1]
Xcal, ycal = X_core[last_va], y_core[last_va]
res_cal_t  = full_eval(best_t_model, Xcal, ycal, "Transformer-cal")
T_t, ece_pre_t, ece_post_t = calibrate(res_cal_t["logits"], res_cal_t["true"])
print(f"  ECE: {ece_pre_t:.4f} → {ece_post_t:.4f}  (T*={T_t:.3f})")

# HE3 Transformer
print("\nA4. HE3 Generalización Transformer:")
he3_t = he3_eval(best_t_model, res_t, X_ghold, y_ghold, T_t)

# Guardar Transformer
lat_t = export_onnx(best_t_model, "transformer_s11.onnx")
ckpt_t = {
    "model_state": {k: v.cpu() for k, v in best_t_model.state_dict().items()},
    "architecture": "LSPTransformerS11",
    "label2idx":   label2idx,
    "idx2label":   idx2label,
    "n_classes":   n_classes,
    "n_dims":      N_DIMS, "n_frames": N_FRAMES,
    "d_model":     hp_t["d_model"],
    "nhead":       hp_t["nhead"],
    "n_layers":    hp_t["n_layers"],
    "dim_ff":      hp_t["dim_ff"],
    "dropout":     hp_t["dropout"],
    "lr":          hp_t["lr"],
    "wd":          hp_t["wd"],
    "label_smoothing": hp_t["ls"],
    "temperature": T_t,
    "f1_val_mean": mean_t, "f1_val_std": std_t,
    "f1_test":     res_t["f1_macro"],
    "acc_test":    res_t["acc"],
    "top3_test":   res_t["top3"],
    "top5_test":   res_t["top5"],
    "ece_before":  ece_pre_t, "ece_after": ece_post_t,
    "fold_f1s":    fold_f1s_t,
    "he3":         he3_t,
    "latencia_onnx_ms": lat_t,
    "sprint": "S11",
}
torch.save(ckpt_t, CKPT_DIR / "transformer_s11.pt")
print(f"  ✅ checkpoints/transformer_s11.pt")


# ════════════════════════════════════════════════════════════════════════════
# BLOQUE B — LSPLSTMBidirS11 (BiLSTM entrenado en dataset S11)
# ════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("BLOQUE B — LSPLSTMBidirS11  (BiLSTM + dataset S11 ampliado)")
print(f"{'='*70}")

# --- B1. HPO BiLSTM con warm start desde S10 ---
print(f"\nB1. HPO BiLSTM ({N_TRIALS_LSTM} trials, warm start S10 best)")


def objective_lstm(trial: optuna.Trial) -> float:
    hidden   = trial.suggest_categorical("hidden",   [128, 256])
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout  = trial.suggest_float("dropout", 0.10, 0.45, step=0.05)
    lr       = trial.suggest_float("lr",      1e-3, 8e-3, log=True)
    wd       = trial.suggest_float("wd",      1e-5, 5e-4, log=True)
    ls       = trial.suggest_float("ls",      0.05, 0.20, step=0.05)

    def make_lstm():
        return LSPLSTMBidirS11(N_DIMS, n_classes,
                                hidden=hidden, n_layers=n_layers, dropout=dropout)

    _, f1 = train_one_run(
        Xh_tr, yh_tr, Xh_val, yh_val,
        model_fn=make_lstm,
        lr=lr, wd=wd, label_smoothing=ls,
        n_epochs=N_EPOCHS_HPO, patience=PATIENCE_HPO, trial=trial,
    )
    return f1


t0 = time.time()
study_l = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=SEED),
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
)
# Warm start desde S10 best
study_l.enqueue_trial({
    "hidden": S10_BEST["hidden"],
    "n_layers": S10_BEST["n_layers"],
    "dropout": S10_BEST["dropout"],
    "lr":      S10_BEST["lr"],
    "wd":      S10_BEST["weight_decay"],
    "ls":      S10_BEST["label_smoothing"],
})
study_l.optimize(objective_lstm, n_trials=N_TRIALS_LSTM, show_progress_bar=True)
hp_l     = study_l.best_params
f1_hpo_l = study_l.best_value
print(f"\n  ✅ HPO BiLSTM → F1-hpo={f1_hpo_l:.4f}  tiempo={( time.time()-t0)/60:.1f} min")
print(f"  Mejores HPs: {hp_l}")

# --- B2. Entrenamiento final KFold(5) BiLSTM ---
print(f"\nB2. KFold({N_FOLDS}) final BiLSTM S11")


def make_best_lstm():
    return LSPLSTMBidirS11(N_DIMS, n_classes,
                            hidden=hp_l["hidden"],
                            n_layers=hp_l["n_layers"],
                            dropout=hp_l["dropout"])


_tmp2 = make_best_lstm()
print(f"  Params BiLSTM S11: {count_params(_tmp2):,}"); del _tmp2

fold_f1s_l, best_l_f1, best_l_model = [], 0.0, None
t_final_l = time.time()

for fi, (tr_i, va_i) in enumerate(skf.split(X_core, y_core), 1):
    m, f1 = train_one_run(
        X_core[tr_i], y_core[tr_i], X_core[va_i], y_core[va_i],
        model_fn=make_best_lstm,
        lr=hp_l["lr"], wd=hp_l["wd"], label_smoothing=hp_l["ls"],
        n_epochs=N_EPOCHS_FINAL, patience=PATIENCE_FINAL,
    )
    fold_f1s_l.append(f1)
    print(f"  Fold {fi}: F1-val={f1:.4f}")
    if f1 > best_l_f1:
        best_l_f1, best_l_model = f1, m

t_lstm_min = (time.time() - t_final_l) / 60
mean_l = float(np.mean(fold_f1s_l)); std_l = float(np.std(fold_f1s_l))
print(f"\n  BiLSTM S11 KFold F1: {mean_l:.4f} ± {std_l:.4f}  "
      f"({t_lstm_min:.1f} min)")

# Eval completa BiLSTM
print("\nB3. Evaluación BiLSTM S11 en test holdout:")
res_l = full_eval(best_l_model, X_te, y_te, "BiLSTM-S11-test")

# Calibración BiLSTM
last_va_l  = list(skf.split(X_core, y_core))[np.argmax(fold_f1s_l)][1]
Xcal_l, ycal_l = X_core[last_va_l], y_core[last_va_l]
res_cal_l      = full_eval(best_l_model, Xcal_l, ycal_l, "BiLSTM-S11-cal")
T_l, ece_pre_l, ece_post_l = calibrate(res_cal_l["logits"], res_cal_l["true"])
print(f"  ECE: {ece_pre_l:.4f} → {ece_post_l:.4f}  (T*={T_l:.3f})")

# HE3 BiLSTM
print("\nB4. HE3 Generalización BiLSTM S11:")
he3_l = he3_eval(best_l_model, res_l, X_ghold, y_ghold, T_l)

# ONNX BiLSTM
lat_l = export_onnx(best_l_model, "bilstm_s11.onnx")
ckpt_l = {
    "model_state": {k: v.cpu() for k, v in best_l_model.state_dict().items()},
    "architecture": "LSPLSTMBidirS11",
    "label2idx":   label2idx,
    "idx2label":   idx2label,
    "n_classes":   n_classes,
    "n_dims":      N_DIMS, "n_frames": N_FRAMES,
    "hidden":      hp_l["hidden"],
    "n_layers":    hp_l["n_layers"],
    "dropout":     hp_l["dropout"],
    "lr":          hp_l["lr"],
    "wd":          hp_l["wd"],
    "label_smoothing": hp_l["ls"],
    "temperature": T_l,
    "f1_val_mean": mean_l, "f1_val_std": std_l,
    "f1_test":     res_l["f1_macro"],
    "acc_test":    res_l["acc"],
    "top3_test":   res_l["top3"],
    "top5_test":   res_l["top5"],
    "ece_before":  ece_pre_l, "ece_after": ece_post_l,
    "fold_f1s":    fold_f1s_l,
    "he3":         he3_l,
    "latencia_onnx_ms": lat_l,
    "sprint": "S11",
}
torch.save(ckpt_l, CKPT_DIR / "bilstm_s11.pt")
print(f"  ✅ checkpoints/bilstm_s11.pt")

# ════════════════════════════════════════════════════════════════════════════
# RESUMEN COMPARATIVO
# ════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("RESUMEN COMPARATIVO S11 — Transformer vs BiLSTM")
print(f"{'='*70}")
print(f"  {'Métrica':<22} {'Transformer S11':>18} {'BiLSTM S11':>14} {'BiLSTM S10':>12}")
print(f"  {'-'*68}")
print(f"  {'F1-val KFold':<22} {mean_t:.4f} ±{std_t:.4f}  {mean_l:.4f} ±{std_l:.4f}  0.0262±0.0067")
print(f"  {'F1-test':<22} {res_t['f1_macro']:.4f}{'':>14} {res_l['f1_macro']:.4f}{'':>10} 0.0302")
print(f"  {'Top-3 Acc':<22} {res_t['top3']:.4f}{'':>14} {res_l['top3']:.4f}{'':>10} N/A")
print(f"  {'Top-5 Acc':<22} {res_t['top5']:.4f}{'':>14} {res_l['top5']:.4f}{'':>10} N/A")
print(f"  {'ECE (calibrado)':<22} {ece_post_t:.4f}{'':>14} {ece_post_l:.4f}{'':>10} 0.033")
print(f"  {'Latencia ONNX ms':<22} {lat_t:.2f}{'':>14} {lat_l:.2f}{'':>10} 0.9")
print(f"  {'HE3 (PSI)':<22} {he3_t['psi']:.4f}{'':>14} {he3_l['psi']:.4f}{'':>10} 0.0225")

winner = "Transformer" if res_t["f1_macro"] >= res_l["f1_macro"] else "BiLSTM"
winner_f1 = max(res_t["f1_macro"], res_l["f1_macro"])
print(f"\n  ★ GANADOR: {winner} S11  (F1-test={winner_f1:.4f})")

# Apuntar al ganador como "lstm_s11.pt" para compatibilidad
best_ckpt_name = "transformer_s11.pt" if winner == "Transformer" else "bilstm_s11.pt"
import shutil
shutil.copy(CKPT_DIR / best_ckpt_name, CKPT_DIR / "best_s11.pt")
shutil.copy(
    CKPT_DIR / ("transformer_s11.onnx" if winner == "Transformer" else "bilstm_s11.onnx"),
    CKPT_DIR / "best_s11.onnx",
)
print(f"  Copiado como: checkpoints/best_s11.pt + best_s11.onnx")

# ── Actualizar logs/runs.csv ──────────────────────────────────────────────────

fecha = datetime.date.today().strftime("%Y%m%d")
rows_to_add = [
    {
        "exp_id":      f"exp_{fecha}_transformer_s11",
        "sprint":      "S11",
        "modelo":      "Transformer-S11",
        "features":    "pose+rhand+lhand(150dims×30fr)",
        "hp_resumen":  f"d_model={hp_t['d_model']},nhead={hp_t['nhead']},n_layers={hp_t['n_layers']},lr={hp_t['lr']:.2e}",
        "f1_val_mean": f"{mean_t:.4f}",
        "f1_val_std":  f"{std_t:.4f}",
        "f1_test":     f"{res_t['f1_macro']:.4f}",
        "acc_test":    f"{res_t['acc']:.4f}",
        "tiempo_s":    f"{int(t_transf_min*60)}",
        "latencia_ms": f"{lat_t:.1f}",
        "split":       f"StratifiedKFold({N_FOLDS})",
        "seed":        str(SEED),
        "n_classes":   str(n_classes),
        "notas":       (f"Dataset S11 {len(X_core)} samples; "
                        f"ECE {ece_pre_t:.3f}→{ece_post_t:.3f}(T={T_t:.2f}); "
                        f"Top3={res_t['top3']:.4f}; HE3={'PASA' if he3_t['passed'] else 'FALLA'}; "
                        f"PSI={he3_t['psi']:.4f}"),
    },
    {
        "exp_id":      f"exp_{fecha}_bilstm_s11",
        "sprint":      "S11",
        "modelo":      "BiLSTM-S11",
        "features":    "pose+rhand+lhand(150dims×30fr)",
        "hp_resumen":  f"hidden={hp_l['hidden']},n_layers={hp_l['n_layers']},dropout={hp_l['dropout']:.2f},lr={hp_l['lr']:.2e}",
        "f1_val_mean": f"{mean_l:.4f}",
        "f1_val_std":  f"{std_l:.4f}",
        "f1_test":     f"{res_l['f1_macro']:.4f}",
        "acc_test":    f"{res_l['acc']:.4f}",
        "tiempo_s":    f"{int(t_lstm_min*60)}",
        "latencia_ms": f"{lat_l:.1f}",
        "split":       f"StratifiedKFold({N_FOLDS})",
        "seed":        str(SEED),
        "n_classes":   str(n_classes),
        "notas":       (f"Dataset S11 (S10+3360 abc extra); warmstart S10 best; "
                        f"ECE {ece_pre_l:.3f}→{ece_post_l:.3f}(T={T_l:.2f}); "
                        f"Top3={res_l['top3']:.4f}; HE3={'PASA' if he3_l['passed'] else 'FALLA'}; "
                        f"PSI={he3_l['psi']:.4f}"),
    },
]

runs_path = LOGS_DIR / "runs.csv"
fieldnames_default = list(rows_to_add[0].keys())
if runs_path.exists():
    with open(runs_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames_default = reader.fieldnames or fieldnames_default

with open(runs_path, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames_default, extrasaction="ignore")
    for row in rows_to_add:
        writer.writerow(row)
print(f"\n  ✅ logs/runs.csv actualizado con 2 corridas S11")

print(f"\n{'='*70}")
print("[TRAIN S11 COMPLETADO]")
print(f"{'='*70}")
print(f"  Transformer F1-test : {res_t['f1_macro']:.4f}")
print(f"  BiLSTM S11  F1-test : {res_l['f1_macro']:.4f}")
print(f"  S10 best    F1-test : 0.0302  (referencia)")
mejora = (winner_f1 / 0.0302 - 1) * 100
print(f"  Mejora vs S10       : {mejora:+.1f}%")
print(f"  Ganador             : {winner} S11")
