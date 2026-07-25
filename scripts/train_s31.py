"""train_s31.py — Sprint 31: BiLSTM sobre dataset_s31 (glosas reales en
contexto narrativo continuo, cruzando data/SRT con data/landmarks — ver
scripts/build_dataset_s31_continuo.py).

Diferencia de fondo respecto a S13-S30: todos los sprints anteriores
entrenaron y midieron sobre CLIPS YA AISLADOS (una seña, ya cortada). Este
es el primer sprint que entrena sobre ventanas extraídas de video narrativo
CONTINUO real, con la etiqueta correcta de cada ventana (glosa verificada
por el timestamp del SRT), no la clase del video completo. Es la tarea que
realmente falla en producción (traducir narración/palabras seguidas) —
antes nunca se había medido ni entrenado directamente sobre ella.

Split: GroupShuffleSplit a nivel de VIDEO (no de ventana) — ventanas
contiguas del mismo video comparten hasta 15 de 30 frames. 5 videos
completos quedan reservados aparte (s31_videos_wer_holdout.json) para la
evaluación WER real (scripts/evaluar_wer_s31.py), nunca vistos aquí.

Dataset pequeño (1538 muestras, 51 clases) — se usa --skip-hpo con los
hiperparámetros S13-best directamente (ya validados en 20+ sprints) en vez
de gastar tiempo en HPO sobre tan pocos datos.

Checkpoints: bilstm_s31.pt / bilstm_s31.onnx
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
from sklearn.model_selection import StratifiedKFold, GroupShuffleSplit
from sklearn.metrics import f1_score
from scipy.optimize import minimize_scalar
from collections import Counter

warnings.filterwarnings("ignore")

ROOT     = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
LOGS_DIR = ROOT / "logs"
CKPT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

SEED     = 42
N_FRAMES = 30
N_DIMS   = 150
BATCH    = 32  # dataset pequeño — batch menor que S29 (64) para más pasos/epoch

S13_BEST = {
    "hidden": 256, "n_layers": 1, "dropout": 0.20,
    "lr": 1.709e-3, "wd": 1.296e-4, "ls": 0.15,
}

N_EPOCHS_FINAL = 100
PATIENCE_FINAL = 15
N_FOLDS        = 5

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device  : {DEVICE}")
print("Sprint 31 — BiLSTM S31 | dataset_s31 (glosas reales, contexto narrativo continuo)")
print("=" * 65)


def normalize_sample(x: np.ndarray) -> np.ndarray:
    mu, std = x.mean(), x.std()
    if std < 1e-8:
        return x
    return ((x - mu) / std).astype(np.float32)


def normalize_batch(X: np.ndarray) -> np.ndarray:
    return np.stack([normalize_sample(X[i]) for i in range(len(X))])


def load_dataset():
    npz_path = DATA_DIR / "dataset_s31.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            "dataset_s31.npz no encontrado.\n"
            "  python3 scripts/build_dataset_s31_continuo.py"
        )
    data   = np.load(npz_path)
    X_raw  = data["X"]
    y_raw  = data["y"]
    groups = data["groups"]

    with open(DATA_DIR / "s31_label2idx.json", encoding="utf-8") as f:
        label2idx = json.load(f)
    idx2label = {int(v): k for k, v in label2idx.items()}

    print("  Aplicando normalización por muestra (z-score)...")
    X = normalize_batch(X_raw)
    n_cl = len(label2idx)

    print(f"\nDataset S31 (glosas reales en contexto narrativo continuo):")
    print(f"  Muestras  : {len(X)}  |  Clases activas: {n_cl}  |  Videos: {len(set(groups.tolist()))}")
    cnt = Counter(y_raw.tolist())
    v   = sorted(cnt.values())
    print(f"  Samples/cls: min={v[0]} max={v[-1]} mean={np.mean(v):.1f} median={np.median(v):.0f}")
    return X, y_raw, groups, n_cl, label2idx, idx2label


X_all, y_all, groups_all, n_classes, label2idx, idx2label = load_dataset()

gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
core_idx, test_idx = next(gss.split(X_all, y_all, groups=groups_all))
X_core, y_core, g_core = X_all[core_idx], y_all[core_idx], groups_all[core_idx]
X_te,   y_te           = X_all[test_idx],  y_all[test_idx]
print(f"  Train pool (videos): {len(set(g_core.tolist()))}  |  muestras: {len(X_core)}")
print(f"  Test holdout (videos nunca vistos): {len(set(groups_all[test_idx].tolist()))}  |  muestras: {len(X_te)}")

clases_core = set(y_core.tolist())
clases_te   = set(y_te.tolist())
sin_training = clases_te - clases_core
if sin_training:
    print(f"  ⚠️  Clases en test sin ninguna muestra de train: "
          f"{[idx2label.get(c, str(c)) for c in sorted(sin_training)]}")


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
                  lr, wd, label_smoothing, n_epochs, patience):
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
# HPs directas (--skip-hpo implícito: dataset chico, HPs ya validadas 20+ sprints)
# ════════════════════════════════════════════════════════════════════════════
hp = {k: S13_BEST[k] for k in ("hidden", "n_layers", "dropout", "lr", "wd", "ls")}
print(f"\nHPs (S13-best, sin HPO): {hp}")


# ════════════════════════════════════════════════════════════════════════════
# KFold(5) final
# ════════════════════════════════════════════════════════════════════════════
print(f"\nKFold({N_FOLDS}) final BiLSTM S31")


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

print("\nEvaluación en test holdout (videos completos nunca vistos en train):")
res_te = full_eval(best_model, X_te, y_te, "BiLSTM-S31-test")

best_fold_idx = int(np.argmax(fold_f1s))
last_va = list(skf.split(X_core, y_core))[best_fold_idx][1]
res_cal = full_eval(best_model, X_core[last_va], y_core[last_va], "BiLSTM-S31-cal")
T_opt, ece_pre, ece_post = calibrate(res_cal["logits"], res_cal["true"])
print(f"  ECE: {ece_pre:.4f} → {ece_post:.4f}  (T*={T_opt:.3f})")

lat = export_onnx(best_model, "bilstm_s31.onnx")

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
    "latencia_onnx_ms": lat,
    "sprint":           "S31",
}
torch.save(ckpt, CKPT_DIR / "bilstm_s31.pt")
print(f"  ✅ checkpoints/bilstm_s31.pt")

fecha = datetime.date.today().strftime("%Y%m%d")
row = {
    "exp_id":      f"exp_{fecha}_bilstm_s31",
    "sprint":      "S31",
    "modelo":      "BiLSTM-S31",
    "features":    "pose+rhand+lhand(150dims×30fr)+per-sample-zscore+srt-continuo",
    "hp_resumen":  (f"hidden={hp['hidden']},n_layers={hp['n_layers']},"
                    f"dropout={hp['dropout']:.2f},lr={hp['lr']:.2e}"),
    "f1_val_mean": f"{mean_f1:.4f}",
    "f1_val_std":  f"{std_f1:.4f}",
    "f1_test":     f"{res_te['f1_macro']:.4f}",
    "acc_test":    f"{res_te['acc']:.4f}",
    "tiempo_s":    f"{int(total_min*60)}",
    "latencia_ms": f"{lat:.1f}",
    "split":       f"GroupShuffleSplit(video)+StratifiedKFold({N_FOLDS})",
    "seed":        str(SEED),
    "n_classes":   str(n_classes),
    "notas":       (f"Dataset S31 (SRT x landmarks, contexto narrativo continuo) "
                    f"{len(X_all)} muestras {n_classes} clases per-sample-zscore aug-agresivo; "
                    f"skip-hpo S13-best; ECE {ece_pre:.3f}→{ece_post:.3f}(T={T_opt:.2f}); "
                    f"Top3={res_te['top3']:.4f} Top5={res_te['top5']:.4f}; "
                    f"5 videos reservados para evaluación WER real (ver s31_videos_wer_holdout.json)"),
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
print("[TRAIN S31 COMPLETADO]")
print(f"{'='*65}")
print(f"  BiLSTM S31  F1-test (videos nunca vistos): {res_te['f1_macro']:.4f}")
print(f"  Top-3 test          : {res_te['top3']:.4f}")
print(f"  Top-5 test          : {res_te['top5']:.4f}")
print(f"  Siguiente paso: scripts/evaluar_wer_s31.py — WER real contra SRT")
print(f"  sobre los 5 videos 100% reservados en s31_videos_wer_holdout.json")
