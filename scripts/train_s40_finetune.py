"""train_s40_finetune.py — Sprint 40: transferencia de aprendizaje.

Método combinado (a pedido explícito: "usa todos los métodos posibles"):
en vez de entrenar el BiLSTM de 96 clases desde cero (como v4/S27) o
reescribir la arquitectura desde cero (ST-GCN, S39 — fracasó sin tuning),
se transfiere el backbone (proj+lstm+attn+head.1) ya entrenado del
checkpoint bilstm_s36.pt — entrenado con presupuesto COMPLETO (KFold(5),
568.9 min) sobre el corpus más grande disponible en el proyecto
(dataset_s35, 274 clases, 21 221 muestras: S18 + s31_continuo + s34_eaf) —
y se hace fine-tuning solo de la capa final (head.4, específica de clase)
más ajuste fino del resto sobre el vocabulario objetivo real (dataset_s17,
96 clases, el mismo dataset y split que entrenó v4/S27, F1=0.4426).

Hipótesis: el backbone entrenado sobre 21 221 muestras (5.2x más datos que
las 12 150 de dataset_s17) aprendió representaciones más generales de
movimiento de LSP (mano/pose) que un backbone entrenado solo con las
12 150 muestras del vocabulario de 96 clases — exactamente el principio de
transferencia de aprendizaje ya validado en NLP/CV de bajo recurso, pero
usando datos que YA existen en el proyecto (sin depender de un dataset
externo no-LSP).

Mismo dataset_s17.npz + mismo split (StratifiedShuffleSplit test=15%,
GroupShuffleSplit HE3, seed=42) que v4/S27 para comparación directa y no
circular contra F1=0.4426.

Checkpoints: bilstm_s40_finetune.pt / bilstm_s40_finetune.onnx
"""
import json, time, warnings, pathlib, csv, datetime, gc, os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


def _free_mps():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.metrics import f1_score
from scipy.optimize import minimize_scalar
from scipy.stats import ks_2samp
from collections import Counter

ROOT = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
LOGS_DIR = ROOT / "logs"

SEED = 42
N_FRAMES = 30
N_DIMS = 150
BATCH = 64
MIN_SAMPLES = 15

S13_BEST = {"hidden": 256, "n_layers": 1, "dropout": 0.20,
            "lr": 1.709e-3, "wd": 1.296e-4, "ls": 0.15}
LR_BACKBONE_MULT = 0.3   # backbone preentrenado: LR más bajo que la cabeza nueva
N_EPOCHS_FINAL = 40
PATIENCE_FINAL = 8

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device  : {DEVICE}")
print("Sprint 40 — Fine-tuning desde backbone S36 (274 clases) -> dataset_s17 (96 clases, split de v4)")
print("=" * 65)


def normalize_sample(x):
    mu, std = x.mean(), x.std()
    if std < 1e-8:
        return x
    return ((x - mu) / std).astype(np.float32)


def normalize_batch(X):
    return np.stack([normalize_sample(X[i]) for i in range(len(X))])


def load_dataset():
    data = np.load(DATA_DIR / "dataset_s17.npz")
    X_raw, y_raw, groups = data["X"], data["y"], data["groups"]
    with open(DATA_DIR / "s17_label2idx.json", encoding="utf-8") as f:
        label2idx = json.load(f)
    idx2label = {int(v): k for k, v in label2idx.items()}

    counts = Counter(y_raw.tolist())
    keep = np.array([counts[int(v)] >= MIN_SAMPLES for v in y_raw])
    X_raw, y_raw, groups = X_raw[keep], y_raw[keep], groups[keep]
    n_cl = len(set(y_raw.tolist()))

    X = normalize_batch(X_raw)
    print(f"\nDataset S17 (96 clases, mismo dataset y split que v4/S27):")
    print(f"  Muestras  : {len(X)}  |  Clases activas: {n_cl}")
    return X, y_raw, groups, n_cl, label2idx, idx2label


X_all, y_all, groups_all, n_classes, label2idx, idx2label = load_dataset()

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
tv_idx, te_idx = next(sss.split(X_all, y_all))
X_tv, y_tv, g_tv = X_all[tv_idx], y_all[tv_idx], groups_all[tv_idx]
X_te, y_te = X_all[te_idx], y_all[te_idx]
print(f"  Train+Val : {len(X_tv)}  |  Test holdout: {len(X_te)}  (idéntico al split que reportó F1=0.4426)")

gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
core_idx, group_hold_idx = next(gss.split(X_tv, y_tv, groups=g_tv))
X_core, y_core = X_tv[core_idx], y_tv[core_idx]
X_ghold, y_ghold = X_tv[group_hold_idx], y_tv[group_hold_idx]
print(f"  Core train: {len(X_core)}  |  Group holdout HE3: {len(X_ghold)}")

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
tr_idx, va_idx = next(sss2.split(X_core, y_core))
X_tr, y_tr = X_core[tr_idx], y_core[tr_idx]
X_va, y_va = X_core[va_idx], y_core[va_idx]
print(f"  Train: {len(X_tr)}  |  Val: {len(X_va)}")


def time_warp(x, sigma=0.04, knots=4):
    T = x.shape[0]
    orig = np.linspace(0, T - 1, T)
    warp = np.clip(np.sort(orig + np.random.randn(T) * sigma * T / knots), 0, T - 1)
    out = np.zeros_like(x)
    for d in range(x.shape[1]):
        out[:, d] = np.interp(orig, warp, x[:, d])
    return out.astype(np.float32)


class SignDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X.copy()
        self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].copy()
        if self.augment:
            x = x + np.random.randn(*x.shape).astype(np.float32) * 0.020
            if np.random.rand() < 0.60:
                x = x * np.random.uniform(0.85, 1.15)
            if np.random.rand() < 0.70:
                x[:, 66:87] = -x[:, 66:87]
                x[:, 108:129] = -x[:, 108:129]
            if np.random.rand() < 0.60:
                x = time_warp(x, sigma=0.06)
            if np.random.rand() < 0.50:
                n_drop = int(N_DIMS * np.random.uniform(0.05, 0.15))
                x[:, np.random.choice(N_DIMS, n_drop, replace=False)] = 0.0
            x = normalize_sample(x)
        return torch.from_numpy(x).float(), self.y[i]


def make_loaders(X_tr, y_tr, X_val, y_val, batch=BATCH):
    cc = Counter(y_tr.tolist())
    w = [1.0 / cc[int(c)] for c in y_tr]
    smp = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    dl_tr = DataLoader(SignDataset(X_tr, y_tr, augment=True),
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
        x = self.proj(x)
        h, _ = self.lstm(x)
        return self.head(self.attn(h))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


hp = {k: S13_BEST[k] for k in ("hidden", "n_layers", "dropout", "lr", "wd", "ls")}


def make_finetune_model():
    """Construye el modelo de 96 clases y transfiere proj+lstm+attn+head.1
    desde bilstm_s36.pt (274 clases, presupuesto completo). Solo head.4
    (clasificador final, dependiente del número de clases) queda con
    inicialización aleatoria."""
    model = LSPLSTMBidirS27(N_DIMS, n_classes, hidden=hp["hidden"],
                             n_layers=hp["n_layers"], dropout=hp["dropout"])
    ckpt_s36 = torch.load(CKPT_DIR / "bilstm_s36.pt", map_location="cpu", weights_only=False)
    sd_s36 = ckpt_s36["model_state"]
    own_sd = model.state_dict()
    transferidas, nuevas = [], []
    for k, v in sd_s36.items():
        if k in own_sd and own_sd[k].shape == v.shape:
            own_sd[k] = v.clone()
            transferidas.append(k)
        else:
            nuevas.append(k)
    model.load_state_dict(own_sd)
    print(f"  Transferidas desde bilstm_s36.pt: {len(transferidas)} tensores ({', '.join(transferidas[:4])}, ...)")
    print(f"  Inicializadas de nuevo (dependientes de n_classes): {nuevas}")
    return model


print(f"\nHPs (S13-best, sin HPO — igual que v4/S27): {hp}")
print(f"  Params: {count_params(make_finetune_model()):,}")


def train_finetune(X_tr, y_tr, X_val, y_val, n_epochs, patience):
    dl_tr, dl_val = make_loaders(X_tr, y_tr, X_val, y_val)
    model = make_finetune_model().to(DEVICE)
    criterion = make_criterion(y_tr, hp["ls"])

    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        (backbone_params if not name.startswith("head.4") else head_params).append(p)
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": hp["lr"] * LR_BACKBONE_MULT},
        {"params": head_params, "lr": hp["lr"]},
    ], weight_decay=hp["wd"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=25, T_mult=2, eta_min=hp["lr"] * 0.01)

    best_f1, best_state, pat_cnt = 0.0, None, 0
    t0 = time.time()
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
        print(f"    epoch {epoch:>3d}  F1-val={f1_val:.4f}  ({time.time()-t0:.0f}s)")

        if f1_val > best_f1:
            best_f1 = f1_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat_cnt = 0
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
    probs = torch.softmax(logits_t, dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    true_arr = np.array(all_true)

    acc = (preds == true_arr).mean()
    f1_m = f1_score(true_arr, preds, average="macro", zero_division=0)
    top3 = sum(int(t) in np.argsort(probs[i])[-3:] for i, t in enumerate(true_arr)) / len(true_arr)
    top5 = sum(int(t) in np.argsort(probs[i])[-5:] for i, t in enumerate(true_arr)) / len(true_arr)

    print(f"  [{label}] Acc={acc:.4f}  F1-macro={f1_m:.4f}  Top-3={top3:.4f}  Top-5={top5:.4f}")
    return {"f1_macro": f1_m, "acc": acc, "top3": top3, "top5": top5,
            "probs": probs, "preds": preds, "true": true_arr, "logits": logits_t.numpy()}


def ece_score(logits_np, labels_np, n_bins=15):
    logits = torch.from_numpy(logits_np)
    labels = torch.from_numpy(labels_np).long()
    probs = torch.softmax(logits, dim=1)
    confs, preds = probs.max(dim=1)
    accs = preds.eq(labels)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        mask = (confs > lo) & (confs <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.float().mean().item() * abs(
            accs[mask].float().mean().item() - confs[mask].float().mean().item())
    return ece


def calibrate(logits_np, labels_np):
    logits_t = torch.from_numpy(logits_np)
    labels_t = torch.from_numpy(labels_np).long()
    ece_pre = ece_score(logits_np, labels_np)

    def nll_t(T):
        return F.cross_entropy(logits_t / T, labels_t).item()

    res = minimize_scalar(nll_t, bounds=(0.1, 10.0), method="bounded")
    T_opt = float(res.x)
    return T_opt, ece_pre, ece_score(logits_np / T_opt, labels_np)


def he3_eval(res_test, X_ghold, y_ghold, model):
    res_gh = full_eval(model, X_ghold, y_ghold, label="group-holdout")
    delta_f1 = abs(res_test["f1_macro"] - res_gh["f1_macro"])

    conf_te = res_test["probs"].max(axis=1)
    conf_gh = res_gh["probs"].max(axis=1)
    bins = np.linspace(0, 1, 11)
    p_ref, _ = np.histogram(conf_te, bins=bins, density=True)
    p_new, _ = np.histogram(conf_gh, bins=bins, density=True)
    p_ref = np.clip(p_ref / (p_ref.sum() + 1e-8), 1e-8, None)
    p_new = np.clip(p_new / (p_new.sum() + 1e-8), 1e-8, None)
    psi = float(np.sum((p_ref - p_new) * np.log(p_ref / p_new)))
    ks_stat, ks_pval = ks_2samp(conf_te, conf_gh)

    ok_f1, ok_psi, ok_ks = delta_f1 <= 0.15, psi < 0.20, ks_pval > 0.05
    passed = ok_f1 and ok_psi and ok_ks
    print(f"  HE3 → ΔF1={delta_f1:.4f} {'✅' if ok_f1 else '❌'}  "
          f"PSI={psi:.4f} {'✅' if ok_psi else '❌'}  "
          f"KS p={ks_pval:.4f} {'✅' if ok_ks else '❌'}  "
          f"→ {'PASA ✅' if passed else 'FALLA ❌'}")
    return {"delta_f1": delta_f1, "psi": psi, "ks_pval": float(ks_pval), "passed": passed,
            "f1_ghold": res_gh["f1_macro"]}


def export_onnx(model, name):
    model.cpu().eval()
    dummy = torch.randn(1, N_FRAMES, N_DIMS)
    onnx_p = CKPT_DIR / name
    try:
        torch.onnx.export(
            model, dummy, str(onnx_p),
            input_names=["sequence"], output_names=["logits"],
            dynamic_axes={"sequence": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_p))
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


print(f"\nFine-tuning (split único, {N_EPOCHS_FINAL} épocas máx, paciencia {PATIENCE_FINAL}, "
      f"LR backbone x{LR_BACKBONE_MULT}):")
t_final = time.time()
best_model, f1_val = train_finetune(X_tr, y_tr, X_va, y_va,
                                     n_epochs=N_EPOCHS_FINAL, patience=PATIENCE_FINAL)
total_min = (time.time() - t_final) / 60
print(f"\n  F1-val: {f1_val:.4f}  ({total_min:.1f} min)")

print("\nEvaluación en test holdout (MISMO split que v4/S27, F1=0.4426):")
res_te = full_eval(best_model, X_te, y_te, "S40-finetune-test")
T_opt, ece_pre, ece_post = calibrate(res_te["logits"], res_te["true"])
print(f"  ECE: {ece_pre:.4f} → {ece_post:.4f}  (T*={T_opt:.3f})")

print("\nHE3 Generalización:")
he3 = he3_eval(res_te, X_ghold, y_ghold, best_model)

lat = export_onnx(best_model, "bilstm_s40_finetune.onnx")

ckpt = {
    "model_state": {k: v.cpu() for k, v in best_model.state_dict().items()},
    "architecture": "LSPLSTMBidirS27",
    "normalize_input": True,
    "label2idx": label2idx,
    "idx2label": idx2label,
    "n_classes": n_classes,
    "n_dims": N_DIMS,
    "n_frames": N_FRAMES,
    "hidden": hp["hidden"],
    "n_layers": hp["n_layers"],
    "dropout": hp["dropout"],
    "temperature": T_opt,
    "f1_val": f1_val,
    "f1_test": res_te["f1_macro"],
    "acc_test": res_te["acc"],
    "top3_test": res_te["top3"],
    "top5_test": res_te["top5"],
    "ece_before": ece_pre,
    "ece_after": ece_post,
    "he3": he3,
    "latencia_onnx_ms": lat,
    "sprint": "S40",
    "pretrained_from": "bilstm_s36.pt (274 clases, dataset_s35, KFold(5) completo)",
}
torch.save(ckpt, CKPT_DIR / "bilstm_s40_finetune.pt")
print(f"  ✅ checkpoints/bilstm_s40_finetune.pt")

fecha = datetime.date.today().strftime("%Y%m%d")
row = {
    "exp_id": f"exp_{fecha}_bilstm_s40_finetune",
    "sprint": "S40",
    "modelo": "BiLSTM-S40-finetune",
    "features": "pose+rhand+lhand(150dims×30fr)+per-sample-zscore+transfer-learning-desde-s36",
    "hp_resumen": f"hidden={hp['hidden']},n_layers={hp['n_layers']},dropout={hp['dropout']:.2f},lr={hp['lr']:.2e},lr_backbone_mult={LR_BACKBONE_MULT}",
    "f1_val_mean": f"{f1_val:.4f}",
    "f1_val_std": "0.0000",
    "f1_test": f"{res_te['f1_macro']:.4f}",
    "acc_test": f"{res_te['acc']:.4f}",
    "tiempo_s": f"{int(total_min*60)}",
    "latencia_ms": f"{lat:.1f}",
    "split": "StratifiedShuffleSplit(single)+GroupHold",
    "seed": str(SEED),
    "n_classes": str(n_classes),
    "notas": (f"Transferencia de aprendizaje: backbone (proj+lstm+attn+head.1) transferido de "
              f"bilstm_s36.pt (274 clases, dataset_s35, KFold(5) completo, 568.9 min), "
              f"fine-tuning sobre dataset_s17 (96 clases, mismo split que v4/S27); "
              f"ECE {ece_pre:.3f}→{ece_post:.3f}(T={T_opt:.2f}); Top3={res_te['top3']:.4f} "
              f"Top5={res_te['top5']:.4f}; HE3={'PASA' if he3['passed'] else 'FALLA'} "
              f"ΔF1={he3['delta_f1']:.4f}; comparar F1 vs 0.4426 (v4 desde cero) y vs 0.70 (OE1)"),
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
print("[TRAIN S40 (FINE-TUNE DESDE S36) COMPLETADO]")
print(f"{'='*65}")
print(f"  F1-test    : {res_te['f1_macro']:.4f}  (96 clases — comparable 1:1 con v4=0.4426)")
print(f"  Top-5 test : {res_te['top5']:.4f}")
print(f"  HE3        : {'PASA' if he3['passed'] else 'FALLA'}  ΔF1={he3['delta_f1']:.4f}")
print(f"  vs. v4 (desde cero, F1=0.4426): {'MEJORA' if res_te['f1_macro'] > 0.4426 else 'NO mejora'}")
print(f"  vs. meta OE1 (F1>=0.70): {'CUMPLE' if res_te['f1_macro'] >= 0.70 else 'no alcanza'}")
