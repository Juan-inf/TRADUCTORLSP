"""train_s39_stgcn.py — Sprint 39: reemplaza BiLSTM sobre vector aplanado
(150-dim por frame) por una red convolucional de grafos espacio-temporal
(ST-GCN, Yan et al. 2018 AAAI) sobre los mismos 75 landmarks, esta vez
preservando su estructura anatómica (qué articulación está conectada con
cuál) en vez de aplanarlos a un vector genérico.

Hipótesis (revisión técnica de la conversación): el techo de F1=0.4426 con
BiLSTM no es solo falta de datos — es el método de referencia en SLR
basado en esqueleto (ST-GCN y sus variantes 2s-AGCN) el que inyecta
conocimiento estructural (conectividad de dedos/mano/muñeca/brazo) que
compensa parcialmente la escasez de muestras por clase, en vez de que el
modelo tenga que aprenderla desde cero con un MLP+BiLSTM sobre un vector
sin estructura espacial explícita.

Mismo dataset (dataset_s17.npz, min15, 96 clases — el mismo que entrenó
v4/S27) y misma metodología de partición (StratifiedShuffleSplit test=15%,
GroupShuffleSplit para HE3) para una comparación directa y no circular
contra F1=0.4426.

Los features se DES-aplanan de vuelta a [T,75,2] (mismo orden que
kp_seq_to_features: pose_x,pose_y,left_x,left_y,right_x,right_y) sin
volver a correr MediaPipe — la información espacial ya está en el dataset
existente, solo estaba aplanada.

Checkpoints: stgcn_s39.pt / stgcn_s39.onnx
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
CKPT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

SEED = 42
N_FRAMES = 30
N_JOINTS = 75  # left(0:21) + right(21:42) + pose(42:75)
BATCH = 64
MIN_SAMPLES = 15
N_EPOCHS_FINAL = 30
PATIENCE_FINAL = 6

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device  : {DEVICE}")
print("Sprint 39 — ST-GCN | dataset_s17 (mismo dataset y split que v4/S27), 96 clases")
print("=" * 65)


# ─── Grafo anatómico (mismos 75 landmarks, conectividad real) ────────────────

def construir_adyacencia():
    import mediapipe as mp
    hand_edges = sorted(mp.solutions.hands.HAND_CONNECTIONS)
    pose_edges = sorted(mp.solutions.pose.POSE_CONNECTIONS)

    edges = []
    # mano izquierda: índices 0..20 (offset 0)
    edges += [(a, b) for a, b in hand_edges]
    # mano derecha: índices 21..41 (offset 21)
    edges += [(a + 21, b + 21) for a, b in hand_edges]
    # pose: índices 42..74 (offset 42)
    edges += [(a + 42, b + 42) for a, b in pose_edges]
    # puentes anatómicos: muñeca de cada mano <-> muñeca correspondiente del
    # esqueleto de pose (mediapipe pose: 15=left_wrist, 16=right_wrist)
    edges.append((0, 42 + 15))    # mano izquierda (landmark 0 = wrist) -- pose left_wrist
    edges.append((21, 42 + 16))  # mano derecha (landmark 0 = wrist) -- pose right_wrist

    A = np.eye(N_JOINTS, dtype=np.float32)  # auto-conexiones
    for a, b in edges:
        A[a, b] = 1.0
        A[b, a] = 1.0

    # normalización simétrica D^-1/2 (A+I) D^-1/2 (Kipf & Welling / Yan et al.)
    deg = A.sum(axis=1)
    d_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    d_inv_sqrt[deg == 0] = 0
    D_inv_sqrt = np.diag(d_inv_sqrt)
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.from_numpy(A_norm.astype(np.float32))


# ─── Des-aplanar features [N,30,150] -> [N,30,75,2] (mismo orden que kp_seq_to_features) ──

def desaplanar(X150):
    pose_x, pose_y = X150[..., 0:33], X150[..., 33:66]
    left_x, left_y = X150[..., 66:87], X150[..., 87:108]
    right_x, right_y = X150[..., 108:129], X150[..., 129:150]
    left = np.stack([left_x, left_y], axis=-1)     # [N,T,21,2]
    right = np.stack([right_x, right_y], axis=-1)  # [N,T,21,2]
    pose = np.stack([pose_x, pose_y], axis=-1)      # [N,T,33,2]
    return np.concatenate([left, right, pose], axis=2).astype(np.float32)  # [N,T,75,2]


def normalize_sample(x):
    """z-score escalar sobre toda la secuencia (igual que el resto del proyecto)."""
    mu, std = x.mean(), x.std()
    if std < 1e-8:
        return x
    return ((x - mu) / std).astype(np.float32)


def normalize_batch(X):
    return np.stack([normalize_sample(X[i]) for i in range(len(X))])


def load_dataset():
    npz_path = DATA_DIR / "dataset_s17.npz"
    data = np.load(npz_path)
    X_raw, y_raw, groups = data["X"], data["y"], data["groups"]

    with open(DATA_DIR / "s17_label2idx.json", encoding="utf-8") as f:
        label2idx = json.load(f)
    idx2label = {int(v): k for k, v in label2idx.items()}

    counts = Counter(y_raw.tolist())
    keep = np.array([counts[int(v)] >= MIN_SAMPLES for v in y_raw])
    X_raw, y_raw, groups = X_raw[keep], y_raw[keep], groups[keep]
    n_cl = len(set(y_raw.tolist()))

    print("  Des-aplanando a [N,30,75,2] (estructura de grafo)...")
    X_graph = desaplanar(X_raw)  # [N,30,75,2]
    print("  Aplicando normalización por muestra (z-score)...")
    X = normalize_batch(X_graph)

    print(f"\nDataset S17 (mismo que v4/S27), estructura de grafo:")
    print(f"  Muestras  : {len(X)}  |  Clases activas: {n_cl}")
    return X, y_raw, groups, n_cl, label2idx, idx2label


X_all, y_all, groups_all, n_classes, label2idx, idx2label = load_dataset()

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
tv_idx, te_idx = next(sss.split(X_all, y_all))
X_tv, y_tv, g_tv = X_all[tv_idx], y_all[tv_idx], groups_all[tv_idx]
X_te, y_te = X_all[te_idx], y_all[te_idx]
print(f"  Train+Val : {len(X_tv)}  |  Test holdout: {len(X_te)}  (mismo split que reportó F1=0.4426 de v4)")

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
    """x: [T,V,C]"""
    T = x.shape[0]
    orig = np.linspace(0, T - 1, T)
    warp = np.clip(np.sort(orig + np.random.randn(T) * sigma * T / knots), 0, T - 1)
    out = np.zeros_like(x)
    flat = x.reshape(T, -1)
    out_flat = out.reshape(T, -1)
    for d in range(flat.shape[1]):
        out_flat[:, d] = np.interp(orig, warp, flat[:, d])
    return out.astype(np.float32)


class SignGraphDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X.copy()
        self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].copy()  # [T,75,2]
        if self.augment:
            x = x + np.random.randn(*x.shape).astype(np.float32) * 0.020
            if np.random.rand() < 0.60:
                x = x * np.random.uniform(0.85, 1.15)
            if np.random.rand() < 0.60:
                x = time_warp(x, sigma=0.06)
            if np.random.rand() < 0.40:
                n_drop = int(N_JOINTS * np.random.uniform(0.05, 0.15))
                idx_drop = np.random.choice(N_JOINTS, n_drop, replace=False)
                x[:, idx_drop, :] = 0.0
            x = normalize_sample(x)
        # a [C,T,V] para Conv2d (canal, tiempo, nodo)
        x_t = torch.from_numpy(x).float().permute(2, 0, 1)  # [2,T,75]
        return x_t, self.y[i]


def make_loaders(X_tr, y_tr, X_val, y_val, batch=BATCH):
    cc = Counter(y_tr.tolist())
    w = [1.0 / cc[int(c)] for c in y_tr]
    smp = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    dl_tr = DataLoader(SignGraphDataset(X_tr, y_tr, augment=True),
                        batch_size=batch, sampler=smp, num_workers=0)
    dl_val = DataLoader(SignGraphDataset(X_val, y_val, augment=False),
                         batch_size=batch, shuffle=False, num_workers=0)
    return dl_tr, dl_val


def make_criterion(y_tr, label_smoothing=0.15):
    cc = Counter(y_tr.tolist())
    cw = torch.tensor(
        [len(y_tr) / (n_classes * cc.get(i, 1)) for i in range(n_classes)],
        dtype=torch.float32,
    ).to(DEVICE)
    return nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)


# ─── ST-GCN ───────────────────────────────────────────────────────────────────

class STGCNBlock(nn.Module):
    """Bloque ST-GCN: conv de grafo espacial (X' = A_norm @ X @ W) seguida de
    conv temporal (kernel 9 sobre el eje T), con conexión residual."""

    def __init__(self, in_ch, out_ch, A_norm, kernel_t=9, stride_t=1, dropout=0.2):
        super().__init__()
        self.register_buffer("A", A_norm)
        self.gcn = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.bn_gcn = nn.BatchNorm2d(out_ch)
        pad = (kernel_t - 1) // 2
        self.tcn = nn.Conv2d(out_ch, out_ch, kernel_size=(kernel_t, 1),
                             stride=(stride_t, 1), padding=(pad, 0))
        self.bn_tcn = nn.BatchNorm2d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        if in_ch == out_ch and stride_t == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(stride_t, 1)),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        # x: [B, C, T, V]
        res = self.residual(x)
        # conv de grafo: agregar sobre vecinos usando A_norm, luego proyección de canales
        x = torch.einsum("bctv,vw->bctw", x, self.A)
        x = self.relu(self.bn_gcn(self.gcn(x)))
        x = self.bn_tcn(self.tcn(x))
        x = self.dropout(x)
        return self.relu(x + res)


class STGCN(nn.Module):
    def __init__(self, n_classes, A_norm, in_ch=2, dropout=0.2):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(in_ch * N_JOINTS)
        self.block1 = STGCNBlock(in_ch, 64, A_norm, dropout=dropout)
        self.block2 = STGCNBlock(64, 128, A_norm, dropout=dropout)
        self.block3 = STGCNBlock(128, 256, A_norm, dropout=dropout)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(256, 256), nn.GELU(),
            nn.Dropout(dropout * 0.5), nn.Linear(256, n_classes),
        )

    def forward(self, x):
        # x: [B, C, T, V]
        B, C, T, V = x.shape
        x = x.permute(0, 1, 3, 2).reshape(B, C * V, T)
        x = self.data_bn(x)
        x = x.reshape(B, C, V, T).permute(0, 1, 3, 2)  # back to [B,C,T,V]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.mean(dim=[2, 3])  # global average pool sobre tiempo y nodos
        return self.head(x)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


A_norm = construir_adyacencia().to(DEVICE)


def make_model():
    return STGCN(n_classes, A_norm, in_ch=2, dropout=0.2)


print(f"  Params: {count_params(make_model()):,}")


def train_one_run(X_tr, y_tr, X_val, y_val, lr, wd, label_smoothing, n_epochs, patience):
    dl_tr, dl_val = make_loaders(X_tr, y_tr, X_val, y_val)
    model = make_model().to(DEVICE)
    criterion = make_criterion(y_tr, label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=25, T_mult=2, eta_min=lr * 0.01)

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
    dl = DataLoader(SignGraphDataset(X_eval, y_eval, augment=False),
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
    dummy = torch.randn(1, 2, N_FRAMES, N_JOINTS)
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
        np_in = np.zeros((1, 2, N_FRAMES, N_JOINTS), dtype=np.float32)
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


LR, WD, LS = 1.5e-3, 1.0e-4, 0.15

print(f"\nEntrenamiento (split único, {N_EPOCHS_FINAL} épocas máx, paciencia {PATIENCE_FINAL}):")
t_final = time.time()
best_model, f1_val = train_one_run(
    X_tr, y_tr, X_va, y_va, lr=LR, wd=WD, label_smoothing=LS,
    n_epochs=N_EPOCHS_FINAL, patience=PATIENCE_FINAL,
)
total_min = (time.time() - t_final) / 60
print(f"\n  F1-val: {f1_val:.4f}  ({total_min:.1f} min)")

print("\nEvaluación en test holdout (MISMO split que v4/S27, F1=0.4426):")
res_te = full_eval(best_model, X_te, y_te, "STGCN-S39-test")
T_opt, ece_pre, ece_post = calibrate(res_te["logits"], res_te["true"])
print(f"  ECE: {ece_pre:.4f} → {ece_post:.4f}  (T*={T_opt:.3f})")

print("\nHE3 Generalización:")
he3 = he3_eval(res_te, X_ghold, y_ghold, best_model)

lat = export_onnx(best_model, "stgcn_s39.onnx")

ckpt = {
    "model_state": {k: v.cpu() for k, v in best_model.state_dict().items()},
    "architecture": "STGCN",
    "label2idx": label2idx,
    "idx2label": idx2label,
    "n_classes": n_classes,
    "f1_val": f1_val,
    "f1_test": res_te["f1_macro"],
    "acc_test": res_te["acc"],
    "top3_test": res_te["top3"],
    "top5_test": res_te["top5"],
    "ece_before": ece_pre,
    "ece_after": ece_post,
    "he3": he3,
    "latencia_onnx_ms": lat,
    "sprint": "S39",
}
torch.save(ckpt, CKPT_DIR / "stgcn_s39.pt")
print(f"  ✅ checkpoints/stgcn_s39.pt")

fecha = datetime.date.today().strftime("%Y%m%d")
row = {
    "exp_id": f"exp_{fecha}_stgcn_s39",
    "sprint": "S39",
    "modelo": "STGCN-S39",
    "features": "pose+rhand+lhand(75 nodos x2 coords x30fr)+grafo-anatomico+per-sample-zscore",
    "hp_resumen": f"canales=64-128-256,kernel_t=9,dropout=0.2,lr={LR:.2e}",
    "f1_val_mean": f"{f1_val:.4f}",
    "f1_val_std": "0.0000",
    "f1_test": f"{res_te['f1_macro']:.4f}",
    "acc_test": f"{res_te['acc']:.4f}",
    "tiempo_s": f"{int(total_min*60)}",
    "latencia_ms": f"{lat:.1f}",
    "split": "StratifiedShuffleSplit(single)+GroupHold",
    "seed": str(SEED),
    "n_classes": str(n_classes),
    "notas": (f"ST-GCN (Yan et al. 2018) sobre dataset_s17 (mismo dataset y split que v4/S27, "
              f"96 clases, F1=0.4426 de referencia). Grafo anatómico real (HAND_CONNECTIONS + "
              f"POSE_CONNECTIONS + puentes muñeca-mano/muñeca-pose), sin aplanar a vector; "
              f"ECE {ece_pre:.3f}→{ece_post:.3f}(T={T_opt:.2f}); Top3={res_te['top3']:.4f} "
              f"Top5={res_te['top5']:.4f}; HE3={'PASA' if he3['passed'] else 'FALLA'} "
              f"ΔF1={he3['delta_f1']:.4f}; objetivo: comparar F1 vs. 0.4426 (BiLSTM) y vs. 0.70 (OE1)"),
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
print("[TRAIN S39 (ST-GCN) COMPLETADO]")
print(f"{'='*65}")
print(f"  F1-test    : {res_te['f1_macro']:.4f}  (96 clases — comparable 1:1 con v4=0.4426)")
print(f"  Top-5 test : {res_te['top5']:.4f}")
print(f"  HE3        : {'PASA' if he3['passed'] else 'FALLA'}  ΔF1={he3['delta_f1']:.4f}")
print(f"  vs. v4 (BiLSTM, F1=0.4426): {'MEJORA' if res_te['f1_macro'] > 0.4426 else 'NO mejora'}")
print(f"  vs. meta OE1 (F1>=0.70): {'CUMPLE' if res_te['f1_macro'] >= 0.70 else 'no alcanza'}")
