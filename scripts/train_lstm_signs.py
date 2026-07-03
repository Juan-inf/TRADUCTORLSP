"""
train_lstm_signs.py — Entrena LSTM Bidireccional + Attention sobre señas LSP.
Input:  data/dataset_lstm.npz  [N, T=30, 150 dims]
Output: checkpoints/lstm_signs.pt  +  checkpoints/lstm_signs.onnx
"""

import json, time, warnings, pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from collections import Counter

warnings.filterwarnings("ignore")

ROOT     = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

# ── Hiperparámetros ───────────────────────────────────────────────────────────

SEED       = 42
N_FRAMES   = 30
N_DIMS     = 150
HIDDEN     = 256
N_LAYERS   = 2
DROPOUT    = 0.35
BATCH      = 64
LR         = 1e-3
WEIGHT_DECAY = 1e-4
N_EPOCHS   = 80
PATIENCE   = 12
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device: {DEVICE}")

# ── Cargar dataset ────────────────────────────────────────────────────────────

print("Cargando dataset_lstm.npz...")
data      = np.load(DATA_DIR / "dataset_lstm.npz")
X_all     = data["X"]   # [N, 30, 150]
y_all     = data["y"]   # [N]
n_classes = int(y_all.max()) + 1

with open(DATA_DIR / "lstm_label2idx.json", encoding="utf-8") as f:
    label2idx = json.load(f)
idx2label = {int(v): k for k, v in label2idx.items()}

print(f"  X: {X_all.shape}  |  LSP - Vocabulario-palabras: {n_classes}")

# ── Filtrar LSP - Vocabulario-palabras con < 2 muestras y re-mapear ───────────────────────────────

counts    = Counter(y_all.tolist())
keep_mask = np.array([counts[int(y)] >= 2 for y in y_all])
X_all     = X_all[keep_mask]
y_raw     = y_all[keep_mask]

# Re-mapear idx originales a índices contiguos 0..K-1
old_ids   = sorted(set(y_raw.tolist()))
remap     = {old: new for new, old in enumerate(old_ids)}
y_all     = np.array([remap[int(v)] for v in y_raw], dtype=np.int64)
n_classes = len(old_ids)
idx2label = {remap[old]: idx2label[old] for old in old_ids}
label2idx = {v: k for k, v in idx2label.items()}
print(f"  Muestras filtradas: {len(X_all)}  |  LSP - Vocabulario-palabras ≥2: {n_classes}")

# ── Splits estratificados 70/15/15 ───────────────────────────────────────────

sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
tr_idx, tmp_idx = next(sss1.split(X_all, y_all))

# Segundo split: no estratificado (evita LSP - Vocabulario-palabras con 1 muestra en tmp)
rng     = np.random.default_rng(SEED)
perm    = rng.permutation(len(tmp_idx))
half    = len(tmp_idx) // 2
val_idx = tmp_idx[perm[:half]]
te_idx  = tmp_idx[perm[half:]]

X_tr, y_tr = X_all[tr_idx],  y_all[tr_idx]
X_val, y_val = X_all[val_idx], y_all[val_idx]
X_te,  y_te  = X_all[te_idx],  y_all[te_idx]

print(f"  Train={len(X_tr)}  Val={len(X_val)}  Test={len(X_te)}")

# ── Dataset y DataLoader ──────────────────────────────────────────────────────

class SignDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        if self.augment:
            # Ruido gaussiano sobre coordenadas
            x = x + torch.randn_like(x) * 0.008
            # Flip horizontal de manos (x → 1-x para manos, no para pose)
            if torch.rand(1) < 0.5:
                x = x.clone()
                # left_hand x: indices 66-86, right_hand x: 108-128
                x[:, 66:87]  = 1.0 - x[:, 66:87]
                x[:, 108:129] = 1.0 - x[:, 108:129]
        return x, self.y[i]

# WeightedRandomSampler para balancear LSP - Vocabulario-palabras en train
class_counts = Counter(y_tr.tolist())
weights = [1.0 / class_counts[int(c)] for c in y_tr]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

dl_tr  = DataLoader(SignDataset(X_tr,  y_tr,  augment=True),
                    batch_size=BATCH, sampler=sampler,  num_workers=0)
dl_val = DataLoader(SignDataset(X_val, y_val, augment=False),
                    batch_size=BATCH, shuffle=False, num_workers=0)
dl_te  = DataLoader(SignDataset(X_te,  y_te,  augment=False),
                    batch_size=BATCH, shuffle=False, num_workers=0)

# ── Modelo: LSTM Bidireccional + Attention ────────────────────────────────────

class TemporalAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attn = nn.Linear(hidden * 2, 1)

    def forward(self, h):  # h: [B, T, H*2]
        w = torch.softmax(self.attn(h), dim=1)  # [B, T, 1]
        return (w * h).sum(dim=1)               # [B, H*2]


class LSPLSTMBidir(nn.Module):
    def __init__(self, n_dims, n_classes, hidden=HIDDEN, n_layers=N_LAYERS,
                 dropout=DROPOUT):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_dims, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.lstm = nn.LSTM(
            128, hidden, n_layers,
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

    def forward(self, x):          # x: [B, T, 150]
        x = self.proj(x)           # [B, T, 128]
        h, _ = self.lstm(x)        # [B, T, H*2]
        c = self.attn(h)           # [B, H*2]
        return self.head(c)        # [B, n_classes]


model = LSPLSTMBidir(N_DIMS, n_classes).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Modelo: {n_params:,} parámetros")

# ── Entrenamiento ─────────────────────────────────────────────────────────────

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

# Pesos de clase para CrossEntropy
cc = Counter(y_tr.tolist())
cw = torch.tensor([len(y_tr) / (n_classes * cc.get(i, 1))
                    for i in range(n_classes)], dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=cw)

best_f1, best_state, patience_cnt = 0.0, None, 0
hist = {"f1_tr": [], "f1_val": [], "loss_tr": [], "loss_val": []}

print(f"\nEntrenando {N_EPOCHS} épocas (patience={PATIENCE})...")
t0 = time.time()

for epoch in range(1, N_EPOCHS + 1):
    # ── Train ──
    model.train()
    tr_loss, tr_preds, tr_true = 0.0, [], []
    for xb, yb in dl_tr:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tr_loss += loss.item() * len(xb)
        tr_preds.extend(logits.argmax(1).cpu().tolist())
        tr_true.extend(yb.cpu().tolist())
    scheduler.step()

    tr_loss /= len(y_tr)
    f1_tr = f1_score(tr_true, tr_preds, average="macro", zero_division=0)

    # ── Val ──
    model.eval()
    val_loss, val_preds, val_true = 0.0, [], []
    with torch.no_grad():
        for xb, yb in dl_val:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            val_loss += criterion(logits, yb).item() * len(xb)
            val_preds.extend(logits.argmax(1).cpu().tolist())
            val_true.extend(yb.cpu().tolist())
    val_loss /= len(y_val)
    f1_val = f1_score(val_true, val_preds, average="macro", zero_division=0)

    hist["f1_tr"].append(f1_tr)
    hist["f1_val"].append(f1_val)
    hist["loss_tr"].append(tr_loss)
    hist["loss_val"].append(val_loss)

    if epoch % 5 == 0 or epoch == 1:
        elapsed = (time.time() - t0) / 60
        print(f"  Epoch {epoch:3d}/{N_EPOCHS} | "
              f"loss={val_loss:.4f} | F1-val={f1_val:.4f} | "
              f"F1-tr={f1_tr:.4f} | {elapsed:.1f}min")

    if f1_val > best_f1:
        best_f1   = f1_val
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"  Early stopping en época {epoch}")
            break

# ── Evaluación final en test ──────────────────────────────────────────────────

model.load_state_dict(best_state)
model.to(DEVICE).eval()
te_preds, te_true = [], []
with torch.no_grad():
    for xb, yb in dl_te:
        xb = xb.to(DEVICE)
        te_preds.extend(model(xb).argmax(1).cpu().tolist())
        te_true.extend(yb.tolist())

acc_te = (np.array(te_preds) == np.array(te_true)).mean()
f1_te  = f1_score(te_true, te_preds, average="macro",    zero_division=0)
f1_w   = f1_score(te_true, te_preds, average="weighted", zero_division=0)

print(f"\n=== Test ===")
print(f"  Accuracy : {acc_te:.4f}")
print(f"  F1-macro : {f1_te:.4f}")
print(f"  F1-weight: {f1_w:.4f}")
print(f"  Best val F1: {best_f1:.4f}")

# ── Guardar checkpoint PyTorch ────────────────────────────────────────────────

ckpt = {
    "model_state": best_state,
    "label2idx":   label2idx,
    "idx2label":   idx2label,
    "n_classes":   n_classes,
    "n_dims":      N_DIMS,
    "n_frames":    N_FRAMES,
    "hidden":      HIDDEN,
    "n_layers":    N_LAYERS,
    "f1_val":      best_f1,
    "f1_test":     f1_te,
    "acc_test":    acc_te,
    "history":     hist,
}
pt_path = CKPT_DIR / "lstm_signs.pt"
torch.save(ckpt, pt_path)
print(f"\n✅ {pt_path}  ({pt_path.stat().st_size/1e6:.1f} MB)")

# ── Exportar ONNX ─────────────────────────────────────────────────────────────

try:
    model.cpu().eval()
    dummy   = torch.randn(1, N_FRAMES, N_DIMS)
    onnx_p  = CKPT_DIR / "lstm_signs.onnx"
    torch.onnx.export(
        model, dummy, str(onnx_p),
        input_names=["sequence"], output_names=["logits"],
        dynamic_axes={"sequence": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"✅ {onnx_p}  ({onnx_p.stat().st_size/1e6:.1f} MB)")
except Exception as e:
    print(f"⚠️  ONNX export: {e}")

print(f"\n[LSTM COMPLETADO]  F1-val={best_f1:.4f}  F1-test={f1_te:.4f}")
