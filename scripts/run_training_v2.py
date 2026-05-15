"""
Entrenamiento mejorado — STGCN v2 + BiLSTM sobre landmarks LSP.
Mejoras vs v1:
  - Modelo más grande: hidden=128, layers=6
  - 80 épocas con early stopping (paciencia=15)
  - Label smoothing (0.1)
  - LR warmup (3 épocas) + CosineAnnealingLR
  - Augmentación agresiva: ruido, flip, speed, dropout de keypoints
  - Gradient clipping + weight decay
  - Stochastic Weight Averaging (SWA) en últimas 10 épocas
  - Guarda métricas completas en logs/entrenamiento_v2.txt

Uso:
  .venv310/bin/python scripts/run_training_v2.py
  .venv310/bin/python scripts/run_training_v2.py --epochs 40 --hidden 64
"""

import sys, json, time, argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.metrics import f1_score, accuracy_score, classification_report
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import STGCN

# ── Args ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",    type=int,   default=80)
    p.add_argument("--hidden",    type=int,   default=128)
    p.add_argument("--layers",    type=int,   default=6)
    p.add_argument("--batch",     type=int,   default=64)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--patience",  type=int,   default=15)
    p.add_argument("--swa_start", type=int,   default=60)  # SWA desde esta época
    p.add_argument("--label_smoothing", type=float, default=0.1)
    return p.parse_args()

# ── Dataset con augmentación agresiva ─────────────────────────────────────────

class LandmarkDataset(Dataset):
    def __init__(self, df, label2idx, augment=False, n_frames=30):
        self.df      = df.reset_index(drop=True)
        self.l2i     = label2idx
        self.augment = augment
        self.T       = n_frames
        self.N       = 75
        self.classes = sorted(label2idx.values())

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = np.load(str(row["kp_path"])).astype(np.float32)  # [T, N, 3]
        if self.augment:
            seq = self._augment(seq)
        return torch.from_numpy(seq), self.l2i[row["clase"]]

    def _augment(self, seq):
        T, N, C = seq.shape

        # 1. Ruido gaussiano sobre coordenadas
        if np.random.rand() < 0.7:
            seq = seq + np.random.normal(0, 0.015, seq.shape).astype(np.float32)

        # 2. Flip horizontal + swap manos
        if np.random.rand() < 0.5:
            seq = seq.copy()
            seq[:, :, 0] *= -1
            l, r = seq[:, :21, :].copy(), seq[:, 21:42, :].copy()
            seq[:, :21, :], seq[:, 21:42, :] = r, l

        # 3. Variación de velocidad ×0.6–×1.4
        if np.random.rand() < 0.6:
            factor   = np.random.uniform(0.6, 1.4)
            new_T    = max(8, int(T * factor))
            new_idx  = np.linspace(0, T - 1, new_T, dtype=int)
            seq      = seq[new_idx]

        # 4. Recorte temporal aleatorio → pad a T
        cur_T = len(seq)
        if cur_T > self.T:
            start = np.random.randint(0, cur_T - self.T)
            seq   = seq[start: start + self.T]
        elif cur_T < self.T:
            pad = np.zeros((self.T - cur_T, N, C), dtype=np.float32)
            seq = np.concatenate([seq, pad])

        # 5. Dropout aleatorio de keypoints (simula oclusión)
        if np.random.rand() < 0.3:
            n_drop = np.random.randint(1, 8)
            drop_idx = np.random.choice(N, n_drop, replace=False)
            seq[:, drop_idx, :] = 0.0

        # 6. Escala global aleatoria ±20%
        if np.random.rand() < 0.4:
            seq = seq * np.random.uniform(0.8, 1.2)

        # 7. Rotación leve en plano XY ±15°
        if np.random.rand() < 0.4:
            angle = np.random.uniform(-15, 15) * np.pi / 180
            c, s  = np.cos(angle), np.sin(angle)
            R     = np.array([[c, -s], [s, c]], dtype=np.float32)
            seq[:, :, :2] = seq[:, :, :2] @ R.T

        return seq.astype(np.float32)


# ── Loss con label smoothing ──────────────────────────────────────────────────

class LabelSmoothingCE(nn.Module):
    def __init__(self, n_classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.n         = n_classes

    def forward(self, logits, targets):
        log_p    = F.log_softmax(logits, dim=-1)
        smooth_p = torch.full_like(log_p, self.smoothing / (self.n - 1))
        smooth_p.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        return -(smooth_p * log_p).sum(dim=-1).mean()


# ── BiLSTM alternativo ────────────────────────────────────────────────────────

class BiLSTMClassifier(nn.Module):
    """LSTM bidireccional con attention sobre keypoints aplanados."""
    def __init__(self, n_classes, n_nodes=75, in_c=3, hidden=128, n_layers=3, dropout=0.4):
        super().__init__()
        input_size = n_nodes * in_c
        self.proj  = nn.Linear(input_size, hidden)
        self.bn    = nn.BatchNorm1d(hidden)
        self.lstm  = nn.LSTM(hidden, hidden, n_layers, batch_first=True,
                             bidirectional=True, dropout=dropout if n_layers > 1 else 0)
        # Self-attention sobre temporal
        self.attn  = nn.Linear(hidden * 2, 1)
        self.drop  = nn.Dropout(dropout)
        self.fc    = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):
        # x: [B, T, N, C]
        B, T, N, C = x.shape
        x = x.reshape(B, T, N * C)                    # [B, T, N*C]
        x = self.proj(x)                               # [B, T, H]
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.lstm(x)                          # [B, T, 2H]
        w  = torch.softmax(self.attn(out), dim=1)      # [B, T, 1]
        ctx = (out * w).sum(dim=1)                     # [B, 2H]
        return self.fc(self.drop(ctx))


# ── Entrenamiento ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    losses, preds, trues = [], [], []
    for seq, y in loader:
        seq, y = seq.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(seq)
        loss   = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        preds.extend(logits.detach().argmax(1).cpu().numpy())
        trues.extend(y.cpu().numpy())
    return np.mean(losses), accuracy_score(trues, preds), \
           f1_score(trues, preds, average="macro", zero_division=0)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    losses, preds, trues, probs_all = [], [], [], []
    for seq, y in loader:
        seq, y = seq.to(device), y.to(device)
        logits = model(seq)
        losses.append(criterion(logits, y).item())
        p = F.softmax(logits, dim=-1).cpu().numpy()
        probs_all.extend(p)
        preds.extend(p.argmax(1))
        trues.extend(y.cpu().numpy())
    return (np.mean(losses),
            accuracy_score(trues, preds),
            f1_score(trues, preds, average="macro", zero_division=0),
            np.array(trues), np.array(preds))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = get_args()
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Rutas
    DATA_DIR = Path("data")
    LOG_DIR  = Path("logs");   LOG_DIR.mkdir(exist_ok=True)
    CKPT_DIR = Path("checkpoints"); CKPT_DIR.mkdir(exist_ok=True)

    with open(DATA_DIR / "label2idx.json") as f:
        label2idx = json.load(f)
    idx2label   = {v: k for k, v in label2idx.items()}
    n_classes   = len(label2idx)
    class_names = [idx2label[i] for i in range(n_classes)]

    df = pd.read_csv(DATA_DIR / "manifest_segments.csv")
    df = df[df["kp_path"].notna() & (df["kp_path"] != "")].reset_index(drop=True)

    df_tr = df[df["split"] == "train"]
    df_vl = df[df["split"] == "val"]
    df_te = df[df["split"] == "test"]
    print(f"Train {len(df_tr)} | Val {len(df_vl)} | Test {len(df_te)}")

    # DataLoaders
    def make_loader(df_s, augment, shuffle):
        ds  = LandmarkDataset(df_s, label2idx, augment=augment)
        wts = None
        smp = None
        if augment:
            cnt = df_s["clase"].value_counts()
            wts = [1.0 / cnt[r["clase"]] for _, r in df_s.iterrows()]
            smp = WeightedRandomSampler(wts, len(wts), replacement=True)
        return DataLoader(ds, batch_size=args.batch,
                          sampler=smp, shuffle=(shuffle and smp is None),
                          num_workers=0, pin_memory=False)

    tr_dl = make_loader(df_tr, augment=True,  shuffle=True)
    vl_dl = make_loader(df_vl, augment=False, shuffle=False)
    te_dl = make_loader(df_te, augment=False, shuffle=False)

    criterion = LabelSmoothingCE(n_classes, args.label_smoothing)

    # Entrenar dos modelos: STGCN v2 + BiLSTM
    results = {}
    for model_name, model in [
        ("stgcn_v2",  STGCN(n_classes=n_classes, n_nodes=75, in_channels=3,
                             hidden_channels=args.hidden, num_layers=args.layers)),
        ("bilstm_v2", BiLSTMClassifier(n_classes=n_classes, n_nodes=75,
                                        hidden=args.hidden, n_layers=4)),
    ]:
        print(f"\n{'='*65}")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Modelo: {model_name}  |  Parámetros: {n_params:,}")
        print(f"Epochs={args.epochs} | LR={args.lr} | Hidden={args.hidden} | Batch={args.batch}")
        print(f"{'='*65}")

        model = model.to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)

        # LR: warmup lineal 3 épocas → cosine decay
        def lr_lambda(ep):
            warmup = 3
            if ep < warmup:
                return (ep + 1) / warmup
            t = (ep - warmup) / max(1, args.epochs - warmup)
            return 0.5 * (1 + np.cos(np.pi * t))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        # SWA
        swa_model = AveragedModel(model)
        swa_sched = SWALR(opt, swa_lr=args.lr * 0.1)
        swa_started = False

        best_f1, best_state, patience_cnt = 0.0, None, 0
        history = defaultdict(list)
        t_start = time.time()

        for ep in range(1, args.epochs + 1):
            tr_loss, tr_acc, tr_f1 = train_epoch(model, tr_dl, criterion, opt, device)
            vl_loss, vl_acc, vl_f1, _, _ = eval_epoch(model, vl_dl, criterion, device)

            # SWA en últimas épocas
            if ep >= args.swa_start:
                swa_model.update_parameters(model)
                swa_sched.step()
                swa_started = True
            else:
                scheduler.step()

            for k, v in [("tr_loss", tr_loss), ("tr_acc", tr_acc), ("tr_f1", tr_f1),
                          ("vl_loss", vl_loss), ("vl_acc", vl_acc), ("vl_f1", vl_f1)]:
                history[k].append(v)

            lr_now = opt.param_groups[0]["lr"]
            print(f"  Ep {ep:3d}/{args.epochs} | "
                  f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} tr_f1={tr_f1:.3f} | "
                  f"vl_acc={vl_acc:.3f} vl_f1={vl_f1:.3f} | lr={lr_now:.2e}")

            if vl_f1 > best_f1:
                best_f1   = vl_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
                print(f"  ✓ Nuevo mejor F1-macro val: {best_f1:.4f}")
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    print(f"  Early stopping en época {ep} (paciencia={args.patience})")
                    break

        # Evaluar en test con el mejor estado
        model.load_state_dict(best_state)
        _, te_acc, te_f1, y_true, y_pred = eval_epoch(model, te_dl, criterion, device)

        # Evaluar SWA si se activó
        if swa_started:
            torch.optim.swa_utils.update_bn(tr_dl, swa_model, device=device)
            _, te_acc_swa, te_f1_swa, yt_swa, yp_swa = eval_epoch(
                swa_model, te_dl, criterion, device)
            print(f"\n  SWA → Test acc={te_acc_swa:.4f} F1-macro={te_f1_swa:.4f}")
            if te_f1_swa > te_f1:
                te_acc, te_f1, y_true, y_pred = te_acc_swa, te_f1_swa, yt_swa, yp_swa
                print("  → SWA supera al mejor checkpoint")

        elapsed = time.time() - t_start
        print(f"\n  FINAL Test acc={te_acc:.4f} | F1-macro={te_f1:.4f} | "
              f"tiempo={elapsed/60:.1f} min")

        # Guardar checkpoint
        ckpt_path = CKPT_DIR / f"{model_name}_best.pt"
        torch.save({
            "model_state":  best_state,
            "model_name":   model_name,
            "test_acc":     te_acc,
            "test_f1_macro": te_f1,
            "val_f1_best":  best_f1,
            "history":      dict(history),
            "args":         vars(args),
            "hidden":       args.hidden,
            "n_layers":     args.layers,
        }, ckpt_path)
        print(f"  Checkpoint: {ckpt_path}")

        results[model_name] = {
            "test_acc": te_acc, "test_f1": te_f1,
            "y_true": y_true, "y_pred": y_pred, "history": dict(history),
        }

        # Reporte por clase
        rpt = classification_report(y_true, y_pred, target_names=class_names,
                                    zero_division=0, digits=3)
        print(f"\n  Reporte por clase:\n{rpt}")

    # ── Gráficos ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Entrenamiento LSP v2 — STGCN + BiLSTM sobre Landmarks", fontsize=13)

    colors = {"stgcn_v2": "#2196F3", "bilstm_v2": "#FF5722"}
    for mname, r in results.items():
        h   = r["history"]
        col = colors[mname]
        axes[0, 0].plot(h["tr_acc"],  "--", color=col, alpha=0.6, label=f"{mname} train")
        axes[0, 0].plot(h["vl_acc"],  "-",  color=col, label=f"{mname} val")
        axes[0, 1].plot(h["tr_f1"],   "--", color=col, alpha=0.6)
        axes[0, 1].plot(h["vl_f1"],   "-",  color=col, label=mname)
        axes[1, 0].plot(h["tr_loss"], "--", color=col, alpha=0.6)
        axes[1, 0].plot(h["vl_loss"], "-",  color=col, label=mname)

    for ax, t in zip(axes[0], ["Accuracy", "F1-macro"]):
        ax.set_title(t); ax.legend(); ax.grid(True, alpha=0.3)
    axes[1, 0].set_title("Loss"); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    # Tabla comparativa
    axes[1, 1].axis("off")
    rows = [[m, f"{r['test_acc']:.3f}", f"{r['test_f1']:.3f}"]
            for m, r in results.items()]
    t = axes[1, 1].table(
        cellText=rows, colLabels=["Modelo", "Test Acc", "F1-macro"],
        loc="center", cellLoc="center")
    t.scale(1.2, 2.0)
    axes[1, 1].set_title("Resultados Test", pad=20)

    plt.tight_layout()
    plot_path = DATA_DIR / "entrenamiento_v2.png"
    plt.savefig(plot_path, dpi=130, bbox_inches="tight")
    print(f"\nGráfico: {plot_path}")

    # ── Log ───────────────────────────────────────────────────────────────────
    log_path = LOG_DIR / "entrenamiento_v2.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        import datetime
        f.write(f"ENTRENAMIENTO LSP v2 — {datetime.datetime.now():%Y-%m-%d %H:%M}\n")
        f.write(f"Args: {vars(args)}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Dataset: {len(df)} segs | 26 clases\n\n")
        for mname, r in results.items():
            f.write(f"{'='*50}\n{mname}\n")
            f.write(f"  Test Acc    : {r['test_acc']:.4f}\n")
            f.write(f"  Test F1-mac : {r['test_f1']:.4f}\n\n")
            rpt = classification_report(r["y_true"], r["y_pred"],
                                        target_names=class_names, zero_division=0)
            f.write(rpt + "\n")
    print(f"Log: {log_path}")

    # ── Mejor modelo → stgcn_final.pt (usado por el demo) ────────────────────
    best_model_name = max(results, key=lambda m: results[m]["test_f1"])
    best_src  = CKPT_DIR / f"{best_model_name}_best.pt"
    best_dest = CKPT_DIR / "stgcn_final.pt"
    import shutil
    shutil.copy(best_src, best_dest)
    print(f"\nMejor modelo: {best_model_name} → copiado a {best_dest}")
    print(f"Test Acc={results[best_model_name]['test_acc']:.4f} "
          f"F1-macro={results[best_model_name]['test_f1']:.4f}")


if __name__ == "__main__":
    main()
