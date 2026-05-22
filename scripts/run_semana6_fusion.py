"""
Semana 6 — Experimento de Fusión Tardía LSP
============================================
Combina embeddings de ST-GCN v2 y BiLSTM v2 (ambos sobre landmarks).
Carga backbones preentrenados, los congela, y entrena solo la cabeza de fusión.

Estrategias disponibles: concat | attention | weighted_sum

Uso:
  .venv310/bin/python scripts/run_semana6_fusion.py --strategy concat
  .venv310/bin/python scripts/run_semana6_fusion.py --strategy attention --epochs 30
  .venv310/bin/python scripts/run_semana6_fusion.py --strategy weighted_sum --lr 3e-4
"""

import sys, json, time, argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score, classification_report
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.models import STGCN

# ── Args ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Semana 6 — Fusión tardía LSP")
    p.add_argument("--strategy",    type=str,   default="concat",
                   choices=["concat", "attention", "weighted_sum"])
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch",       type=int,   default=64)
    p.add_argument("--lr",          type=float, default=5e-4)
    p.add_argument("--hidden_dim",  type=int,   default=512)
    p.add_argument("--patience",    type=int,   default=10)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--stgcn_ckpt",  type=str,   default="checkpoints/stgcn_v2_best.pt")
    p.add_argument("--bilstm_ckpt", type=str,   default="checkpoints/bilstm_v2_best.pt")
    return p.parse_args()


# ── Dataset ───────────────────────────────────────────────────────────────────

class LandmarkDataset(Dataset):
    def __init__(self, df, label2idx, augment=False, n_frames=30):
        self.df      = df.reset_index(drop=True)
        self.l2i     = label2idx
        self.augment = augment
        self.T       = n_frames
        self.N       = 75

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = np.load(str(row["kp_path"])).astype(np.float32)   # [T, N, 3]
        if self.augment:
            seq = self._augment(seq)
        return torch.from_numpy(seq), self.l2i[row["clase"]]

    def _augment(self, seq):
        T, N, C = seq.shape
        if np.random.rand() < 0.7:
            seq = seq + np.random.normal(0, 0.015, seq.shape).astype(np.float32)
        if np.random.rand() < 0.5:
            seq = seq.copy()
            seq[:, :, 0] *= -1
            l, r = seq[:, :21, :].copy(), seq[:, 21:42, :].copy()
            seq[:, :21, :], seq[:, 21:42, :] = r, l
        if np.random.rand() < 0.6:
            factor  = np.random.uniform(0.7, 1.3)
            new_T   = max(8, int(T * factor))
            new_idx = np.linspace(0, T - 1, new_T, dtype=int)
            seq     = seq[new_idx]
        cur_T = len(seq)
        if cur_T > self.T:
            start = np.random.randint(0, cur_T - self.T)
            seq   = seq[start: start + self.T]
        elif cur_T < self.T:
            seq = np.concatenate([seq, np.zeros((self.T - cur_T, N, C), np.float32)])
        return seq.astype(np.float32)


# ── BiLSTM (misma arquitectura que run_training_v2.py) ───────────────────────

class BiLSTMClassifier(nn.Module):
    def __init__(self, n_classes, n_nodes=75, in_c=3, hidden=128, n_layers=3, dropout=0.4):
        super().__init__()
        input_size = n_nodes * in_c
        self.proj  = nn.Linear(input_size, hidden)
        self.bn    = nn.BatchNorm1d(hidden)
        self.lstm  = nn.LSTM(hidden, hidden, n_layers, batch_first=True,
                             bidirectional=True, dropout=dropout if n_layers > 1 else 0)
        self.attn  = nn.Linear(hidden * 2, 1)
        self.drop  = nn.Dropout(dropout)
        self.fc    = nn.Linear(hidden * 2, n_classes)
        self.emb_dim = hidden * 2

    def forward(self, x):
        B, T, N, C = x.shape
        x = x.reshape(B, T, N * C)
        x = self.proj(x)
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.lstm(x)
        w   = torch.softmax(self.attn(out), dim=1)
        ctx = (out * w).sum(dim=1)
        return self.fc(self.drop(ctx))

    def get_embedding(self, x):
        B, T, N, C = x.shape
        x = x.reshape(B, T, N * C)
        x = self.proj(x)
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.lstm(x)
        w   = torch.softmax(self.attn(out), dim=1)
        return (out * w).sum(dim=1)          # [B, emb_dim]


class STGCNWithEmbedding(nn.Module):
    """Wrapper para extraer embedding de ST-GCN antes de la cabeza de clasificación."""
    def __init__(self, stgcn_model):
        super().__init__()
        self.backbone = stgcn_model

    def get_embedding(self, x):
        # x: [B, T, N, C] → ST-GCN espera [B, C, T, N]
        x = x.permute(0, 3, 1, 2)
        emb = self.backbone.get_embedding(x)   # [B, hidden_channels]
        return emb

    def forward(self, x):
        return self.backbone(x.permute(0, 3, 1, 2))


# ── Módulo de fusión tardía ───────────────────────────────────────────────────

class FusionHead(nn.Module):
    """
    Fusión tardía de dos embeddings.
    strategy: 'concat' | 'attention' | 'weighted_sum'
    """
    def __init__(self, dim_a, dim_b, n_classes, strategy="concat",
                 hidden_dim=512, dropout=0.3):
        super().__init__()
        self.strategy = strategy

        if strategy == "concat":
            self.classifier = nn.Sequential(
                nn.Linear(dim_a + dim_b, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )

        elif strategy == "attention":
            self.proj_a = nn.Linear(dim_a, hidden_dim)
            self.proj_b = nn.Linear(dim_b, hidden_dim)
            self.attn   = nn.MultiheadAttention(hidden_dim, num_heads=8,
                                                 batch_first=True, dropout=dropout)
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )

        elif strategy == "weighted_sum":
            proj_dim = min(dim_a, dim_b, hidden_dim)
            self.proj_a = nn.Linear(dim_a, proj_dim)
            self.proj_b = nn.Linear(dim_b, proj_dim)
            self.gate   = nn.Parameter(torch.tensor(0.5))
            self.classifier = nn.Sequential(
                nn.LayerNorm(proj_dim),
                nn.Dropout(dropout),
                nn.Linear(proj_dim, n_classes),
            )

    def forward(self, emb_a, emb_b):
        if self.strategy == "concat":
            return self.classifier(torch.cat([emb_a, emb_b], dim=-1))

        elif self.strategy == "attention":
            qa  = self.proj_a(emb_a).unsqueeze(1)
            kb  = self.proj_b(emb_b).unsqueeze(1)
            out, _ = self.attn(qa, kb, kb)
            return self.classifier(out.squeeze(1))

        elif self.strategy == "weighted_sum":
            alpha  = torch.sigmoid(self.gate)
            fused  = alpha * self.proj_a(emb_a) + (1 - alpha) * self.proj_b(emb_b)
            return self.classifier(fused)


class FusionModel(nn.Module):
    """Backbones congelados + cabeza de fusión entrenable."""
    def __init__(self, backbone_a, backbone_b, dim_a, dim_b,
                 n_classes, strategy, hidden_dim, dropout):
        super().__init__()
        self.backbone_a = backbone_a
        self.backbone_b = backbone_b
        self.head = FusionHead(dim_a, dim_b, n_classes, strategy, hidden_dim, dropout)

    def forward(self, x):
        with torch.no_grad():
            emb_a = self.backbone_a.get_embedding(x)
            emb_b = self.backbone_b.get_embedding(x)
        return self.head(emb_a, emb_b)


# ── Entrenamiento ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
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
    return (np.mean(losses),
            accuracy_score(trues, preds),
            f1_score(trues, preds, average="macro", zero_division=0))


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    preds, trues = [], []
    for seq, y in loader:
        seq, y = seq.to(device), y.to(device)
        logits = model(seq)
        preds.extend(logits.argmax(1).cpu().numpy())
        trues.extend(y.cpu().numpy())
    return (accuracy_score(trues, preds),
            f1_score(trues, preds, average="macro", zero_division=0),
            np.array(trues), np.array(preds))


# ── Carga de backbones ────────────────────────────────────────────────────────

def load_stgcn(ckpt_path, n_classes, device):
    ckpt    = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hidden  = ckpt.get("hidden", 128)
    n_layers = ckpt.get("n_layers", 6)
    model   = STGCN(n_classes=n_classes, n_nodes=75, in_channels=3,
                    hidden_channels=hidden, num_layers=n_layers)
    state   = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return STGCNWithEmbedding(model).to(device), hidden


def load_bilstm(ckpt_path, n_classes, device):
    ckpt   = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hidden = ckpt.get("hidden", 128)
    model  = BiLSTMClassifier(n_classes=n_classes, hidden=hidden)
    state  = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device), model.emb_dim


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = ("mps" if torch.backends.mps.is_available() else
              "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Estrategia de fusión: {args.strategy}")

    DATA_DIR = ROOT / "data"
    LOG_DIR  = ROOT / "logs";        LOG_DIR.mkdir(exist_ok=True)
    CKPT_DIR = ROOT / "checkpoints"; CKPT_DIR.mkdir(exist_ok=True)

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

    def make_loader(df_s, augment, shuffle):
        ds  = LandmarkDataset(df_s, label2idx, augment=augment)
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

    # Cargar backbones
    stgcn_path  = ROOT / args.stgcn_ckpt
    bilstm_path = ROOT / args.bilstm_ckpt

    if not stgcn_path.exists():
        print(f"[WARN] Checkpoint ST-GCN no encontrado: {stgcn_path}")
        print("  Ejecutar primero: .venv310/bin/python scripts/run_training_v2.py")
        sys.exit(1)
    if not bilstm_path.exists():
        print(f"[WARN] Checkpoint BiLSTM no encontrado: {bilstm_path}")
        print("  Ejecutar primero: .venv310/bin/python scripts/run_training_v2.py")
        sys.exit(1)

    print(f"\nCargando ST-GCN desde {stgcn_path}")
    backbone_stgcn,  dim_stgcn  = load_stgcn(stgcn_path, n_classes, device)
    print(f"Cargando BiLSTM desde {bilstm_path}")
    backbone_bilstm, dim_bilstm = load_bilstm(bilstm_path, n_classes, device)

    # Verificar dims de embedding
    with torch.no_grad():
        dummy = torch.zeros(2, 30, 75, 3, device=device)
        emb_s = backbone_stgcn.get_embedding(dummy)
        emb_b = backbone_bilstm.get_embedding(dummy)
    dim_stgcn  = emb_s.shape[-1]
    dim_bilstm = emb_b.shape[-1]
    print(f"Dim embeddings: ST-GCN={dim_stgcn} | BiLSTM={dim_bilstm}")

    # Modelo de fusión
    model = FusionModel(
        backbone_a=backbone_stgcn,
        backbone_b=backbone_bilstm,
        dim_a=dim_stgcn,
        dim_b=dim_bilstm,
        n_classes=n_classes,
        strategy=args.strategy,
        hidden_dim=args.hidden_dim,
        dropout=0.3,
    ).to(device)

    head_params = sum(p.numel() for p in model.head.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nParámetros cabeza de fusión: {head_params:,}")
    print(f"Parámetros total (backbones congelados): {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.head.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    print(f"\n{'='*65}")
    print(f"Fusión: {args.strategy} | Epochs={args.epochs} | LR={args.lr}")
    print(f"{'='*65}")

    best_f1, best_state, patience_cnt = 0.0, None, 0
    history = defaultdict(list)
    t_start = time.time()
    log_path = LOG_DIR / f"semana6_fusion_{args.strategy}.txt"
    log_lines = []

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1 = train_epoch(model, tr_dl, criterion, optimizer, device)
        vl_acc, vl_f1, _, _    = eval_epoch(model, vl_dl, device)
        scheduler.step()

        for k, v in [("tr_loss", tr_loss), ("tr_acc", tr_acc), ("tr_f1", tr_f1),
                     ("vl_acc", vl_acc), ("vl_f1", vl_f1)]:
            history[k].append(v)

        line = (f"  Ep {ep:3d}/{args.epochs} | "
                f"tr_loss={tr_loss:.4f} tr_f1={tr_f1:.3f} | "
                f"vl_acc={vl_acc:.3f} vl_f1={vl_f1:.3f}")
        print(line)
        log_lines.append(line)

        if vl_f1 > best_f1:
            best_f1   = vl_f1
            best_state = {k: v.cpu().clone() for k, v in model.head.state_dict().items()}
            patience_cnt = 0
            msg = f"  ✓ Nuevo mejor F1-macro val: {best_f1:.4f}"
            print(msg); log_lines.append(msg)
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                msg = f"  Early stopping en época {ep}"
                print(msg); log_lines.append(msg)
                break

    # Evaluación final en test
    model.head.load_state_dict(best_state)
    te_acc, te_f1, y_true, y_pred = eval_epoch(model, te_dl, device)
    elapsed = time.time() - t_start

    sep   = "=" * 65
    lines = [
        sep,
        f"SEMANA 6 — FUSIÓN TARDÍA ({args.strategy.upper()}) — RESULTADOS FINALES",
        f"Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        sep,
        f"Estrategia:      {args.strategy}",
        f"Backbone ST-GCN: {args.stgcn_ckpt}",
        f"Backbone BiLSTM: {args.bilstm_ckpt}",
        f"Épocas:          {ep}",
        f"Tiempo:          {elapsed/60:.1f} min",
        "",
        f"TEST  Accuracy={te_acc:.4f} | F1-macro={te_f1:.4f}",
        f"VAL   F1-macro mejor={best_f1:.4f}",
        "",
        "REPORTE POR CLASE:",
        classification_report(y_true, y_pred, target_names=class_names,
                              zero_division=0, digits=3),
    ]
    output = "\n".join(lines + log_lines)
    log_path.write_text(output)
    print(f"\n{sep}")
    print(f"FINAL Test acc={te_acc:.4f} | F1-macro={te_f1:.4f} | {elapsed/60:.1f} min")
    print(f"Log guardado: {log_path}")

    # Guardar checkpoint cabeza de fusión
    ckpt_path = CKPT_DIR / f"semana6_fusion_{args.strategy}_best.pt"
    torch.save({
        "head_state":     best_state,
        "strategy":       args.strategy,
        "dim_a":          dim_stgcn,
        "dim_b":          dim_bilstm,
        "hidden_dim":     args.hidden_dim,
        "n_classes":      n_classes,
        "test_acc":       te_acc,
        "test_f1_macro":  te_f1,
        "val_f1_best":    best_f1,
        "history":        dict(history),
        "stgcn_ckpt":     str(stgcn_path),
        "bilstm_ckpt":    str(bilstm_path),
        "args":           vars(args),
    }, ckpt_path)
    print(f"Checkpoint: {ckpt_path}")

    # Gráfico de curvas de aprendizaje
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Semana 6 — Fusión tardía ({args.strategy})", fontsize=12)

    axes[0].plot(history["tr_f1"], "--", color="#2196F3", alpha=0.7, label="Train F1")
    axes[0].plot(history["vl_f1"], "-",  color="#2196F3", label="Val F1")
    axes[0].axhline(0.636, color="gray", ls=":", label="ST-GCN v2 solo (0.636)")
    axes[0].set_title("F1-macro"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["tr_acc"], "--", color="#FF5722", alpha=0.7, label="Train Acc")
    axes[1].plot(history["vl_acc"], "-",  color="#FF5722", label="Val Acc")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = ROOT / "data" / f"semana6_fusion_{args.strategy}.png"
    plt.savefig(fig_path, dpi=120)
    print(f"Gráfico: {fig_path}")


if __name__ == "__main__":
    main()
