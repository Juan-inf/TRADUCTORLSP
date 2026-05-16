"""
Entrenamiento CNN-LSTM ligero para dataset LSP con sliding window.
Lee frames on-the-fly desde videos (sin pre-almacenar pixels).

Uso CPU: .venv310/bin/python scripts/run_training.py --epochs 10 --batch_size 4
Uso GPU (Colab): python scripts/run_training.py --epochs 30 --batch_size 16

El modelo usa MobileNetV3-Small como backbone para ser viable en CPU.
"""
import argparse, sys, os, cv2, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Config por defecto ────────────────────────────────────────────────────
DATA_DIR   = Path("data")
CKPT_DIR   = Path("checkpoints")
IMG_SIZE   = 112        # 112 en CPU, 224 en GPU
N_FRAMES   = 30
DEVICE     = 'cuda' if torch.cuda.is_available() else (
             'mps'  if torch.backends.mps.is_available() else 'cpu')

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ── Dataset ───────────────────────────────────────────────────────────────
class LSPSegmentDataset(Dataset):
    def __init__(self, df, label2idx, img_size=IMG_SIZE, augment=False):
        self.df       = df.reset_index(drop=True)
        self.l2i      = label2idx
        self.img_size = img_size
        self.augment  = augment
        self.mean     = np.array(MEAN, dtype=np.float32)
        self.std      = np.array(STD, dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def _load_frames(self, video_path, start, end):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(end - start):
            ok, frame = cap.read()
            if not ok:
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((self.img_size, self.img_size, 3), np.float32))
                continue
            frame = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR
            )
            frame = (frame.astype(np.float32) / 255.0 - self.mean) / self.std
            frames.append(frame)
        cap.release()
        return np.stack(frames)  # [T, H, W, 3]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frames = self._load_frames(row['video_path'], row['start_frame'], row['end_frame'])

        # Augmentación temporal
        if self.augment and np.random.rand() < 0.5:
            frames = frames[::-1].copy()  # flip temporal

        # [T, H, W, 3] → [C, T, H, W] para Conv3D / [T, C, H, W] para CNN-LSTM
        tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float()  # [C, T, H, W]
        label  = self.l2i[row['clase']]
        return tensor, label


# ── Modelo CNN-LSTM ligero ────────────────────────────────────────────────
class LightCNNLSTM(nn.Module):
    def __init__(self, n_classes: int, hidden: int = 256, img_size: int = IMG_SIZE):
        super().__init__()
        # Backbone MobileNetV3-Small (rápido en CPU, ~1ms/frame)
        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        # Extrae features hasta AdaptiveAvgPool → 576 features
        self.feature_dim = 576
        self.backbone = nn.Sequential(*list(backbone.features.children()))
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Proyección
        self.proj = nn.Sequential(
            nn.Linear(self.feature_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.3)
        )
        # LSTM temporal
        self.lstm = nn.LSTM(hidden, hidden // 2, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.3)
        # Clasificador
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, n_classes)
        )
        self._freeze_backbone()

    def _freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def _encode(self, x):
        """Shared encoder: pixels → LSTM embedding [B, hidden]."""
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        feats = self.pool(self.backbone(x)).squeeze(-1).squeeze(-1)
        feats = feats.reshape(B, T, -1)
        feats = self.proj(feats)
        out, _ = self.lstm(feats)
        return out.mean(dim=1)  # [B, hidden]

    def forward(self, x):
        return self.classifier(self._encode(x))

    def get_embedding(self, x):
        """Returns LSTM embedding before classifier: [B, hidden]."""
        return self._encode(x)


# ── Training loop ─────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []
    for x, y in tqdm(loader, desc='  Train', leave=False, mininterval=5):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item()
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    n = len(loader)
    return total_loss / n, accuracy_score(all_labels, all_preds)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    for x, y in tqdm(loader, desc='  Eval', leave=False, mininterval=5):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    n = len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / n, acc, f1


def main(args):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Img size: {args.img_size}x{args.img_size} | Epochs: {args.epochs} | Batch: {args.batch_size}")

    # ── Cargar manifest ───────────────────────────────────────────────────
    df = pd.read_csv(DATA_DIR / "manifest_segments.csv")
    with open(DATA_DIR / "label2idx.json") as f:
        label2idx = json.load(f)
    n_classes = len(label2idx)
    print(f"Clases: {n_classes} | Segmentos: {len(df)}")

    df_train = df[df['split'] == 'train']
    df_val   = df[df['split'] == 'val']
    df_test  = df[df['split'] == 'test']
    print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    # ── Datasets y DataLoaders ────────────────────────────────────────────
    train_ds = LSPSegmentDataset(df_train, label2idx, args.img_size, augment=True)
    val_ds   = LSPSegmentDataset(df_val,   label2idx, args.img_size, augment=False)
    test_ds  = LSPSegmentDataset(df_test,  label2idx, args.img_size, augment=False)

    # WeightedRandomSampler para balancear clases
    class_counts = df_train['clase'].value_counts()
    weights = [1.0 / class_counts[row['clase']] for _, row in df_train.iterrows()]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          num_workers=args.num_workers, pin_memory=(DEVICE != 'cpu'))
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers)

    # ── Modelo ────────────────────────────────────────────────────────────
    model = LightCNNLSTM(n_classes, hidden=256, img_size=args.img_size).to(DEVICE)
    print(f"\nParámetros: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainables")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = torch.cuda.amp.GradScaler() if DEVICE == 'cuda' else None

    best_val_acc, patience_cnt = 0, 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    print("\n{'═'*60}")
    print("ENTRENAMIENTO")
    print("═" * 60)

    for epoch in range(1, args.epochs + 1):
        # Descongelar backbone a mitad del entrenamiento
        if epoch == args.epochs // 2 + 1:
            model.unfreeze_backbone()
            print(f"\n  [Epoch {epoch}] Backbone descongelado — todos los parámetros trainables")
            optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - epoch + 1)

        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_dl, optimizer, criterion, DEVICE, scaler)
        vl_loss, vl_acc, vl_f1 = eval_epoch(model, val_dl, criterion, DEVICE)
        scheduler.step()
        elapsed = time.time() - t0

        for k, v in [('train_loss', tr_loss), ('train_acc', tr_acc),
                     ('val_loss', vl_loss), ('val_acc', vl_acc), ('val_f1', vl_f1)]:
            history[k].append(v)

        flag = ' ← MEJOR' if vl_acc > best_val_acc else ''
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Loss: {tr_loss:.4f}/{vl_loss:.4f} | "
              f"Acc: {tr_acc:.4f}/{vl_acc:.4f} | "
              f"F1: {vl_f1:.4f} | {elapsed:.0f}s{flag}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            patience_cnt = 0
            ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_acc': vl_acc,
                'val_f1': vl_f1,
                'history': history,
                'label2idx': label2idx,
                'args': vars(args),
            }
            torch.save(ckpt, CKPT_DIR / 'cnn_lstm_best.pt')
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"\n  Early stopping (patience={args.patience})")
                break

    # ── Evaluación final en test ──────────────────────────────────────────
    print("\n" + "═" * 60)
    print("EVALUACIÓN EN TEST SET")
    ckpt = torch.load(CKPT_DIR / 'cnn_lstm_best.pt', map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    te_loss, te_acc, te_f1 = eval_epoch(model, test_dl, criterion, DEVICE)
    print(f"  Test Loss: {te_loss:.4f} | Test Acc: {te_acc:.4f} | Test F1-macro: {te_f1:.4f}")
    print(f"  Mejor Val Acc: {best_val_acc:.4f}")
    print(f"\n  Checkpoint guardado: checkpoints/cnn_lstm_best.pt")

    # Exportar ONNX
    try:
        dummy = torch.zeros(1, 3, N_FRAMES, args.img_size, args.img_size, device=DEVICE)
        torch.onnx.export(
            model, dummy, str(CKPT_DIR / 'cnn_lstm_best.onnx'),
            opset_version=17, dynamic_axes={'input': {0: 'batch_size'}},
            input_names=['input'], output_names=['output']
        )
        print(f"  ONNX exportado: checkpoints/cnn_lstm_best.onnx")
    except Exception as e:
        print(f"  ONNX export fallido: {e}")

    print("\n╔══════════════════════════════════════════════╗")
    print(f"║  Acc test:      {te_acc:.4f}                       ║")
    print(f"║  F1-macro test: {te_f1:.4f}                       ║")
    print(f"║  Best val acc:  {best_val_acc:.4f}                       ║")
    print("╚══════════════════════════════════════════════╝")


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',      type=int,   default=10)
    parser.add_argument('--batch_size',  type=int,   default=4)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--patience',    type=int,   default=5)
    parser.add_argument('--img_size',    type=int,   default=112)
    parser.add_argument('--num_workers', type=int,   default=0)
    args = parser.parse_args()
    main(args)
