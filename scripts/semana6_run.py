"""
Sprint Semana 6 — Ejecución completa de fusión tardía LSP.
Carga checkpoints lstm_best.pt + stgcn_best.pt (arquitecturas exactas de run_local_pipeline.py),
entrena cabezas de fusión concat y attention, imprime tabla comparativa y guarda resultados.

Uso: .venv310/bin/python scripts/semana6_run.py
"""
import sys, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent

torch.manual_seed(42)
np.random.seed(42)

device = ("mps"  if torch.backends.mps.is_available() else
          "cuda" if torch.cuda.is_available()          else "cpu")
print(f"Device: {device}")

# ── Datos ─────────────────────────────────────────────────────────────────────
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
LOG_DIR  = ROOT / "logs"; LOG_DIR.mkdir(exist_ok=True)

with open(DATA_DIR / "label2idx.json") as f:
    label2idx = json.load(f)
idx2label   = {v: k for k, v in label2idx.items()}
N_CLASSES   = len(label2idx)
CLASS_NAMES = [idx2label[i] for i in range(N_CLASSES)]

df = pd.read_csv(DATA_DIR / "manifest_segments.csv")
df = df[df["kp_path"].notna() & (df["kp_path"] != "")].reset_index(drop=True)
df_tr = df[df["split"] == "train"]
df_vl = df[df["split"] == "val"]
df_te = df[df["split"] == "test"]
print(f"Train {len(df_tr)} | Val {len(df_vl)} | Test {len(df_te)}")

T_FRAMES, N_KP = 30, 75


class LandmarkDS(Dataset):
    def __init__(self, df, l2i, augment=False):
        self.df = df.reset_index(drop=True)
        self.l2i = l2i
        self.aug = augment

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = ROOT / row["kp_path"]          # ruta absoluta desde raíz del proyecto
        seq  = np.load(str(path)).astype(np.float32)
        T = len(seq)
        if T > T_FRAMES:
            seq = seq[:T_FRAMES]
        elif T < T_FRAMES:
            seq = np.concatenate([seq, np.zeros((T_FRAMES-T, N_KP, 3), np.float32)])
        if self.aug:
            if np.random.rand() < 0.6:
                seq += np.random.normal(0, 0.012, seq.shape).astype(np.float32)
            if np.random.rand() < 0.5:
                seq = seq.copy(); seq[:, :, 0] *= -1
                seq[:, :21], seq[:, 21:42] = seq[:, 21:42].copy(), seq[:, :21].copy()
        return torch.from_numpy(seq), self.l2i[row["clase"]]


def make_loader(df_s, augment=False, batch=64):
    ds  = LandmarkDS(df_s, label2idx, augment=augment)
    smp = None
    if augment:
        cnt = df_s["clase"].value_counts()
        wts = [1.0 / cnt[r["clase"]] for _, r in df_s.iterrows()]
        smp = WeightedRandomSampler(wts, len(wts), replacement=True)
    return DataLoader(ds, batch_size=batch, sampler=smp,
                      shuffle=(not augment), num_workers=0, pin_memory=False)

tr_dl = make_loader(df_tr, augment=True)
vl_dl = make_loader(df_vl)
te_dl = make_loader(df_te)

# ── Arquitecturas exactas de run_local_pipeline.py ───────────────────────────

class LSTMBidir(nn.Module):
    def __init__(self, n_kp=75, n_coords=3, n_clases=26, hidden=128, n_layers=2, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(n_kp * n_coords, 128)
        self.lstm = nn.LSTM(128, hidden, n_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)
        self.attn = nn.Linear(hidden * 2, 1)
        self.fc   = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2, n_clases))
        self.emb_dim = hidden * 2

    def get_embedding(self, x):
        B, T, KP, C = x.shape
        x = F.gelu(self.proj(x.view(B, T, KP * C)))
        h, _ = self.lstm(x)
        w = torch.softmax(self.attn(h), dim=1)
        return (w * h).sum(1)

    def forward(self, x):
        return self.fc(self.get_embedding(x))


class SimpleSTGCN(nn.Module):
    def __init__(self, n_kp=75, n_coords=3, n_clases=26, hidden=128, dropout=0.3):
        super().__init__()
        self.embed = nn.Linear(n_coords, 32)
        self.A     = nn.Parameter(torch.eye(n_kp))
        self.gcn1  = nn.Linear(32, hidden)
        self.gcn2  = nn.Linear(hidden, hidden)
        self.tcn   = nn.Sequential(nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU())
        self.fc    = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_clases))
        self.emb_dim = hidden

    def get_embedding(self, x):
        B, T, KP, C = x.shape
        x = F.gelu(self.embed(x))
        A = F.softmax(self.A, dim=-1)
        xr = x.reshape(B * T, KP, 32)
        xr = F.gelu(self.gcn1(torch.bmm(A.unsqueeze(0).expand(B*T,-1,-1), xr)))
        xr = F.gelu(self.gcn2(torch.bmm(A.unsqueeze(0).expand(B*T,-1,-1), xr)))
        xr = xr.mean(1).reshape(B, T, -1).permute(0, 2, 1)
        return self.tcn(xr).mean(-1)

    def forward(self, x):
        return self.fc(self.get_embedding(x))


# ── Cargar checkpoints ────────────────────────────────────────────────────────
m_lstm  = LSTMBidir().to(device)
ck      = torch.load(CKPT_DIR / "lstm_best.pt", map_location="cpu", weights_only=False)
m_lstm.load_state_dict(ck["state_dict"])
print(f"BiLSTM cargado  — val F1={float(ck['f1_val']):.4f}")

m_stgcn = SimpleSTGCN().to(device)
ck2     = torch.load(CKPT_DIR / "stgcn_best.pt", map_location="cpu", weights_only=False)
m_stgcn.load_state_dict(ck2["state_dict"])
print(f"ST-GCN cargado  — val F1={float(ck2['f1_val']):.4f}")

for p in m_lstm.parameters():  p.requires_grad_(False)
for p in m_stgcn.parameters(): p.requires_grad_(False)

DIM_A, DIM_B = m_lstm.emb_dim, m_stgcn.emb_dim
print(f"Dims embeddings: BiLSTM={DIM_A}  ST-GCN={DIM_B}")

# ── Evaluación de backbones individuales ──────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, label=""):
    model.eval()
    preds, trues = [], []
    for seq, y in loader:
        preds.extend(model(seq.to(device)).argmax(1).cpu().numpy())
        trues.extend(y.numpy())
    acc = accuracy_score(trues, preds)
    f1  = f1_score(trues, preds, average="macro",     zero_division=0)
    f1w = f1_score(trues, preds, average="weighted",  zero_division=0)
    if label:
        print(f"  {label:<32}  acc={acc:.4f}  F1-macro={f1:.4f}  F1-w={f1w:.4f}")
    return acc, f1, f1w, np.array(trues), np.array(preds)

print("\nEvaluación TEST (backbones individuales):")
acc_l, f1_l, f1w_l, yt_l, yp_l = evaluate(m_lstm,  te_dl, "BiLSTM (backbone A)")
acc_s, f1_s, f1w_s, yt_s, yp_s = evaluate(m_stgcn, te_dl, "ST-GCN (backbone B)")

results = {
    "BiLSTM (baseline)": {"acc": acc_l, "f1": f1_l, "f1w": f1w_l,
                          "y_true": yt_l, "y_pred": yp_l, "hist": None},
    "ST-GCN (baseline)": {"acc": acc_s, "f1": f1_s, "f1w": f1w_s,
                          "y_true": yt_s, "y_pred": yp_s, "hist": None},
}

# ── Módulo de combinación tardía ──────────────────────────────────────────────

class CabezaCombinada(nn.Module):
    def __init__(self, dim_a, dim_b, n_classes, strategy="concat", hidden=256, dropout=0.35):
        super().__init__()
        self.strategy = strategy
        if strategy == "concat":
            self.net = nn.Sequential(
                nn.Linear(dim_a + dim_b, hidden), nn.LayerNorm(hidden),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))
        elif strategy == "attention":
            self.pa  = nn.Linear(dim_a, hidden)
            self.pb  = nn.Linear(dim_b, hidden)
            self.att = nn.MultiheadAttention(hidden, num_heads=4,
                                             batch_first=True, dropout=dropout)
            self.net = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout),
                                     nn.Linear(hidden, n_classes))

    def forward(self, ea, eb):
        if self.strategy == "concat":
            return self.net(torch.cat([ea, eb], dim=-1))
        elif self.strategy == "attention":
            q, k = self.pa(ea).unsqueeze(1), self.pb(eb).unsqueeze(1)
            out, _ = self.att(q, k, k)
            return self.net(out.squeeze(1))


class ModeloCombinado(nn.Module):
    def __init__(self, bb_a, bb_b, strategy, hidden=256, dropout=0.35):
        super().__init__()
        self.bb_a = bb_a
        self.bb_b = bb_b
        self.head = CabezaCombinada(bb_a.emb_dim, bb_b.emb_dim,
                               N_CLASSES, strategy, hidden, dropout)

    def forward(self, x):
        with torch.no_grad():
            ea = self.bb_a.get_embedding(x)
            eb = self.bb_b.get_embedding(x)
        return self.head(ea, eb)


def run_variante(strategy, epochs=25, lr=5e-4, patience=8):
    model = ModeloCombinado(m_lstm, m_stgcn, strategy).to(device)
    opt   = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_f1, best_state, no_imp = 0.0, None, 0
    hist = {"tr_f1": [], "vl_f1": [], "vl_acc": []}
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        p_all, t_all = [], []
        for seq, y in tr_dl:
            seq, y = seq.to(device), y.to(device)
            opt.zero_grad()
            out  = model(seq)
            loss = crit(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                p_all.extend(model(seq).argmax(1).cpu().numpy())
            t_all.extend(y.cpu().numpy())
        sch.step()
        tr_f1 = f1_score(t_all, p_all, average="macro", zero_division=0)

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for seq, y in vl_dl:
                vp.extend(model(seq.to(device)).argmax(1).cpu().numpy())
                vt.extend(y.numpy())
        vl_acc = accuracy_score(vt, vp)
        vl_f1  = f1_score(vt, vp, average="macro", zero_division=0)
        hist["tr_f1"].append(tr_f1)
        hist["vl_f1"].append(vl_f1)
        hist["vl_acc"].append(vl_acc)

        mark = ""
        if vl_f1 > best_f1:
            best_f1 = vl_f1
            best_state = {k: v.cpu().clone() for k, v in model.head.state_dict().items()}
            no_imp = 0; mark = " ✓"
        else:
            no_imp += 1
        print(f"  ep {ep:2d}/{epochs} | tr_f1={tr_f1:.3f} vl_f1={vl_f1:.3f} vl_acc={vl_acc:.3f}{mark}")
        if no_imp >= patience:
            print(f"  Early stop ep={ep}"); break

    model.head.load_state_dict(best_state)
    acc_te, f1_te, f1w_te, y_true, y_pred = evaluate(model, te_dl)
    print(f"\n  TEST acc={acc_te:.4f}  F1-macro={f1_te:.4f}  F1-w={f1w_te:.4f}  ({(time.time()-t0)/60:.1f} min)")

    ckpt_path = CKPT_DIR / f"semana6_{strategy}_best.pt"
    torch.save({"head_state": best_state, "strategy": strategy,
                "dim_a": DIM_A, "dim_b": DIM_B, "n_classes": N_CLASSES,
                "test_f1_macro": f1_te, "test_acc": acc_te, "val_f1_best": best_f1}, ckpt_path)
    print(f"  Checkpoint: {ckpt_path}")

    return {"acc": acc_te, "f1": f1_te, "f1w": f1w_te,
            "y_true": y_true, "y_pred": y_pred,
            "hist": hist, "best_val_f1": best_f1}


# ── Experimentos ──────────────────────────────────────────────────────────────
SEP = "=" * 60

print(f"\n{SEP}\nVariante A — Combinación CONCAT  (BiLSTM {DIM_A}d + ST-GCN {DIM_B}d)\n{SEP}")
results["Combinación concat"]   = run_variante("concat",    epochs=25, lr=5e-4)

print(f"\n{SEP}\nVariante B — Combinación ATTENTION  (cross-attn 4 heads)\n{SEP}")
results["Combinación attention"] = run_variante("attention", epochs=25, lr=3e-4)

# ── Tabla comparativa ─────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("SPRINT SEMANA 6 — TABLA COMPARATIVA FINAL")
print(f"{'='*65}")
print(f"{'Modelo':<32} {'Acc':>7} {'F1-macro':>9} {'F1-weighted':>12}")
print("-" * 65)
for name, r in sorted(results.items(), key=lambda x: -x[1]["f1"]):
    print(f"  {name:<30} {r['acc']:>7.4f} {r['f1']:>9.4f} {r['f1w']:>12.4f}")

# Reporte por clase — mejor variante
mejor_variante = max((k for k in results if "baseline" not in k),
                     key=lambda k: results[k]["f1"])
print(f"\nMejor variante Semana 6: {mejor_variante}  (F1-macro={results[mejor_variante]['f1']:.4f})")
print(classification_report(results[mejor_variante]["y_true"],
                             results[mejor_variante]["y_pred"],
                             target_names=CLASS_NAMES, zero_division=0, digits=3))

# ── Guardar log ───────────────────────────────────────────────────────────────
import datetime
log_lines = [
    "=" * 65,
    "SPRINT SEMANA 6 — RESULTADOS FINALES COMBINACIÓN TARDÍA LSP",
    f"Fecha: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
    "=" * 65, "",
    f"{'Modelo':<32} {'Acc':>7} {'F1-macro':>9} {'F1-weighted':>12}",
    "-" * 65,
]
for name, r in sorted(results.items(), key=lambda x: -x[1]["f1"]):
    log_lines.append(f"  {name:<30} {r['acc']:>7.4f} {r['f1']:>9.4f} {r['f1w']:>12.4f}")
log_lines += ["", "Reporte por clase — mejor variante:",
              classification_report(results[mejor_variante]["y_true"],
                                    results[mejor_variante]["y_pred"],
                                    target_names=CLASS_NAMES, zero_division=0, digits=3)]
(LOG_DIR / "semana6_resultados_finales.txt").write_text("\n".join(log_lines))
print(f"\nLog: logs/semana6_resultados_finales.txt")

# ── CSV comparativo ───────────────────────────────────────────────────────────
rows = [{"Modelo": n, "F1-macro": r["f1"], "F1-weighted": r["f1w"], "Accuracy": r["acc"]}
        for n, r in results.items()]
pd.DataFrame(rows).set_index("Modelo").sort_values("F1-macro", ascending=False
    ).to_csv(DATA_DIR / "semana6_resultados.csv")
print("CSV: data/semana6_resultados.csv")

# ── Gráfico ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.suptitle("Sprint Semana 6 — Combinación tardía LSP", fontsize=12)
colors = {"Combinación concat": "#2196F3", "Combinación attention": "#FF5722"}
for name, col in colors.items():
    if name not in results or results[name]["hist"] is None: continue
    h  = results[name]["hist"]
    ep = range(1, len(h["vl_f1"]) + 1)
    axes[0].plot(ep, h["tr_f1"], "--", color=col, alpha=0.6, label=f"{name} train")
    axes[0].plot(ep, h["vl_f1"], "-",  color=col, lw=2,  label=f"{name} val")
    axes[1].plot(ep, h["vl_acc"], "-", color=col, lw=2,  label=name)

for ax in axes:
    ax.axhline(f1_l, color="gray",  ls=":", lw=1.5, label=f"BiLSTM solo ({f1_l:.3f})")
    ax.axhline(f1_s, color="#888",  ls="-.", lw=1.2, label=f"ST-GCN solo ({f1_s:.3f})")
    ax.legend(fontsize=7.5); ax.grid(True, alpha=0.3)
axes[0].set_title("F1-macro"); axes[0].set_xlabel("Época")
axes[1].set_title("Accuracy val"); axes[1].set_xlabel("Época")
plt.tight_layout()
fig.savefig(DATA_DIR / "semana6_curvas.png", dpi=130, bbox_inches="tight")
print("Gráfico: data/semana6_curvas.png")
