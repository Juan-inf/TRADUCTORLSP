#!/usr/bin/env python3
"""
Pipeline local completo — Traductor LSP
Usa: /usr/local/bin/python3.10 scripts/run_local_pipeline.py

Datos esperados en data/:
  manifest_segments.csv   — splits + rutas de .npy
  landmarks/*.npy         — (30, 75, 3)  float32
  label2idx.json
"""

import os, sys, json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# ── Configuración ─────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR  = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

N_FRAMES  = 30
N_KP      = 75
N_COORDS  = 3
BATCH     = 32
N_EPOCHS  = 40
LR        = 1e-3
PATIENCE  = 8
DEVICE    = "cpu"   # CPU — sin GPU local

print(f"Directorio raíz: {ROOT}")
print(f"Device: {DEVICE} | torch {torch.__version__}")


# ── 1. CARGAR DATOS ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PASO 1 — Cargando datos")
print("="*60)

df = pd.read_csv(DATA_DIR / "manifest_segments.csv")
print(f"manifest_segments: {df.shape}")
print(df["split"].value_counts().to_string())
print(f"Clases únicas: {df['clase'].nunique()}")

# Cargar landmarks
def cargar_split(df_split, label_encoder):
    X, y = [], []
    faltantes = 0
    for _, row in df_split.iterrows():
        kp_path = ROOT / row["kp_path"]
        if not kp_path.exists():
            faltantes += 1
            continue
        kp = np.load(kp_path)                        # (30, 75, 3)
        kp = kp.astype(np.float32)
        # Normalizar respecto al punto 0 (muñeca izquierda)
        ref = kp[:, 0:1, :]
        kp  = kp - ref
        X.append(kp)
        y.append(label_encoder.transform([row["clase"]])[0])
    if faltantes:
        print(f"  ⚠️  {faltantes} archivos no encontrados")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

le = LabelEncoder()
le.fit(df["clase"].unique())
N_CLASES = len(le.classes_)
print(f"\nN_CLASES: {N_CLASES} | {le.classes_.tolist()[:5]}...")

df_train = df[df["split"] == "train"]
df_val   = df[df["split"] == "val"]
df_test  = df[df["split"] == "test"]

print("\nCargando train...")
X_train, y_train = cargar_split(df_train, le)
print(f"  → {X_train.shape}")

print("Cargando val...")
X_val, y_val = cargar_split(df_val, le)
print(f"  → {X_val.shape}")

print("Cargando test...")
X_test, y_test = cargar_split(df_test, le)
print(f"  → {X_test.shape}")

# Guardar label2idx actualizado
label2idx = {c: int(i) for i, c in enumerate(le.classes_)}
idx2label = {v: k for k, v in label2idx.items()}
with open(DATA_DIR / "label2idx.json", "w") as f:
    json.dump(label2idx, f, ensure_ascii=False, indent=2)
print("\n✅ label2idx.json actualizado")


# ── 2. EDA ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PASO 2 — EDA")
print("="*60)

conteo = Counter(df["clase"])
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle("EDA — Dataset LSP (Landmarks locales)", fontweight="bold")

# Distribución de LSP - Vocabulario-palabras
sorted_clases = sorted(conteo.items(), key=lambda x: -x[1])
axes[0].barh([x[0] for x in sorted_clases], [x[1] for x in sorted_clases], color="steelblue")
axes[0].set_title("Segmentos por viñeta")
axes[0].set_xlabel("N° segmentos")

# Split distribution
splits = df.groupby(["clase", "split"]).size().unstack(fill_value=0)
splits.plot(kind="barh", ax=axes[1], colormap="Set2")
axes[1].set_title("Distribución train/val/test por clase")
axes[1].set_xlabel("N° segmentos")

plt.tight_layout()
eda_path = OUT_DIR / "eda_local.png"
plt.savefig(eda_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"✅ Gráfica EDA guardada: {eda_path}")

# Riesgos
total = len(df)
print(f"\n⚠️  Análisis de riesgos:")
print(f"   Desbalance — max/min: {max(conteo.values())}/{min(conteo.values())} = {max(conteo.values())/min(conteo.values()):.1f}x")
print(f"   Leakage: split por segmentos solapados (stride=15 frames) — verificar")


# ── 3. BASELINE ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PASO 3 — Baseline (KNN, NaiveBayes, LogReg)")
print("="*60)

X_tr_flat = X_train.reshape(len(X_train), -1)
X_te_flat = X_test.reshape(len(X_test), -1)

# Submuestrear si necesario
MAX_BASE = 3000
if len(X_tr_flat) > MAX_BASE:
    np.random.seed(42)
    sel = np.random.choice(len(X_tr_flat), MAX_BASE, replace=False)
    X_tr_b, y_tr_b = X_tr_flat[sel], y_train[sel]
    print(f"  (submuestreado a {MAX_BASE} para KNN)")
else:
    X_tr_b, y_tr_b = X_tr_flat, y_train

baseline_modelos = {
    "KNN (k=5)":      KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "Naive Bayes":    GaussianNB(),
    "Reg. Logística": LogisticRegression(max_iter=300, C=1.0, solver="lbfgs", n_jobs=-1),
}

baseline_res = {}
for nombre, modelo in baseline_modelos.items():
    t0 = time.time()
    modelo.fit(X_tr_b, y_tr_b)
    t_train = time.time() - t0
    y_pred  = modelo.predict(X_te_flat)
    f1m  = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    f1w  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    acc  = (y_pred == y_test).mean()
    baseline_res[nombre] = {"accuracy": acc, "f1_macro": f1m, "f1_weighted": f1w,
                             "t_train_s": t_train}
    print(f"  {nombre}: Acc={acc:.3f} F1-macro={f1m:.3f} F1-w={f1w:.3f} ({t_train:.1f}s)")

mejor_base = max(baseline_res, key=lambda k: baseline_res[k]["f1_macro"])
mejor_f1_base = baseline_res[mejor_base]["f1_macro"]
print(f"\n  Mejor baseline: {mejor_base} (F1-macro={mejor_f1_base:.3f})")

df_base = pd.DataFrame(baseline_res).T.round(4)
df_base.to_csv(OUT_DIR / "baseline_results_local.csv")
print(f"✅ baseline_results_local.csv guardado")


# ── 4. DEEP LEARNING ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PASO 4 — Deep Learning (LSTM Bidir + ST-GCN)")
print("="*60)

# Pesos de clase
conteo_y  = Counter(y_train.tolist())
class_weights = torch.tensor(
    [len(y_train) / (N_CLASES * conteo_y.get(i, 1)) for i in range(N_CLASES)],
    dtype=torch.float32,
)

class LSPDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

dl_train = DataLoader(LSPDataset(X_train, y_train), batch_size=BATCH, shuffle=True,  num_workers=0)
dl_val   = DataLoader(LSPDataset(X_val,   y_val),   batch_size=BATCH, shuffle=False, num_workers=0)
dl_test  = DataLoader(LSPDataset(X_test,  y_test),  batch_size=BATCH, shuffle=False, num_workers=0)

# MODELO 1 — LSTM Bidireccional + Attention
class LSTMBidir(nn.Module):
    def __init__(self, n_kp, n_coords, n_clases, hidden=128, n_layers=2, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(n_kp * n_coords, 128)
        self.lstm = nn.LSTM(128, hidden, n_layers, batch_first=True,
                             bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)
        self.attn = nn.Linear(hidden * 2, 1)
        self.fc   = nn.Sequential(nn.Dropout(dropout),
                                   nn.Linear(hidden * 2, n_clases))

    def forward(self, x):             # x: [B, T, KP, 3]
        B, T, KP, C = x.shape
        x = F.gelu(self.proj(x.view(B, T, KP * C)))  # [B, T, 128]
        h, _ = self.lstm(x)                            # [B, T, H*2]
        w = torch.softmax(self.attn(h), dim=1)        # [B, T, 1]
        return self.fc((w * h).sum(1))                 # [B, N_CLASES]

# MODELO 2 — ST-GCN
class STGCN(nn.Module):
    def __init__(self, n_kp, n_coords, n_clases, hidden=128, dropout=0.3):
        super().__init__()
        self.embed = nn.Linear(n_coords, 32)
        self.A     = nn.Parameter(torch.eye(n_kp))
        self.gcn1  = nn.Linear(32, hidden)
        self.gcn2  = nn.Linear(hidden, hidden)
        self.tcn   = nn.Sequential(nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU())
        self.fc    = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_clases))

    def forward(self, x):             # x: [B, T, KP, C]
        B, T, KP, C = x.shape
        x = F.gelu(self.embed(x))                    # [B, T, KP, 32]
        A = F.softmax(self.A, dim=-1)
        xr = x.reshape(B * T, KP, 32)
        xr = F.gelu(self.gcn1(torch.bmm(A.unsqueeze(0).expand(B*T,-1,-1), xr)))
        xr = F.gelu(self.gcn2(torch.bmm(A.unsqueeze(0).expand(B*T,-1,-1), xr)))
        xr = xr.mean(1).reshape(B, T, -1).permute(0, 2, 1)  # [B, hidden, T]
        xr = self.tcn(xr)
        return self.fc(xr.mean(-1))

# Loop de entrenamiento
def entrenar(modelo, dl_train, dl_val, nombre, n_epochs=N_EPOCHS, lr=LR, patience=PATIENCE):
    optimizer = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    hist = {"loss_tr": [], "loss_val": [], "acc_val": [], "f1_val": []}
    mejor_f1, espera, mejor_ckpt = 0.0, 0, None

    for epoch in range(1, n_epochs + 1):
        modelo.train()
        ls, n = 0.0, 0
        for xb, yb in dl_train:
            optimizer.zero_grad()
            loss = criterion(modelo(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            optimizer.step()
            ls += loss.item() * len(xb); n += len(xb)
        scheduler.step()
        loss_tr = ls / n

        modelo.eval()
        preds, trues, vl = [], [], 0.0
        with torch.no_grad():
            for xb, yb in dl_val:
                logits = modelo(xb)
                vl += criterion(logits, yb).item() * len(xb)
                preds.extend(logits.argmax(1).numpy())
                trues.extend(yb.numpy())
        loss_val = vl / len(dl_val.dataset)
        acc_val  = (np.array(preds) == np.array(trues)).mean()
        f1_val   = f1_score(trues, preds, average="macro", zero_division=0)

        hist["loss_tr"].append(loss_tr)
        hist["loss_val"].append(loss_val)
        hist["acc_val"].append(acc_val)
        hist["f1_val"].append(f1_val)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  [{nombre}] Epoch {epoch:3d}: loss={loss_val:.4f} acc={acc_val:.3f} F1={f1_val:.3f}")

        if f1_val > mejor_f1:
            mejor_f1   = f1_val
            mejor_ckpt = {k: v.clone() for k, v in modelo.state_dict().items()}
            espera     = 0
        else:
            espera += 1
            if espera >= patience:
                print(f"  Early stopping en época {epoch}")
                break

    if mejor_ckpt:
        modelo.load_state_dict(mejor_ckpt)
    return hist, mejor_f1

# Entrenar los dos modelos
print("\n--- Entrenando LSTM Bidireccional ---")
m_lstm  = LSTMBidir(N_KP, N_COORDS, N_CLASES)
hist_lstm, f1_lstm = entrenar(m_lstm, dl_train, dl_val, "LSTM")
torch.save({"state_dict": m_lstm.state_dict(), "label2idx": label2idx,
             "n_clases": N_CLASES, "f1_val": f1_lstm},
           CKPT_DIR / "lstm_best.pt")
print(f"✅ LSTM guardado | F1-val={f1_lstm:.3f}")

print("\n--- Entrenando ST-GCN ---")
m_stgcn = STGCN(N_KP, N_COORDS, N_CLASES)
hist_stgcn, f1_stgcn = entrenar(m_stgcn, dl_train, dl_val, "STGCN")
torch.save({"state_dict": m_stgcn.state_dict(), "label2idx": label2idx,
             "n_clases": N_CLASES, "f1_val": f1_stgcn},
           CKPT_DIR / "stgcn_best.pt")
print(f"✅ ST-GCN guardado | F1-val={f1_stgcn:.3f}")


# ── 5. EVALUACIÓN EN TEST ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("PASO 5 — Evaluación en Test Set")
print("="*60)

def eval_test(modelo, dl, nombre):
    modelo.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in dl:
            preds.extend(modelo(xb).argmax(1).numpy())
            trues.extend(yb.numpy())
    acc = (np.array(preds) == np.array(trues)).mean()
    f1m = f1_score(trues, preds, average="macro",    zero_division=0)
    f1w = f1_score(trues, preds, average="weighted", zero_division=0)
    cm  = confusion_matrix(trues, preds)
    print(f"  {nombre}: Acc={acc:.4f} F1-macro={f1m:.4f} F1-w={f1w:.4f}")
    return {"accuracy": acc, "f1_macro": f1m, "f1_weighted": f1w, "cm": cm,
            "preds": preds, "trues": trues}

res_lstm  = eval_test(m_lstm,  dl_test, "LSTM Bidir")
res_stgcn = eval_test(m_stgcn, dl_test, "ST-GCN")

# Curvas de aprendizaje
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Curvas de Aprendizaje — LSTM vs ST-GCN", fontweight="bold")
for ax, metrica, titulo in zip(axes,
                                ["loss_val", "acc_val", "f1_val"],
                                ["Loss Val", "Accuracy Val", "F1-macro Val"]):
    ax.plot(hist_lstm[metrica],  label="LSTM Bidir",  color="steelblue",  lw=2)
    ax.plot(hist_stgcn[metrica], label="ST-GCN",      color="darkorange", lw=2)
    ax.set_title(titulo); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "curvas_aprendizaje_local.png", dpi=120, bbox_inches="tight")
plt.close()
print("✅ curvas_aprendizaje_local.png guardado")

# Matrices de confusión
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Matrices de Confusión (Test)", fontweight="bold")
for ax, res, titulo in zip(axes, [res_lstm, res_stgcn], ["LSTM Bidir", "ST-GCN"]):
    cm_n = res["cm"].astype(float) / (res["cm"].sum(axis=1, keepdims=True) + 1e-8)
    sns.heatmap(cm_n, ax=ax, cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=7)
    ax.set_title(f"{titulo}\nF1-macro={res['f1_macro']:.3f}")
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix_local.png", dpi=120, bbox_inches="tight")
plt.close()
print("✅ confusion_matrix_local.png guardado")


# ── 6. TABLA COMPARATIVA ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("TABLA FINAL")
print("="*60)

tabla = pd.DataFrame({
    "Modelo":       ["Baseline KNN", "Baseline LogReg", "LSTM Bidir", "ST-GCN"],
    "Accuracy":     [baseline_res["KNN (k=5)"]["accuracy"],
                     baseline_res["Reg. Logística"]["accuracy"],
                     res_lstm["accuracy"], res_stgcn["accuracy"]],
    "F1-macro":     [baseline_res["KNN (k=5)"]["f1_macro"],
                     baseline_res["Reg. Logística"]["f1_macro"],
                     res_lstm["f1_macro"], res_stgcn["f1_macro"]],
    "F1-weighted":  [baseline_res["KNN (k=5)"]["f1_weighted"],
                     baseline_res["Reg. Logística"]["f1_weighted"],
                     res_lstm["f1_weighted"], res_stgcn["f1_weighted"]],
    "Tipo":         ["Baseline", "Baseline", "Deep Learning", "Deep Learning"],
}).round(4)

print(tabla.to_string(index=False))
tabla.to_csv(OUT_DIR / "tabla_comparativa_local.csv", index=False)

mejor_dl  = "LSTM Bidir" if res_lstm["f1_macro"] >= res_stgcn["f1_macro"] else "ST-GCN"
mejor_f1  = max(res_lstm["f1_macro"], res_stgcn["f1_macro"])
supera    = "✅" if mejor_f1 > mejor_f1_base else "❌"
print(f"\n{supera} Mejor DL={mejor_dl} (F1={mejor_f1:.3f}) vs Baseline (F1={mejor_f1_base:.3f})")


# ── 7. EXPORTAR ONNX ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PASO 7 — Exportar ONNX")
print("="*60)

try:
    import onnx, onnxruntime as ort

    modelo_export = m_lstm if res_lstm["f1_macro"] >= res_stgcn["f1_macro"] else m_stgcn
    nombre_onnx   = "lstm_best" if modelo_export is m_lstm else "stgcn_best"
    modelo_export.eval()
    dummy    = torch.randn(1, N_FRAMES, N_KP, N_COORDS)
    onnx_out = CKPT_DIR / f"{nombre_onnx}.onnx"

    torch.onnx.export(
        modelo_export, dummy, str(onnx_out),
        input_names=["keypoints"], output_names=["logits"],
        dynamic_axes={"keypoints": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(str(onnx_out)))
    size_mb = onnx_out.stat().st_size / 1e6
    print(f"✅ ONNX exportado: {onnx_out} ({size_mb:.2f} MB)")

    # Benchmark latencia
    sess = ort.InferenceSession(str(onnx_out), providers=["CPUExecutionProvider"])
    x_bench = dummy.numpy()
    tiempos = []
    for _ in range(50):
        t0 = time.perf_counter()
        sess.run(None, {"keypoints": x_bench})
        tiempos.append((time.perf_counter() - t0) * 1000)
    lat_mean = np.mean(tiempos)
    status   = "✅" if lat_mean < 200 else "⚠️"
    print(f"{status} Latencia ONNX: {lat_mean:.1f} ± {np.std(tiempos):.1f} ms (objetivo <200ms)")
except Exception as e:
    print(f"⚠️  ONNX no disponible: {e}")


# ── RESUMEN ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RESUMEN — PIPELINE LOCAL COMPLETADO")
print("="*60)
archivos = [
    "data/label2idx.json",
    "data/eda_local.png",
    "data/baseline_results_local.csv",
    "data/curvas_aprendizaje_local.png",
    "data/confusion_matrix_local.png",
    "data/tabla_comparativa_local.csv",
    "checkpoints/lstm_best.pt",
    "checkpoints/stgcn_best.pt",
]
for a in archivos:
    existe = "✅" if (ROOT / a).exists() else "❌"
    print(f"  {existe} {a}")
print("\n🏁 Pipeline local completado.")
