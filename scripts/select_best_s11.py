"""
select_best_s11.py — Compara Transformer S11 vs BiLSTM S11 y copia el ganador
                     a checkpoints/best_s11.pt + best_s11.onnx

Ejecutar una vez que ambos checkpoints estén disponibles.
"""
import shutil, pathlib
import torch

ROOT     = pathlib.Path(__file__).parent.parent
CKPT_DIR = ROOT / "checkpoints"

def load(name):
    path = CKPT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{name} no encontrado en {CKPT_DIR}")
    return torch.load(path, map_location="cpu", weights_only=False), path

ckpt_t, path_t = load("transformer_s11.pt")
ckpt_l, path_l = load("bilstm_s11.pt")

f1_t = ckpt_t["f1_test"]
f1_l = ckpt_l["f1_test"]

print("=" * 55)
print("COMPARATIVA S11 — Transformer vs BiLSTM")
print("=" * 55)
print(f"{'Métrica':<28} {'Transformer':>12} {'BiLSTM S11':>12}")
print("-" * 55)
for key, label in [
    ("f1_test",          "F1-test macro"),
    ("f1_val_mean",      "F1-val KFold media"),
    ("top3_test",        "Top-3 Accuracy"),
    ("top5_test",        "Top-5 Accuracy"),
    ("ece_after",        "ECE (calibrado)"),
    ("temperature",      "T* calibración"),
    ("latencia_onnx_ms", "Latencia ONNX (ms)"),
]:
    vt = ckpt_t.get(key, "N/A")
    vl = ckpt_l.get(key, "N/A")
    fmt = lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)
    print(f"{label:<28} {fmt(vt):>12} {fmt(vl):>12}")

print()
if f1_t >= f1_l:
    winner_name = "Transformer S11"
    winner_src  = path_t
    winner_onnx = CKPT_DIR / "transformer_s11.onnx"
    mejora = (f1_t / 0.0302 - 1) * 100
    best_f1 = f1_t
else:
    winner_name = "BiLSTM S11"
    winner_src  = path_l
    winner_onnx = CKPT_DIR / "bilstm_s11.onnx"
    mejora = (f1_l / 0.0302 - 1) * 100
    best_f1 = f1_l

print(f"★ GANADOR: {winner_name}  (F1-test={best_f1:.4f}, vs S10=0.0302 {mejora:+.1f}%)")
print()

# Copiar ganador a best_s11.pt / best_s11.onnx
dst_pt   = CKPT_DIR / "best_s11.pt"
dst_onnx = CKPT_DIR / "best_s11.onnx"

shutil.copy2(winner_src, dst_pt)
print(f"✅ {dst_pt.name}  ← {winner_src.name}")

if winner_onnx.exists():
    shutil.copy2(winner_onnx, dst_onnx)
    print(f"✅ {dst_onnx.name}  ← {winner_onnx.name}")
else:
    print(f"⚠️  {winner_onnx.name} no encontrado — best_s11.onnx no copiado")

print()
print("Checklist final S11:")
for name, path in [
    ("dataset_s11.npz",       ROOT / "data" / "dataset_s11.npz"),
    ("transformer_s11.pt",    CKPT_DIR / "transformer_s11.pt"),
    ("transformer_s11.onnx",  CKPT_DIR / "transformer_s11.onnx"),
    ("bilstm_s11.pt",         CKPT_DIR / "bilstm_s11.pt"),
    ("bilstm_s11.onnx",       CKPT_DIR / "bilstm_s11.onnx"),
    ("best_s11.pt",           dst_pt),
    ("best_s11.onnx",         dst_onnx),
]:
    mark = "✅" if path.exists() else "❌"
    print(f"  {mark}  {name}")
