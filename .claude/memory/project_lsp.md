---
name: project-lsp-state
description: Estado técnico del proyecto TRADUCTOR LSP — Sprint 11 — dataset, arquitecturas, métricas, roadmap
metadata:
  type: project
---

## Estado actual: Sprint 11

**Branch activa:** SEMANA10 (S11 pendiente de nuevo branch)

### Dataset S11 (dataset_s11.npz) — NUEVO
- **6,855 muestras** (vs 4,176 S10 = +64%), 479 clases activas
- Media 14.3 muestras/clase (vs 3.6 S10 = **4× más datos**)
- 31 clases con ≥50 muestras, 26 con ≥100
- Fuentes: vineta=3,028 / glosa=227 / **abecedario=3,600** (sin cap, S10 usaba solo 240)
- Archivos: data/dataset_s11.npz + s11_label2idx.json + s11_idx2label.json + s11_groups.json

**Why:** CAP_ABC=10 en S10 limitaba artificialmente el abecedario. Al removerlo: +3,360 muestras de alta calidad (150/clase × 24 letras).

### Scripts S11 (nuevos)
- `scripts/build_dataset_s11.py` — builder S11 (sin cap, normalización etiquetas)
- `scripts/train_s11.py` — entrenamiento Transformer + BiLSTM comparativo
- `notebooks/semana11.ipynb` — notebook orquestador

### Arquitecturas S11
**LSPTransformerS11** (nueva):
- Input [B,30,150] → Linear(150→d_model) + LayerNorm
- CLS token + Positional Embedding aprendibles (T+1 posiciones)
- TransformerEncoder(Pre-LN, nhead=4/8, dim_ff=256/512, n_layers=2-5)
- CLS → Dropout → Linear(d_model → d_model//2) → GELU → Linear → 479 clases
- Justificación: self-attention capta dependencias inter-frame en paralelo; SPOTER 2021 y Sign-ILP 2023 muestran Transformer > LSTM en SLR

**LSPLSTMBidirS11** (mejorada S10):
- Misma arquitectura BiLSTM + Atención temporal de S10
- Entrenada sobre dataset S11 (4× más datos)
- HPO warm start desde S10 best HPs

### Mejores modelos guardados
- `checkpoints/lstm_s10.pt` — **MEJOR PREVIO**: F1-test=0.0302, ECE=0.033, lat=0.9ms, T*=3.04
- `checkpoints/transformer_s11.pt` — Transformer S11 (en entrenamiento)
- `checkpoints/bilstm_s11.pt` — BiLSTM S11 (en entrenamiento)
- `checkpoints/best_s11.pt` — alias al ganador S11

### Métricas históricas
| Sprint | Modelo | F1-test | Latencia |
|--------|--------|---------|---------|
| S5 | LogReg | 0.0058 | 0.1ms |
| S7 | RF-HPO | 0.0036 | 0.02ms |
| S9 | LSTM-Bidir | 0.0109 | 48ms |
| S10★ | LSTM-Bidir-S10 | 0.0302 | 0.9ms |
| S10v2 | LSTM+HE3 | 0.0091 | 0.9ms |
| S11 | Transformer/BiLSTM | EN TRAINING | — |

**How to apply:** Siempre arrancar desde lstm_s10.pt como baseline comparativo. S11 debe superar 0.0302 para considerarse mejora.

### Pipeline de inferencia
1. OpenCV/WebRTC → frame 640×480
2. MediaPipe Holistic → 75 landmarks (33 pose + 21 lhand + 21 rhand)
3. Vector 150 dims/frame → buffer circular 30 frames
4. ONNX Runtime → best_s11.onnx → logits
5. softmax(logits / T*) → probabilidad calibrada
6. label → texto castellano

### Roadmap pendiente
- S12: Frontend React + WebRTC (Capacidad 1), FastAPI WebSocket, BERT español
- S12: Más datos: 50+ muestras por señas dinámicas (actualmente 3.6/clase)
- S12: SRT.tar integration → WER/BLEU
- S13: Segmentación temporal automática, transfer learning WLASL
