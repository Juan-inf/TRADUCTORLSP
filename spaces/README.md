---
title: Traductor LSP
emoji: 🤟
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.0"
app_file: app.py
pinned: false
license: mit
---

# Traductor LSP → Castellano

Sistema integral de comunicación inclusiva para la traducción de **Lengua de Señas Peruana (LSP)** a texto en castellano.

## Características

- **Tiempo real** desde cámara web (webcam, laptop, tablet, smartphone)
- **Videos pregrabados** MP4, AVI, MOV
- **96 señas LSP** reconocidas (BiLSTM Sprint 27)
- **F1-macro = 0.4349** · Top-5 = 63.6% · latencia ONNX = 0.72ms
- **LSTM Bidireccional + Attention** sobre secuencias temporales de landmarks
- **MediaPipe Holistic**: pose (33 pts) + ambas manos (21 pts c/u) = 150 dims/frame

## Dataset (S17 — fix de grupos cross-source)

| Fuente | Muestras |
|--------|---------|
| vineta (Historias viñetas) | 3,684 |
| dgi156 (múltiples señantes) | 3,642 |
| abecedario | 3,600 |
| AEC (intérprete TV) | 1,102 |
| vocabulario_lsp_p | 80 |
| glosa | 42 |
| **Total (≥15 muestras/clase)** | **12,150** en **96 clases** |

## Uso

1. Abre la tab **📷 Cámara en vivo** y permite acceso a la cámara
2. Realiza una seña LSP frente a la cámara
3. El sistema acumula 30 frames y predice la seña en español

## Para desplegar

Copia al Space de HuggingFace junto con los archivos:
- `lstm_signs.onnx` (modelo)
- `lstm_label2idx.json` (etiquetas)
