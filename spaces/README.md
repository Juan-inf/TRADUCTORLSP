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
- **1141 señas LSP** reconocidas
- **LSTM Bidireccional + Attention** sobre secuencias temporales de landmarks
- **MediaPipe Holistic**: pose (33 pts) + ambas manos (21 pts c/u) = 150 dims/frame

## Dataset

| Fuente | Muestras | Clases |
|--------|---------|--------|
| Keypoints/pkl (viñetas segmentadas) | 3,684 | 1,086 |
| Glosas (grabaciones individuales) | 252 | 143 |
| **Total** | **3,936** | **1,141** |

## Uso

1. Abre la tab **📷 Cámara en vivo** y permite acceso a la cámara
2. Realiza una seña LSP frente a la cámara
3. El sistema acumula 30 frames y predice la seña en español

## Para desplegar

Copia al Space de HuggingFace junto con los archivos:
- `lstm_signs.onnx` (modelo)
- `lstm_label2idx.json` (etiquetas)
