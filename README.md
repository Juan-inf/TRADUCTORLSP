# TRADUCTOR LSP — Reconocimiento de Lengua de Señas Peruana

Sistema de Deep Learning para reconocimiento de historias narradas en **Lengua de Señas Peruana (LSP)** a partir de videos MP4. Incluye pipeline completo desde EDA hasta inferencia en tiempo real.

---

## Contexto del Problema

| | |
|---|---|
| **Dominio** | Accesibilidad — educación y salud en Perú |
| **Tarea** | Clasificación multiclase de videos continuos de LSP |
| **Dataset** | 26 videos "Historias Vinetas" (historias narradas en señas) |
| **Clases** | 26 (una por historia/vineta) |
| **Métrica principal** | F1-macro (adecuado para desbalance 11:1) |

---

## Dataset Real

```
data/videos/original/Historias vinetas (N).mp4   # N = 2,3,4,...,43
```

| Métrica | Valor |
|---|---|
| Videos totales | 26 (1 inaccesible) |
| Duración total | 60.7 min |
| Duración media | 140s ± 104s (rango: 52–570s) |
| Frames totales | 109,068 |
| FPS | 30 (todos) |
| Resoluciones | 640×480 (62%), 1920×1080 (27%), 1280×720 (8%) |
| Tamaño total | ~744 MB |
| Desbalance máximo | 11:1 (1137 vs 102 segmentos) |

**Tipo de dataset:** Videos continuos (signing stories), no señas aisladas.
Cada video contiene múltiples señas encadenadas narrando una historia completa.

---

## Estructura del Repositorio

```
TRADUCTOR_LSP/
├── data/
│   ├── manifest.csv              # Metadatos de 26 videos
│   ├── manifest_segments.csv     # 7,235 segmentos sliding window
│   ├── label2idx.json            # Mapa clase → índice
│   ├── eda_visualizaciones.png   # Gráficos EDA
│   ├── eda_frames_muestra.png    # Frames de muestra
│   └── videos/original/          # Videos MP4 descargados
│
├── notebooks/
│   ├── 01_EDA_LSP_Dataset.ipynb          # EDA + riesgos + estadísticas
│   ├── 02_Preprocessing_Landmarks.ipynb  # MediaPipe Holistic
│   ├── 03_Training_Comparison.ipynb      # CNN-LSTM vs ST-GCN vs Fusión
│   └── 04_Error_Analysis_Report.ipynb    # Análisis de errores + informe
│
├── scripts/
│   ├── run_eda_local.py                  # EDA ejecutable localmente
│   ├── preprocess_sliding_window.py      # Genera manifest_segments.csv
│   ├── run_baseline.py                   # Baseline KNN + LogReg (sklearn)
│   ├── run_training.py                   # Entrenamiento CNN-LSTM (PyTorch)
│   └── run_pipeline.py                   # Pipeline end-to-end
│
├── src/
│   ├── models/        # CNN-LSTM, ST-GCN, VideoMAE, Fusión multimodal
│   ├── dataset/       # LSPVideoDataset + sliding window DataLoader
│   ├── preprocessing/ # VideoPreprocessor + LandmarkExtractor (MediaPipe)
│   ├── training/      # LSPTrainer, métricas, curvas de aprendizaje
│   └── inference/     # LSPPredictor (webcam) + ONNXPredictor
│
├── api/main.py         # FastAPI REST + WebSocket (inferencia en tiempo real)
├── demo/app_gradio.py  # Demo HuggingFace Spaces
├── configs/config.yaml # Hiperparámetros centralizados
└── requirements.txt
```

---

## Instalación

```bash
# Requiere Python 3.10 (Python 3.14 del sistema NO soporta PyTorch)
python3.10 -m venv .venv310
source .venv310/bin/activate          # macOS/Linux
# .venv310\Scripts\activate           # Windows

pip install -r requirements.txt
```

**Dependencias principales:**

```
torch>=2.2.0          torchvision>=0.17
mediapipe>=0.10.0     opencv-python>=4.8
scikit-learn>=1.3     pandas numpy matplotlib seaborn
fastapi uvicorn       gradio>=4.0
onnxruntime>=1.16
```

---

## Ejecución Rápida

```bash
# 1. EDA (sin GPU, ~30s)
python scripts/run_eda_local.py

# 2. Generar segmentos sliding window (~5s, solo metadatos)
python scripts/preprocess_sliding_window.py --no_landmarks

# 3. Baseline clásico (KNN + LogReg, ~10 min CPU)
python scripts/run_baseline.py

# 4. Entrenamiento CNN-LSTM (CPU: ~20 min/época | GPU: ~3 min/época)
python scripts/run_training.py --epochs 10 --batch_size 4

# 5. Pipeline completo
python scripts/run_pipeline.py --stage all
```

---

## Modelos Implementados

| Modelo | Entrada | Backbone | Parámetros | Modo |
|--------|---------|----------|-----------|------|
| **CNN-LSTM** (baseline DL) | Pixels [B,C,T,H,W] | MobileNetV3-Small + BiLSTM | ~975K | `pixels` |
| **CNN-LSTM Full** | Pixels [B,C,T,H,W] | ResNet50 + BiLSTM | ~27M | `pixels` |
| **ST-GCN** | Landmarks [B,T,N,3] | Grafo espacio-temporal | ~1.5M | `landmarks` |
| **VideoMAE** | Pixels [B,C,T,H,W] | MCG-NJU/videomae-base | ~86M | `pixels` |
| **Fusión** | Ambos | CNN-LSTM + ST-GCN | ~28M | `both` |

---

## Pipeline Técnico

```
Video MP4
    │
    ▼
VideoPreprocessor           ─── resize 224×224, 30fps, 30 frames
    │                            normalización ImageNet
    ├─── Pixels [B,C,T,H,W] ──► CNN-LSTM / VideoMAE
    │
LandmarkExtractor           ─── MediaPipe Holistic
    │                            42 keypoints manos + 33 pose
    └─── Landmarks [B,T,75,3] ► ST-GCN
              │
              ▼
         Fusión multimodal (concat / attention / weighted_sum)
              │
              ▼
         Clasificador (26 clases)
              │
              ▼
         Inferencia < 200ms
```

**Sliding Window** (dataset continuo):
- Ventana: 30 frames (~1s a 30fps)
- Stride: 15 frames (50% overlap)
- Total segmentos: **7,235** (train 4,499 / val 855 / test 1,881)
- Splits a nivel de **video** (sin data leakage entre splits)

---

## Métricas y Resultados

| Modelo | Val Acc | Test Acc | F1-macro | Épocas |
|--------|---------|---------|---------|--------|
| Random baseline | — | 0.038 | 0.038 | — |
| KNN (k=5, histograma) | — | 0.864 | 0.791 | — |
| Naive Bayes | — | 0.860 | 0.792 | — |
| LogReg (C=1) | — | **0.912** | **0.859** | — |
| CNN-LSTM Light (MobileNetV3) | 0.865 | — | 0.794 | 2 |
| CNN-LSTM Full | — | — | — | pendiente |

> **Splits temporales dentro de cada video** (70/15/15 por video): cada clase aparece en los tres particiones. El baseline LogReg supera al random en +2134%. CNN-LSTM en 2 épocas ya alcanza F1=0.79 y sigue mejorando.

---

## Riesgos Identificados

| Riesgo | Severidad | Mitigación |
|--------|-----------|-----------|
| **Desbalance 11:1** (vineta_003 vs vineta_027) | Alta | WeightedRandomSampler + F1-macro |
| **Data leakage** (frames consecutivos entre splits) | Crítica | Splits a nivel de VIDEO, no de segmento |
| **Drift temporal** (signers distintos, resoluciones variadas) | Media | Normalización + augmentación |
| **Overfitting** (26 clases, 1 video/clase) | Alta | Dropout 0.3/0.4, early stopping, backbone congelado |
| **Vineta 3 outlier** (570s vs media 140s) | Media | WeightedRandomSampler balancea segmentos |

---

## Inferencia en Tiempo Real

```bash
# API FastAPI
uvicorn api.main:app --reload --port 8000
# POST http://localhost:8000/predict/video
# WS  ws://localhost:8000/predict/stream

# Demo Gradio
python demo/app_gradio.py  # abre en http://localhost:7860
```

**Latencia objetivo:** < 200ms por seña (modelo ONNX en CPU)

---

## Entorno de Desarrollo

| | |
|---|---|
| **Python** | 3.10 (venv `.venv310/`) |
| **PyTorch** | 2.2.2 CPU/MPS |
| **MediaPipe** | 0.10.21 |
| **Hardware local** | Intel i9-9980HK, AMD Radeon Pro 5500M |
| **Colab GPU** | Recomendado para entrenamiento completo |

```bash
# Activar entorno
source .venv310/bin/activate

# Verificar instalación
python -c "import torch; print(torch.__version__)"
```
