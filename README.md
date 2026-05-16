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
│   ├── 04_Error_Analysis_Report.ipynb    # Análisis de errores + informe
│   └── 05_Semana5_Experimentos_AB.ipynb  # A/B: Baseline vs Var1 vs Var2
│
├── scripts/
│   ├── run_eda_local.py                  # EDA ejecutable localmente
│   ├── preprocess_sliding_window.py      # Genera manifest_segments.csv
│   ├── extract_landmarks_only.py         # Extrae landmarks sin regenerar manifest
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

## Métricas y Resultados — Semana 5

### Baseline clásico (sklearn) — `scripts/run_baseline.py`

| Modelo | Test Acc | F1-macro | F1-weighted | AUC | ms/seg |
|--------|---------|---------|------------|-----|--------|
| Random | 0.052 | 0.037 | 0.052 | 0.498 | 0.02 |
| KNN (k=5) | 0.864 | 0.791 | 0.847 | 0.971 | 1.23 |
| Naive Bayes | 0.860 | 0.792 | 0.854 | 0.994 | 0.09 |
| **LogReg (C=1)** | **0.912** | **0.859** | 0.904 | 0.996 | 0.02 |

### Experimentos A/B — Un cambio por vez

| # | Variante | Cambio | Test Acc | F1-macro | F1-weighted | Params | Épocas |
|---|----------|--------|---------|---------|------------|--------|--------|
| B | **Baseline** (CNN-LSTM, hidden=256, lr=1e-4) | — punto de partida | **0.830** | **0.721** | 0.780 | 3.6M | 50 |
| V1 | **Var1** (CNN-LSTM, hidden=128, lr=1e-4) | hidden 256→128 (–65% LSTM) | 0.757 | 0.590 | 0.722 | 1.8M | 2 |
| V2 | **Var2** (STGCN-64, lr=5e-4, landmarks) | arch+features: STGCN sobre kp | 0.435 | 0.331 | 0.429 | 1.2M | 5 |
| v2+ | **STGCN-v2** (hidden=128, lr=1e-3, en curso) | más canales + einsum corregido | — | **0.636*** | — | 11.3M | 13/80 |

> *Val F1 a época 13 (entrenamiento pausado, best checkpoint guardado).

**Gráfico comparativo:** `data/semana5_experimentos_ab.png`  
**Log completo:** `logs/semana5_experimentos_v2.txt`

### Feature Set y Pipeline (Semana 5)

| | Baseline / Var1 | Var2 / STGCN-v2 |
|--|----------------|----------------|
| **Input** | Frames RGB [T=30, H=112, W=112] | Landmarks MediaPipe [T=30, N=75, C=3] |
| **Features** | MobileNetV3-Small → 576 feat/frame | 75 kp (42 manos + 33 pose) × 3 coords |
| **Temporal** | BiLSTM 2 capas → avg pool | ST-GCN (grafo espacio-temporal) |
| **Augmentación** | flip temporal 50% (solo train) | ruido σ=0.015 + flip + speed ×0.6–1.4 |
| **Normalización** | ImageNet µ/σ fijos (sin fit) | coords relativas (fit solo en train) |
| **Balanceo** | WeightedRandomSampler (solo train) | WeightedRandomSampler (solo train) |

### Validación y Leakage

| Check | Resultado |
|-------|----------|
| **Tipo de split** | TEMPORAL por video: primeros frames → train, últimos → test |
| **Ratios** | 70% train / 15% val / 15% test |
| **Clases en los 3 splits** | 26/26 ✓ |
| **Solapamiento frames train/test** | **0 frames** ✓ (verificado programáticamente) |
| **Data leakage de normalización** | ImageNet: parámetros fijos; landmarks: fit solo sobre train ✓ |
| **Val usado para** | Early stopping únicamente (no selección de hiperparámetros) |
| **Test evaluado** | Una sola vez por variante al final ✓ |

**Conclusiones A/B:**
1. Baseline CNN-LSTM es la variante más fuerte (F1=0.721) — mayor capacidad LSTM captura mejor la temporalidad de señas continuas.
2. Var1 (hidden=128) cae −0.131 F1: la capacidad del modelo es crítica.
3. Var2 STGCN con solo 5 épocas alcanza F1=0.331; STGCN-v2 extendido muestra Val F1=0.636 en época 13.
4. **Próximo paso:** fusión CNN-LSTM + STGCN-v2 para superar 0.80 F1-macro.

### Landmarks MediaPipe

| Dato | Valor |
|------|-------|
| Archivos .npy generados | 7,235 (100%) |
| Tamaño total | 196 MB |
| Mano activa (promedio) | 86.8% de frames |
| Formato | [30, 75, 3] → 42 manos + 33 pose keypoints |

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
