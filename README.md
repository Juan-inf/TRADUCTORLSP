# TRADUCTOR LSP — Reconocimiento de Lengua de Señas Peruana

Sistema de Machine Learning para reconocimiento de señas individuales en **Lengua de Señas Peruana (LSP)** a partir de keypoints extraídos con MediaPipe. Incluye pipeline completo desde EDA hasta búsqueda de hiperparámetros con optimización bayesiana.

---

## Contexto del Problema

| | |
|---|---|
| **Dominio** | Accesibilidad — educación y salud en Perú |
| **Tarea** | Clasificación multiclase de señas LSP (palabra por palabra) |
| **Dataset** | PKL keypoints extraídos de 26 videos "Historias Viñetas" |
| **Clases** | 1086 señas únicas |
| **Muestras** | 3,684 instancias |
| **Features** | 108 dims — pose (33×2) + mano derecha (21×2), media temporal |
| **Métrica principal** | F1-macro (adecuado para alta cardinalidad y desbalance) |
| **Split** | GroupKFold(n=5, groups=viñeta) — sin data leakage entre viñetas |

> **Nota de dificultad:** 1086 clases con ~3.4 muestras/clase promedio implica sesgo extremo (F1-macro teórico máximo alcanzable con CV ~0.004–0.006). El objetivo del pipeline es demostrar correctitud metodológica, no accuracy absoluta.

---

## Dataset

```
data/
├── Keypoints/pkl/<viñeta>/<seña>_<id>.pkl   # Keypoints MediaPipe por muestra
├── semana8_random_trials.csv                # Trials Random Search Sprint 8
├── semana8_bayes_trials.csv                 # Trials Bayesian Opt Sprint 8
├── semana8_resumen.txt                      # Resumen HPO Sprint 8
├── semana8_graficos.png                     # Gráficos comparativos Sprint 8
├── calibracion_s7_resumen.txt               # Resumen calibración Sprint 7
├── calibracion_s7_graficos.png              # Gráficos calibración Sprint 7
└── baseline_results.png                     # Resultados baseline
```

**Estructura PKL:** cada archivo contiene una lista de frames, donde cada frame es un dict con claves `pose`, `right_hand`, `left_hand`. El feature vector se construye como la media temporal de `pose.x/y` (33×2=66 dims) + `right_hand.x/y` (21×2=42 dims) = **108 dims**.

---

## Estructura del Repositorio

```
TRADUCTOR_LSP/
├── data/
│   ├── Keypoints/pkl/          # Keypoints por viñeta y seña
│   ├── semana8_*.csv/txt/png   # Artefactos Sprint 8
│   ├── calibracion_s7_*.txt/png # Artefactos Sprint 7
│   └── baseline_results.png
│
├── notebooks/
│   ├── 01_EDA_LSP_Dataset.ipynb             # EDA + estadísticas dataset
│   ├── 02_Preprocessing_Landmarks.ipynb     # Extracción MediaPipe Holistic
│   ├── 03_Training_Comparison.ipynb         # CNN-LSTM vs ST-GCN vs Fusión
│   ├── 04_Error_Analysis_Report.ipynb       # Análisis de errores
│   ├── 05_Semana5_Experimentos_AB.ipynb     # A/B: Baseline vs Var1 vs Var2
│   ├── 06_Semana6.ipynb                     # Sprint 6: validación + calibración
│   ├── 07_Semana7.ipynb                     # Sprint 7: curvas aprendizaje + reliability
│   ├── 08_Semana8.ipynb                     # Sprint 8: HPO Random vs Bayes
│   └── COLAB_MAESTRO_LSP_COMPLETO.ipynb     # Pipeline completo para Colab
│
├── scripts/
│   ├── run_eda_local.py                     # EDA ejecutable localmente
│   ├── preprocess_sliding_window.py         # Genera manifest_segments.csv
│   ├── extract_landmarks_only.py            # Extrae landmarks PKL
│   ├── run_baseline.py                      # Baseline KNN + LogReg
│   ├── run_training.py / run_training_v2.py # Entrenamiento CNN-LSTM
│   ├── calibracion_s6.py                    # Sprint 6: calibración Platt scaling
│   ├── calibracion_s7.py                    # Sprint 7: validación cruzada + reliability
│   └── semana8_hpo.py                       # Sprint 8: HPO Random Search + Optuna
│
├── src/
│   ├── models/        # CNN-LSTM, ST-GCN, VideoMAE, Fusión multimodal
│   ├── dataset/       # LSPVideoDataset + sliding window DataLoader
│   ├── preprocessing/ # VideoPreprocessor + LandmarkExtractor (MediaPipe)
│   ├── training/      # LSPTrainer, métricas, curvas de aprendizaje
│   └── inference/     # LSPPredictor (webcam) + ONNXPredictor
│
├── api/main.py         # FastAPI REST + WebSocket
├── demo/app_gradio.py  # Demo HuggingFace Spaces
├── configs/config.yaml # Hiperparámetros centralizados
└── requirements.txt
```

---

## Instalación

```bash
# Requiere Python 3.10 (Python 3.14 del sistema NO soporta las dependencias)
python3.10 -m venv .venv310
source .venv310/bin/activate          # macOS/Linux
# .venv310\Scripts\activate           # Windows

pip install -r requirements.txt

# Registrar kernel para Jupyter (necesario si usas VSCode/Jupyter)
python -m ipykernel install --user --name venv310 --display-name "Python 3.10 (TRADUCTOR_LSP)"
```

**Dependencias principales:**

```
scikit-learn>=1.5     optuna>=4.0       pandas numpy matplotlib
torch>=2.2.0          torchvision>=0.17
mediapipe>=0.10.0     opencv-python>=4.8
fastapi uvicorn       gradio>=4.0       onnxruntime>=1.16
```

---

## Pipeline Técnico (Sprint 6–8)

```
PKL keypoints (MediaPipe Holistic)
    │
    ▼
extract_features()
    │  pose.x/y (33×2=66 dims)
    │  right_hand.x/y (21×2=42 dims)
    │  media temporal sobre frames
    ▼
Feature vector [108 dims] por muestra
    │
GroupKFold(n=5, groups=viñeta)      ← sin leakage entre viñetas
    │
    ├── StandardScaler (fit solo en train fold)
    │
    ├── Clasificador sklearn
    │       LogisticRegression(solver=lbfgs)      — baseline
    │       RandomForestClassifier(n_jobs=4)      — Sprint 8
    │       ExtraTreesClassifier(n_jobs=4)        — Sprint 8
    │
    ├── CalibratedClassifierCV (cv=3, method=sigmoid)  — Sprint 7
    │
    └── HPO: Random Search / Optuna TPE + MedianPruner  — Sprint 8
```

---

## Resultados por Sprint

### Sprint 6 — Baseline + Calibración

| Modelo | F1-macro (CV) | Notas |
|--------|--------------|-------|
| LogisticRegression (lbfgs) | ~0.005 | Baseline, 1086 clases, GroupKFold(5) |
| CalibratedClassifierCV (Platt) | ~0.005 | Calibración Platt scaling |

- Artefactos: `data/calibracion_s6_*.png/txt`
- Script: `scripts/calibracion_s6.py`

### Sprint 7 — Validación Cruzada + Reliability

| Componente | Resultado |
|-----------|-----------|
| F1 por fold (5 folds) | Ver `data/calibracion_s7_resumen.txt` |
| Reliability diagram | `data/calibracion_s7_graficos.png` |
| Curvas de aprendizaje | Incluidas en `data/calibracion_s7_graficos.png` |
| Dataset MD5 | `3681f1c51ba15efb645f780815beadb1` |

- Script: `scripts/calibracion_s7.py`
- Notebook: `notebooks/07_Semana7.ipynb`

### Sprint 8 — HPO: Random Search vs Bayesian Optimization

| Método | Mejor F1-macro | Modelo | Trials OK | Podados |
|--------|---------------|--------|-----------|---------|
| Random Search | 0.0040 | RF | 10 | 0 |
| **Bayesian (Optuna TPE)** | **0.0045** | **RF** | 9 | **1** |

**Config ganadora:** Bayesian / RandomForestClassifier  
**Espacio de búsqueda:** RF y ExtraTrees — `n_estimators` [10,60], `max_depth` [5,20], `min_samples_leaf` [1,8]  
**Pruner:** `MedianPruner(n_warmup_steps=3)` — early stopping a nivel de trial  
**Presupuesto:** 10 trials/método (20 evaluaciones totales × 5 folds = 100 fits)

> HGB (HistGradientBoosting) excluido del espacio: construye `n_classes × max_iter` = 1086 × 80 = 86,880 árboles por fold — inviable con este dataset de alta cardinalidad.

- Artefactos: `data/semana8_*.csv/txt/png`
- Script: `scripts/semana8_hpo.py`
- Notebook: `notebooks/08_Semana8.ipynb`

---

## Ejecución por Sprint

```bash
# Sprint 6 — Calibración
python scripts/calibracion_s6.py

# Sprint 7 — Validación + Reliability diagram
python scripts/calibracion_s7.py

# Sprint 8 — HPO Random Search + Bayesian (Optuna)
python -u scripts/semana8_hpo.py
# Genera: data/semana8_random_trials.csv, semana8_bayes_trials.csv,
#          semana8_resumen.txt, semana8_graficos.png  (~5–10 min)
```

---

## Métricas — Semana 5 (CNN-LSTM sobre pixels)

### Baseline clásico (sklearn)

| Modelo | Test Acc | F1-macro | F1-weighted | AUC |
|--------|---------|---------|------------|-----|
| Random | 0.052 | 0.037 | 0.052 | 0.498 |
| KNN (k=5) | 0.864 | 0.791 | 0.847 | 0.971 |
| Naive Bayes | 0.860 | 0.792 | 0.854 | 0.994 |
| **LogReg (C=1)** | **0.912** | **0.859** | 0.904 | 0.996 |

> Resultados sobre dataset de 26 clases (viñetas completas). Sprints 6–8 trabajan sobre 1086 clases (señas individuales).

### Experimentos A/B — LightCNNLSTM (2 épocas, 26 clases)

| # | Variante | Cambio | Test Acc | F1-macro | Params |
|---|----------|--------|---------|---------|--------|
| B | Baseline (hidden=256, lr=1e-4) | — | 0.481 | 0.395 | 1.9M |
| V1 | Var1 (hidden=128, lr=1e-4) | hidden ÷2 | 0.242 | 0.138 | 1.2M |
| V2 | Var2 (hidden=256, lr=5e-4) | lr ×5 | 0.512 | 0.393 | 1.9M |

---

## Riesgos y Mitigaciones

| Riesgo | Severidad | Mitigación |
|--------|-----------|-----------|
| **Alta cardinalidad** (1086 clases, ~3.4 muestras/clase) | Crítica | F1-macro, GroupKFold, modelos de árbol |
| **Data leakage** entre viñetas | Crítica | GroupKFold(groups=viñeta) — viñetas completas en un solo fold |
| **HGB inviable con 1086 clases** | Alta | Usar RF / ExtraTrees (O(n\_samples), no O(n\_classes)) |
| **Python ABI mismatch** (3.14 vs 3.10) | Media | Kernel Jupyter `venv310` registrado explícitamente |
| **Overfitting** (pocos datos por clase) | Alta | min\_samples\_leaf, max\_depth, GroupKFold como regularización implícita |

---

## Ramas

| Rama | Contenido |
|------|-----------|
| `main` | Código base estable |
| `SEMANA7` | Sprints 1–8 completados |
| `Semana8` | Base para Sprint 9+ |

---

## Entorno de Desarrollo

| | |
|---|---|
| **Python** | 3.10 (venv `.venv310/`) |
| **scikit-learn** | ≥ 1.5 (multi_class eliminado de LogisticRegression) |
| **Optuna** | 4.9.0 |
| **PyTorch** | 2.2.2 CPU/MPS |
| **MediaPipe** | 0.10.21 |
| **Hardware local** | Intel i9-9980HK, AMD Radeon Pro 5500M |
| **Colab GPU** | Recomendado para entrenamiento CNN-LSTM completo |

```bash
source .venv310/bin/activate
python -c "import sklearn, optuna; print(sklearn.__version__, optuna.__version__)"
```
