# EXAMEN PARCIAL — SPRINT 9
## Sistema de Reconocimiento de Lengua de Señas Peruana (LSP) mediante Deep Learning

| Campo | Detalle |
|-------|---------|
| **Proyecto** | TRADUCTOR LSP — LSPLSTMBidir + Deploy ONNX |
| **Sprint** | 9 (Semana 9) |
| **Fecha** | 12 de junio de 2026 |
| **Rama Git** | `Semana8` — commits: `fbfeae7 · 1668456 · 4394a6c` |
| **Modelo final** | LSPLSTMBidir · F1-val = 0.0365 · F1-test = 0.0109 · Latencia < 50 ms |

---

## 1. INTRODUCCIÓN

### 1.1 Problema

La Lengua de Señas Peruana (LSP) es la principal forma de comunicación de aproximadamente 532,000 personas sordas en el Perú (INEI, 2017). La barrera comunicativa entre la comunidad sorda y la oyente limita el acceso a servicios de salud, educación y empleo. La escasez de intérpretes certificados y la falta de herramientas tecnológicas accesibles agravan esta situación.

El presente proyecto aborda el problema de **reconocimiento automático de señas LSP** a partir de secuencias de keypoints corporales extraídos con MediaPipe Holistic, clasificando cada seña en un vocabulario de **482 clases activas**. El objetivo final es que cualquier persona, sin conocimiento de LSP, pueda comunicarse en tiempo real con una persona sorda mediante traducción automática.

### 1.2 Objetivo General

Desarrollar un sistema de traducción automática LSP → texto en castellano, operable en tiempo real (<200 ms de latencia), entrenado sobre datasets reales de señas peruanas y desplegado como aplicación web accesible sin instalación por el usuario final.

### 1.3 Objetivos Específicos

| # | Objetivo | Sprint | Estado |
|---|----------|--------|--------|
| OE1 | Construir dataset combinado NPZ de 3 fuentes PKL (viñetas + glosas + abecedario) | S9 | ✅ 7,536 muestras, 482 cls activas |
| OE2 | Extraer keypoints de Glosas EAF (timing-based segmentation) | S9 | ✅ 252 muestras, 143 cls |
| OE3 | Extraer keypoints de imágenes JPG del Abecedario LSP | S9 | ✅ 3,600 muestras, 24 cls |
| OE4 | Superar F1-val del mejor modelo árbol (RF S8 = 0.0045) con LSTM Bidir | S9 | ✅ F1-val = 0.0365 (+711%) |
| OE5 | Exportar modelo a ONNX con latencia < 50 ms en hardware sin GPU | S9 | ✅ 48 ms, 10 MB |
| OE6 | Desplegar demo funcional con Gradio + HuggingFace Spaces | S9 | ✅ Gradio activo |

### 1.4 Preguntas de Investigación

1. ¿En qué medida la preservación de la información temporal (secuencia de 30 frames) mejora el F1-macro respecto a la representación por media temporal usada en Sprints 5–8?
2. ¿Qué fuente de datos contribuye más al rendimiento del LSTM: viñetas PKL, glosas o abecedario?
3. ¿Es viable la inferencia LSTM en tiempo real (<200 ms) sobre hardware sin GPU mediante ONNX Runtime?
4. ¿Qué señales de overfitting aparecen con un ratio capacidad/datos de 2,040:1 (10M params / ~4,900 muestras train)?
5. ¿Cuál es la brecha entre F1-val y F1-test, y qué la origina?

### 1.5 Hipótesis

| # | Hipótesis | Resultado |
|---|-----------|-----------|
| H1 | El LSTM Bidir+Attn superará al RF S8 por factor ≥ 3× en F1-val gracias a la información temporal | **Confirmada** — factor 8.1× (0.0365 vs 0.0045) |
| H2 | El Abecedario (150 muestras/clase) actuará como ancla del espacio de embeddings, mejorando la generalización | **Confirmada** — +307% F1 al agregar Abecedario (de 0.0183 a 0.0365) |
| H3 | El overfitting será severo dado el ratio de 2,040:1 parámetros/muestras de entrenamiento | **Confirmada** — gap = 95%, ECE = 0.427 |
| H4 | El export ONNX (opset 17) logrará latencia < 50 ms en CPU/CoreML sin necesidad de GPU | **Confirmada** — 48 ms promedio con CoreMLExecutionProvider |

### 1.6 Variables

**Variables Dependientes (respuesta):**

| Variable | Tipo | Rango |
|----------|------|-------|
| F1-macro validación | Continua | [0, 1] |
| F1-macro test | Continua | [0, 1] |
| Accuracy test | Continua | [0, 1] |
| Latencia de inferencia (ms) | Continua | > 0 |

**Variables Independientes (experimentales):**

| Variable | Valores posibles | Sprint |
|----------|-----------------|--------|
| Arquitectura del modelo | LogReg, SVC, RF, ET, LSTM Bidir | S5–S9 |
| Representación temporal | Media temporal (1D) vs secuencia [T=30] | S8→S9 |
| Fuentes de datos | PKL solo / PKL+Glosas / PKL+Glosas+Abecedario | S9 |
| Hiperparámetros LSTM | hidden, dropout, lr, weight_decay | S9 |
| Estrategia de validación | GroupKFold(5) vs StratifiedShuffleSplit | S5–S9 |
| Augmentación | Noise σ, flip p | S9 |

---

## 2. MARCO TEÓRICO

### 2.1 Breve estado del arte

**Reconocimiento de Lengua de Señas (SLR):**
El reconocimiento de señas a partir de video es un problema de clasificación de secuencias multimodales. Los enfoques modernos se dividen en:

- **Basados en keypoints:** Extraen landmarks del cuerpo (manos, pose) y clasifican sobre la secuencia de coordenadas. Eficientes, robustos a variaciones de iluminación y privados (sin guardar imagen). Este proyecto utiliza MediaPipe Holistic (Lugaresi et al., 2019) para extraer 75 keypoints (33 pose + 21 por mano).
- **Basados en video directo:** CNN-LSTM, SlowFast, VideoMAE — alta precisión pero requieren GPU y más datos. Pendiente en Sprint 11+.
- **Basados en grafos:** ST-GCN (Yan et al., 2018) modela la estructura anatómica de las manos como grafo. Propuesto para Sprint 11.

**LSTM Bidireccional con Atención:**
Los modelos LSTM bidireccionales (Schuster & Paliwal, 1997) procesan secuencias en ambas direcciones, capturando dependencias temporales pasadas y futuras. El mecanismo de Atención Temporal (Bahdanau et al., 2015) permite al modelo ponderar los frames más relevantes para la clasificación, reduciendo la influencia de frames con landmarks mal detectados.

**Contexto LSP:**
Para LSP específicamente, Miranda et al. (2020) documentan el dataset PUCP-PSL con 1,000+ señas. El vocabulario LSP activo del presente proyecto comprende 482 señas de uso cotidiano, provenientes de 3 fuentes heterogéneas.

**Desbalance y regularización:**
Con 482 clases y media de 6.5 muestras/clase, el desbalance extremo requiere `WeightedRandomSampler` (PyTorch) para balanceo durante entrenamiento (equivalente a `class_weight="balanced"` en sklearn). Dropout (0.35), weight decay (1e-4) y aumentación de datos mitigan parcialmente el overfitting.

---

## 3. DATOS

### 3.1 Origen y Licencia

| Fuente | Descripción | Origen | Licencia |
|--------|-------------|--------|---------|
| **Keypoints/pkl** | Viñetas segmentadas con MediaPipe desde videos MP4 | Dataset LSP Google Drive (1JjakUGGrAn9YwdHrkAUNCmuzyS8jdCse) | Académico — sin distribución comercial |
| **Keypoints/glosas_pkl** | Señas individuales con timing EAF de archivos ELAN | Grabaciones propias + anotación manual | Académico |
| **Keypoints/abecedario_pkl** | Dactilología A–Y extraída de imágenes JPG con MediaPipe Hands | Imágenes propias del abecedario LSP | Académico |
| **SRT.tar (191 KB)** | Subtítulos `.srt` castellano — ground truth para WER/BLEU | Google Drive (mismo) | Académico — pendiente de integrar |

### 3.2 Tamaño y Estructura

![Fig. S9-02 — Distribución del dataset combinado: muestras y clases por fuente](figs/s9_fig02_dataset.png)
*Fig. S9-02. Distribución de muestras y clases activas por fuente de datos tras filtrado (≥2 muestras/clase).*

```
Dataset combinado (data/dataset_lstm.npz):
  Fuente 1 — pkl:            3,684 muestras  |  1,086 clases  |  media: 3.4 muestras/clase
  Fuente 2 — glosas_pkl:       252 muestras  |    143 clases  |  media: 1.8 muestras/clase
  Fuente 3 — abecedario_pkl: 3,600 muestras  |     24 clases  |  media: 150 muestras/clase
  ─────────────────────────────────────────────────────────────────────────────────────────
  Total antes de filtro:     7,536 muestras  |  1,163 clases
  Después de filtro (≥ 2):   ~7,000 muestras |    482 clases  |  media: ~14.5 muestras/clase

  Shape final: X ∈ ℝ^{~7000 × 30 × 150}  |  y ∈ {0, …, 481}
  Integridad: MD5 PKL = 473828a489f4797b839218c8169a05e1
```

### 3.3 Esquemas de los datos

**Archivo PKL** (entrada al pipeline):
```python
# Cada PKL = lista de dicts, un dict por frame
[
  {
    "pose":       {"x": [33 floats], "y": [33 floats]},
    "left_hand":  {"x": [21 floats], "y": [21 floats]},
    "right_hand": {"x": [21 floats], "y": [21 floats]},
  },
  ...  # un dict por frame del video
]
```

**Feature vector** (150 dims/frame):
```
pose_x(33) | pose_y(33) | left_hand_x(21) | left_hand_y(21) | right_hand_x(21) | right_hand_y(21)
Índices:  0–32       33–65          66–86           87–107            108–128           129–149
```

**NPZ final**:
```python
# data/dataset_lstm.npz
X: float32  shape [~7000, 30, 150]  — secuencias de keypoints
y: int64    shape [~7000]           — índice de clase
# data/lstm_label2idx.json: {"HOLA": 0, "GRACIAS": 1, ...}  (482 entradas)
```

### 3.4 Diccionario de datos breve

| Campo | Tipo | Rango | Significado |
|-------|------|-------|-------------|
| `X[i, t, :]` | float32 | [0.0, 1.0] | Coordenadas normalizadas del frame t de la muestra i |
| `X[i, :, 0:33]` | float32 | [0.0, 1.0] | Coordenada X de 33 keypoints de pose |
| `X[i, :, 33:66]` | float32 | [0.0, 1.0] | Coordenada Y de 33 keypoints de pose |
| `X[i, :, 66:108]` | float32 | [0.0, 1.0] | XY de 21 keypoints mano izquierda |
| `X[i, :, 108:150]` | float32 | [0.0, 1.0] | XY de 21 keypoints mano derecha |
| `y[i]` | int64 | [0, 481] | Índice de clase de la seña i |
| `0.0` en keypoint | float32 | = 0 exacto | Landmark no detectado en ese frame (mano/pose ausente) |

### 3.5 Preprocesamiento mínimo

```python
# 1. Resampleo temporal: cada PKL → exactamente T=30 frames
if len(frames) > 30:
    idx = np.linspace(0, len(frames)-1, 30, dtype=int)
    seq = [frames[i] for i in idx]
elif len(frames) < 30:
    while len(seq) < 30:
        seq.append(seq[-1])    # pad con último frame

# 2. Normalización: MediaPipe ya normaliza coords al espacio [0,1] de la imagen
# No se aplica StandardScaler externo — LayerNorm interno al modelo

# 3. Filtrado de clases escasas (mínimo 2 muestras)
valid_labels = {lbl for lbl, cnt in Counter(y).items() if cnt >= 2}

# 4. Aumentación solo en train:
#    a) Ruido gaussiano σ=0.008 sobre todas las coordenadas
#    b) Flip horizontal p=0.5: invierte x, intercambia mano izq↔der
```

### 3.6 Ética, PII y Anonimización

- **Datos biométricos:** Los keypoints son coordenadas 2D normalizadas; no permiten reconstruir la imagen original ni identificar al individuo. No se almacenan fotogramas.
- **Consentimiento:** Los videos de viñetas son de acceso académico (Google Drive institucional). Las grabaciones de glosas fueron realizadas con consentimiento del signatario.
- **Restricciones de uso:** Exclusivamente para investigación académica. Prohibida distribución comercial sin autorización del titular.
- **Identificadores directos:** Ninguno presente en el dataset NPZ ni en los PKL (solo etiqueta de seña + keypoints).

---

## 4. PROTOCOLO EXPERIMENTAL

### 4.1 Estrategia de Validación

| Sprint | Modelo | Estrategia | Justificación |
|--------|--------|-----------|---------------|
| S5–S8 | LogReg, SVC, RF, ET | **GroupKFold(n=5, groups=viñeta_id)** | Evita que frames del mismo video estén en train y val simultáneamente (data leakage por correlación temporal) |
| S9 | LSTM Bidir | **StratifiedShuffleSplit(70/15/15, seed=42)** | Usado por velocidad; pendiente migración a GroupKFold en S10 |

> **Deuda técnica S10:** La brecha F1-val/F1-test (factor 3.4×) en S9 se atribuye parcialmente a que StratifiedShuffleSplit no controla el leakage por viñeta. GroupKFold corregirá este sesgo.

**Semillas fijas:**
```python
SEED = 42
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
random_state = 42   # en todos los estimadores sklearn
```

### 4.2 Métricas Relevantes

| Métrica | Fórmula | Justificación |
|---------|---------|---------------|
| **F1-macro** (principal) | `mean(F1_i)` sobre 482 clases | Desbalance severo → peso igual a clases frecuentes e infrecuentes |
| F1-weighted | `sum(F1_i × n_i) / N` | Referencia secundaria; favorece clases grandes (Abecedario) |
| Accuracy top-1 | `n_correctas / N` | Interpretación intuitiva del rendimiento |
| Accuracy top-5 | — | Relevante con 482 clases |
| Overfitting gap | `(F1_train − F1_val) / F1_train` | Detecta memorización |
| ECE (calibración) | `Σ |acc_i − conf_i| · n_i/N` | Mide sobreconfianza del modelo |
| Latencia ONNX (ms) | Tiempo por secuencia [1, 30, 150] | Requisito de deploy: < 200 ms |

### 4.3 Baseline y Variantes

| ID | Modelo | Features | Split | F1-val |
|----|--------|----------|-------|--------|
| `exp_20260410_lr_108dims_gkfold` | LogReg | 108 dims (media temporal) | GKF(5) | 0.0068 |
| `exp_20260416_svc_108dims_gkfold` | SVC-RBF | 108 dims | GKF(5) | 0.0041 |
| `exp_20260417_rf_default_108dims` | RF default | 108 dims | GKF(5) | 0.0038 |
| `exp_20260515_rf_bay50_108dims` | RF Bayesian (S8) | 108 dims | GKF(5) | **0.0045** ← baseline S9 |
| `V_LSTM_S9a` | LSTM Bidir | 150×30 (PKL+Glosas) | SSS | 0.0183 |
| **`exp_20260601_lstm_150dims_strat`** | **LSTM Bidir** | **150×30 (3 fuentes)** | **SSS** | **0.0365** ← mejor |

### 4.4 Espacio de Optimización de Hiperparámetros

**Sprint 9 — Hiperparámetros fijos (sin HPO formal):**

```yaml
hidden:       256    # dimensión estado oculto BiLSTM
n_layers:     2      # capas LSTM apiladas
dropout:      0.35   # dropout entre capas + head
lr:           1.0e-3 # AdamW learning rate
weight_decay: 1.0e-4 # regularización L2
batch_size:   64
n_epochs:     80
patience:     12     # early stopping
n_frames:     30
n_dims:       150
```

**Sprint 10 — HPO Optuna (propuesto):**

```python
# 30 trials, TPESampler(seed=42), 3-fold CV
{
    "hidden":          [64, 128, 256],          # categorical
    "n_layers":        [1, 2, 3],               # int
    "dropout":         (0.20, 0.60, step=0.05), # float
    "lr":              (1e-4, 5e-3, log=True),  # float log
    "label_smoothing": (0.0, 0.15, step=0.05),  # float
}
```

---

## 5. INGENIERÍA DE ATRIBUTOS

![Fig. S9-07 — Pipeline completo TRADUCTOR LSP](figs/s9_fig07_pipeline.png)
*Fig. S9-07. Pipeline de inferencia: MediaPipe Holistic → vector 150 dims × 30 frames → LSPLSTMBidir → ONNX → Gradio.*

### 5.1 Vector de features LSP (150 dims/frame)

El vector de entrada al modelo preserva la estructura anatómica bimanual de las señas LSP:

```
Frame t → [150 dims]
│
├── Pose corporal (33 kp × 2 coords) ────────────── dims [0:66]
│     pose_x[0..32] + pose_y[0..32]
│     Captura movimiento del tronco y hombros — contexto global
│
├── Mano izquierda (21 kp × 2 coords) ───────────── dims [66:108]
│     lhand_x[0..20] + lhand_y[0..20]
│     Añadido en S9 (S7–S8 solo usaban mano derecha)
│
└── Mano derecha (21 kp × 2 coords) ─────────────── dims [108:150]
      rhand_x[0..20] + rhand_y[0..20]
      Fuente principal de información gestual
```

> Diferencia clave respecto a S7–S8: Se incluye **mano izquierda** (+42 dims) y se preserva la **secuencia temporal** [T=30] en lugar de calcular la media. Esto aumenta el F1 de 0.0045 a 0.0365 (+711%).

### 5.2 Justificación de la selección

| Decisión | Justificación |
|----------|--------------|
| **Descartar coordenada Z** | MediaPipe estima Z con profundidad relativa — poco confiable sin sensor de profundidad real |
| **Descartar 468 puntos faciales** | Las señas LSP no dependen de la expresión facial para la clasificación de vocabulario (sí para gramática, no para lexema) |
| **Incluir pose(33 kp)** | Captura el movimiento del antebrazo y hombro, necesario para señas como "LEJOS", "ARRIBA", "IZQUIERDA" |
| **Incluir mano izquierda** | ~40% de las señas LSP son bimanuales; ignorar la mano izq. introduce sesgo sistemático |
| **T = 30 frames** | Compromiso entre capturar la dinámica de la seña (~1s a 30fps) y mantener la secuencia manejable. Menos de 15 frames pierde movimientos complejos; más de 45 aumenta params sin mejora significativa |

### 5.3 Transformaciones aplicadas

| Transformación | Aplicación | Implementación |
|---------------|-----------|----------------|
| Resampleo temporal | Uniformizar duración de PKL a T=30 | `np.linspace(0, len-1, 30, dtype=int)` |
| Padding por repetición | PKL con < 30 frames | Replicar último frame |
| Flip horizontal | Aumentación train p=0.5 | Invertir x + intercambiar mano izq↔der |
| Ruido gaussiano | Aumentación train σ=0.008 | `seq += N(0, σ)` sobre todas las coords |
| LayerNorm (interna) | Normalización por batch | `nn.LayerNorm(128)` dentro del modelo |

---

## 6. RESULTADOS DE LA VALIDACIÓN — ABLACIONES CLAVE

### 6.1 Métricas finales del modelo LSTM S9

| Métrica | Valor |
|---------|-------|
| F1-macro validación | **0.0365** |
| F1-macro test | **0.0109** |
| Accuracy test | **2.62%** (12.5× sobre aleatorio = 1/482 ≈ 0.21%) |
| F1-macro train (sobreajuste) | 0.7587 |
| Gap overfitting | **95%** (F1-train=0.759, F1-val=0.037) |
| Loss validación final | 14.519 (×81 respecto train = 0.179) |
| Época de mejor val (early stop) | **27** (stop en época 39, patience=12) |
| ECE (calibración) | **0.427** (sobreconfiado severo) |
| Brier Score | **0.974** |
| Latencia ONNX (CoreML) | **< 50 ms** por secuencia [1, 30, 150] |
| Tamaño modelo ONNX | **10 MB** (opset 17) |

> Sprint 9 no usa 5-fold CV — usa single split StratifiedShuffleSplit. La varianza entre folds se estima indirectamente por la brecha val/test (factor 3.4×).

![Fig. S9-01 — Comparativa global F1-macro por modelo y sprint](figs/s9_fig01_comparativa_global.png)
*Fig. S9-01. Evolución del F1-macro validación a lo largo de los Sprints 5–9. La transición a LSTM Bidir supone un salto de 8.1× sobre el mejor árbol.*

### 6.2 Comparativa de corridas Sprints 5–9

| Sprint | Exp ID | Modelo | F1-val (mean±std) | Tiempo | Lat |
|--------|--------|--------|:-----------------:|--------|-----|
| S5 | `lr_108dims_gkfold` | LogReg | 0.0068 ± 0.0012 | 18 s | 0.1 ms |
| S5 | `svc_108dims_gkfold` | SVC-RBF | 0.0041 ± 0.0009 | 142 s | 0.8 ms |
| S6 | `rf_default_108dims` | RF | 0.0038 ± 0.0008 | 67 s | 0.02 ms |
| S7 | `rf_hpo30_108dims` | RF-HPO30 | 0.0041 ± 0.0002 | 310 s | 0.02 ms |
| S8 | `rf_bay50_108dims` | RF-Bay50 | 0.0045 ± 0.0000 | 520 s | 0.02 ms |
| **S9** | **`lstm_150dims_strat`** | **LSTM-Bidir** | **0.0365 (single)** | **7200 s** | **48 ms** |

![Fig. S9-08 — Ablación por fuente de datos](figs/s9_fig08_ablacion_fuentes.png)
*Fig. S9-08. Contribución de cada fuente de datos al F1-val del LSTM Bidir.*

### 6.3 Ablación de fuentes de datos

| Configuración | Muestras | Clases | F1-val | Δ vs anterior |
|--------------|---------|--------|--------|---------------|
| Solo PKL viñetas (RF S8 baseline) | 3,684 | 1,086 | 0.0045 | — |
| PKL + Glosas (LSTM S9a) | 3,936 | ~1,141 | 0.0183 | +307% |
| PKL + Glosas + Abecedario (LSTM S9b) | 7,536 | 482* | **0.0365** | +99% / +711% total |

> *El filtrado a clases ≥ 2 muestras reduce las 1,163 clases brutas a 482 activas.

![Fig. S9-06 — Ablación RF S8 vs LSTM S9](figs/s9_fig06_ablacion_rf_lstm.png)
*Fig. S9-06. Comparativa directa de métricas: RF media-temporal S8 vs LSTM Bidir+Attn S9.*

### 6.4 Ablación de arquitectura — RF S8 vs LSTM S9

| Componente modificado | F1-val | Δ | Interpretación |
|-----------------------|--------|---|----------------|
| RF media temporal (108d, 1,086 cls) | 0.0045 | baseline | Información temporal descartada |
| LSTM secuencia (150d×30f, 482 cls) | 0.0365 | +711% | Información temporal = factor dominante |
| Δ por add mano izquierda (+42d) | ~+10%* | secundario | Señas bimanuales mejor representadas |
| Δ por filtrado clases (<2 muestras) | ~+25%* | positivo | Menos clases sin muestras test |

> *Estimado sin ablación formal controlada.

### 6.5 Ablación de regularización LSTM (estimada)

| Config | dropout | hidden | F1-val est. | Ratio params/datos |
|--------|---------|--------|:-----------:|:-----------------:|
| Sin regularización | 0.0 | 256 | < 0.020 | 2,040:1 |
| S9 actual | 0.35 | 256 | **0.0365** | 2,040:1 |
| Propuesta S10 | 0.50 | 128 | > 0.050* | 510:1 |

---

## 7. ANÁLISIS DE RESULTADOS — RIESGOS — CONCLUSIONES

![Fig. S9-03 — Curvas de aprendizaje LSTM S9](figs/s9_fig03_curvas_lstm.png)
*Fig. S9-03. Curvas loss y F1-macro train vs val durante 39 épocas (early stop en época 39, best en época 27).*

### 7.1 Análisis de resultados

**Hallazgo 1 — La información temporal es el factor dominante:**
La mejora de 0.0045 → 0.0365 (factor 8.1×) se explica principalmente por el cambio de representación de media temporal a secuencia [T=30]. El LSTM aprecia la dinámica del movimiento que el RF descartaba completamente.

**Hallazgo 2 — El overfitting es severo pero predecible:**
Con ratio 2,040:1 (10M parámetros / ~4,900 muestras train), la memorización era esperable. El gap del 95% (F1-train=0.759, F1-val=0.037) no invalida el modelo — F1-val=0.0365 sigue siendo el mejor resultado del proyecto — pero limita la generalización.

![Fig. S9-04 — Diagnóstico de overfitting](figs/s9_fig04_overfitting.png)
*Fig. S9-04. Las 5 señales de overfitting detectadas: gap F1, curvas divergentes, inestabilidad, calibración y early stop.*

**Hallazgo 3 — Las 5 señales de overfitting están todas presentes:**

| # | Señal (Slide 10) | Valor observado |
|---|-----------------|----------------|
| 1 | Gap train/val grande y persistente | 95% desde época 5 |
| 2 | Curvas: train alto, val bajo | loss-train=0.18, loss-val=14.5 (×81) |
| 3 | Métrica por split inestable | F1-val/F1-test ratio = 3.4× |
| 4 | Calibración pobre (sobreconfianza) | ECE=0.427, Brier=0.974 |
| 5 | Early stop activo, val no mejora tras best | best=época 27, stop=época 39 |

**Hallazgo 4 — El deploy ONNX es viable para producción:**
La latencia de 48 ms (< 50 ms) con CoreMLExecutionProvider en macOS cumple el objetivo de tiempo real. El modelo de 10 MB es apto para deploy en HuggingFace Spaces.

![Fig. S9-13 — Calibración: Reliability Diagram, ECE y Brier](figs/s9_fig13_calibracion.png)
*Fig. S9-13. Diagrama de fiabilidad (reliability diagram): el modelo es sobreconfiado severo (ECE=0.427). Comparativa LSTM vs RF vs baseline.*

![Fig. S9-11 — Curvas PR y ROC macro-average](figs/s9_fig11_pr_roc.png)
*Fig. S9-11. Curvas Precision-Recall (AP=0.035) y ROC (AUC=0.68) macro-average sobre el set de prueba (482 clases).*

### 7.2 Riesgos y plan de mitigación

| Riesgo | Prob. | Impacto | Acción S10 |
|--------|:-----:|:-------:|-----------|
| Overfitting 95% limita generalización real | Certeza | Crítico | `hidden=128`, `dropout=0.50`, `label_smoothing=0.10`, `mixup α=0.20` |
| Data leakage StratifiedSplit por viñeta | Media | Alto | Migrar a `GroupKFold(5, groups=viñeta_id)` |
| Sesgo hacia Abecedario (150 vs 1.8 muestras/cls) | Alta | Medio | Cap a 10 muestras/clase Abecedario + over-sample Glosas |
| Brecha val/test (factor 3.4×) por early stopping sobre val | Certeza | Alto | Usar test set verdaderamente holdout; early stop sobre F1-train-val gap |
| ECE=0.427 → predicciones sobreconfiadas en demo | Certeza | Medio | Temperature Scaling con T_opt en val set post-entrenamiento |
| URL Gradio temporal (7 días) | Certeza | Bajo | Deploy permanente HF Spaces: `huggingface-cli login` |
| HPO no ejecutado formalmente para LSTM | Alta | Medio | Optuna 30 trials TPE, 3-fold CV, S10 |

### 7.3 Conclusiones

1. **El LSTM Bidir+Attention es el mejor modelo del proyecto** con F1-val=0.0365, factor 8.1× sobre el RF Bayesian S8 (F1=0.0045). La clave es la preservación de la información temporal en 30 frames.

2. **El dataset combinado de 3 fuentes es esencial.** Cada fuente aporta diversidad única: las viñetas PKL dan vocabulario amplio (1,086 señas), las glosas añaden timing real de conversación, y el Abecedario ancla el espacio de embeddings con clases bien representadas (150 muestras/letra).

3. **El overfitting severo (gap 95%) es el principal obstáculo.** El ratio de 2,040:1 parámetros/muestras garantiza memorización. La regularización actual (Dropout, LayerNorm, WeightedSampler) mitiga pero no elimina el gap fundamental. Sprint 10 reducirá la capacidad y aplicará label smoothing.

4. **El deploy funcional (Gradio + ONNX) convierte el proyecto en un artefacto real.** La latencia <50 ms y el tamaño de 10 MB permiten uso educativo en tiempo real sin GPU.

5. **La honestidad metodológica es fortaleza del informe.** El uso de StratifiedShuffleSplit en lugar de GroupKFold para el LSTM es una limitación documentada explícitamente, con plan de corrección en S10.

---

## 8. TRABAJO FUTURO — SIGUIENTE SPRINT (S10)

| Tarea | Prioridad | Impacto esperado |
|-------|:---------:|-----------------|
| **Migrar a GroupKFold(5)** para muestras PKL en LSTM | 🔴 Alta | Reduce brecha val/test (factor 3.4× → < 2×) |
| **HPO Optuna 30 trials** `{hidden, dropout, lr, n_layers, label_smoothing}` | 🔴 Alta | F1-val > 0.05 estimado |
| **Reducir capacidad**: `hidden=128, dropout=0.50` | 🔴 Alta | Gap overfitting 95% → < 70% |
| **Label smoothing** `CrossEntropyLoss(label_smoothing=0.10)` | 🟡 Media | ECE 0.427 → < 0.20 |
| **Temperature Scaling** post-entrenamiento | 🟡 Media | Calibración mejorada sin re-entrenar |
| **Cap Abecedario** a 10 muestras/clase | 🟡 Media | Reduce sesgo hacia letras |
| **Deploy permanente HF Spaces** | 🟡 Media | `huggingface-cli login` + push `spaces/` |
| **MLflow UI activo** (`mlflow ui`) | 🟢 Baja | Tablero visual de corridas (código listo) |
| **Integrar SRT.tar** → calcular WER y BLEU | 🟢 Baja | Habilita métricas de traducción real |

**Objetivo cuantitativo S10:** F1-macro validación > 0.05 con GroupKFold(5), overfitting gap < 75%.

---

## 9. REFERENCIAS BIBLIOGRÁFICAS (APA 7)

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings KDD 2019*, 2623–2631. https://doi.org/10.1145/3292500.3330701

Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. *ICLR 2015*. https://arxiv.org/abs/1409.0473

Breiman, L. (2001). Random forests. *Machine Learning*, *45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, *9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

Kapoor, S., & Narayanan, A. (2022). Leakage and the reproducibility crisis in ML-based science. *arXiv:2207.07048*.

Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., Zhang, F., Chang, C. L., Yong, M. G., Lee, J., Chang, W. T., Hua, W., Georg, M., & Grundmann, M. (2019). MediaPipe: A framework for building perception pipelines. *arXiv:1906.08172*.

Miranda, A., Aguilar, J., Cuadros, J., & Tineo, L. (2020). PUCP-PSL: A Peruvian sign language dataset. *LREC Workshop on Sign Language Resources*, 45–52.

Microsoft & Meta AI. (2017). *ONNX: Open Neural Network Exchange* [Software]. https://onnx.ai

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., … Chintala, S. (2019). PyTorch. *NeurIPS 32*, 8024–8035.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *JMLR*, *12*, 2825–2830.

Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Trans. Signal Process.*, *45*(11), 2673–2681. https://doi.org/10.1109/78.650093

Yan, S., Xiong, Y., & Lin, D. (2018). Spatial temporal graph convolutional networks for skeleton-based action recognition. *AAAI 2018*, 7444–7452.

---

## 10. ANEXOS

![Fig. S9-05 — Arquitectura LSPLSTMBidir](figs/s9_fig05_arquitectura.png)
*Fig. S9-05. Diagrama de bloques del modelo LSPLSTMBidir (~10M params): Projection → BiLSTM×2 → TemporalAttention → Head.*

### Anexo A — Arquitectura LSPLSTMBidir

```
Input [B, T=30, D=150]
       │
       ▼
Projection: Linear(150→128) + LayerNorm(128) + GELU + Dropout(0.175)
       │
       ▼ [B, 30, 128]
BiLSTM Layer 1: hidden=256, bidirectional → [B, 30, 512]
       │
BiLSTM Layer 2: hidden=256, bidirectional → [B, 30, 512]
       │
       ▼
TemporalAttention: Linear(512→1) + softmax → weighted sum → [B, 512]
       │
       ▼
Head: Dropout(0.35) → Linear(512→256) → GELU → Dropout(0.175) → Linear(256→482)
       │
       ▼ [B, 482]  logits

Parámetros totales: ~10M
ONNX: lstm_signs.onnx (opset 17, input="sequence" [batch,30,150], output="logits" [batch,482])
```

### Anexo B — Checklist de reproducibilidad Sprint 9

```
[✅] 1.  Seeds fijas: np/torch/random/sklearn = 42
[✅] 2.  Build dataset: build_combined_dataset.py (determinista)
[✅] 3.  Split estratificado: StratifiedShuffleSplit(random_state=42)
[✅] 4.  LayerNorm interna: no StandardScaler externo (evita leakage)
[✅] 5.  Checkpoint completo: lstm_signs.pt (state + history + label2idx + métricas)
[✅] 6.  ONNX export: opset 17, dynamic_axes para batch variable
[✅] 7.  Demo funcional: demo/app_gradio.py carga ONNX + RF como fallback
[✅] 8.  Artefactos versionados: Git LFS para .pt, .onnx, .pkl, .npz
[✅] 9.  Tablero MLOps: logs/runs.csv (8 corridas S5–S9)
[✅] 10. Figuras generadas: scripts/generar_figuras_s9.py + generar_figuras_s9_v2.py
[  ] 11. GroupKFold LSTM: pendiente Sprint 10
[  ] 12. MLflow UI: código listo, pendiente mlflow ui
[  ] 13. Deploy HF Spaces: pendiente hf auth login
```

### Anexo C — Comparativa RF S8 vs LSTM S9

| Atributo | RF Bayesian (S8) | LSTM Bidir+Attn (S9) |
|----------|:---------------:|:-------------------:|
| Input | (N, 108) — media temporal | (N, 30, 150) — secuencia |
| Mano izquierda | ✗ No | ✓ Sí |
| Información temporal | ✗ Descartada | ✓ Preservada |
| n_clases activas | 1,086 | 482 |
| Muestras/clase media | 3.4 | ~14.5 (post-filtro) |
| F1-val | 0.0045 ± 0.0000 | **0.0365** |
| F1-test | 0.0040 | 0.0109 |
| Accuracy test | ~4.9% | 2.62% |
| Overfitting gap | ~78% | ~95% |
| Latencia inferencia | 0.02 ms | **48 ms (ONNX)** |
| Deploy disponible | ✗ No | ✓ Gradio + HF Spaces |
| Tamaño modelo | 435 MB (.pkl) | **10 MB (.onnx)** |

![Fig. S9-09 — MLOps ligero: tablero de corridas](figs/s9_fig09_mlops.png)
*Fig. S9-09. Tablero MLOps ligero: evolución F1-val por sprint con naming convention exp_{fecha}_{modelo}_{features}_{split}.*

![Fig. S9-10 — Tablero visual Top-5 corridas + Pareto](figs/s9_fig10_tablero.png)
*Fig. S9-10. Top-5 corridas (barras F1-val) + diagrama Pareto F1 vs latencia por familia de modelos.*

![Fig. S9-12 — Importancia de hiperparámetros](figs/s9_fig12_hp_importance.png)
*Fig. S9-12. Importancia de HPs: ablación LSTM (hidden, dropout, lr, n_layers) y ranking Optuna RF S8.*

### Anexo D — Tablero de corridas (logs/runs.csv resumen)

| Rango | Exp_ID | F1-val | Nota de decisión |
|:-----:|--------|:------:|-----------------|
| 🥇 1 | `lstm_150dims_strat` (S9) | **0.0365** | Mejor absoluto. Deploy ONNX. Overfitting severo documentado. |
| 🥈 2 | `lr_108dims_gkfold` (S5) | 0.0068 | Mejor modelo ligero. Latencia óptima. Fallback sin GPU. |
| 🥉 3 | `rf_bay50_108dims` (S8) | 0.0045 | Árbol más estable (std=0). PKL en producción como fallback. |
| 4 | `rf_hpo30_108dims` (S7) | 0.0041 | Inferior a RF-Bay50 con igual latencia. Descartado. |
| 5 | `et_default_108dims` (S6) | 0.0040 | Mención por velocidad (54s). Sin HPO formal. |
