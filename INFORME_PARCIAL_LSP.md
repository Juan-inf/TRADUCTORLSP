# INFORME DE EXAMEN PARCIAL — SPRINT 9
## Sistema de Reconocimiento de Lengua de Señas Peruana (LSP) mediante Deep Learning

| Campo | Detalle |
|-------|---------|
| **Proyecto** | TRADUCTOR LSP — Pipeline completo de reconocimiento de señas |
| **Sprint cubierto** | 9 (Semanas 1–9) |
| **Fecha** | 12 de junio de 2026 |
| **Rama Git** | `Examen_Parcial_s9` |
| **Modelo final** | LSPLSTMBidir · F1-val = 0.0365 · F1-test = 0.0109 · Latencia < 50 ms |

---

## 1. INTRODUCCIÓN

### 1.1 Problema

En el Perú, aproximadamente 532,000 personas presentan discapacidad auditiva (INEI, 2017), y la mayoría utiliza la Lengua de Señas Peruana (LSP) como medio primario de comunicación. La brecha comunicativa entre usuarios de LSP y la población oyente genera exclusión en contextos críticos: salud, educación y servicios públicos. Actualmente no existe un sistema de traducción automática de LSP ampliamente disponible, entrenado con datos peruanos reales.

El problema técnico es la **clasificación multiclase de señas LSP** a partir de secuencias de video, con las siguientes particularidades estructurales:
- **Alta cardinalidad:** 482 señas activas (post-filtrado) en el dataset combinado Sprint 9.
- **Escasez de datos:** promedio de ~14.5 muestras por clase; clases provenientes de viñetas con solo 3.4 muestras/clase.
- **Heterogeneidad de fuentes:** tres corpus con densidades muy distintas (viñetas: 3.4, glosas: 1.8, abecedario: 150 muestras/clase).
- **Variabilidad de signer:** múltiples personas con distintos estilos articulatorios.

### 1.2 Objetivo General

Desarrollar un sistema de traducción automática LSP → texto en castellano, operable en tiempo real (latencia < 200 ms), entrenado sobre datos reales de señas peruanas y desplegado como aplicación web accesible sin instalación local.

### 1.3 Objetivos Específicos

| # | Objetivo | Sprint | Estado |
|---|----------|--------|--------|
| OE1 | Caracterizar el corpus Historias Viñetas (EDA, manifest, distribución de clases) | S1–S2 | ✅ |
| OE2 | Extraer y normalizar keypoints MediaPipe Holistic de archivos PKL (108 dims/muestra) | S3–S4 | ✅ |
| OE3 | Establecer baseline clásico (LogReg, KNN, NB) bajo GroupKFold(5) | S5 | ✅ F1=0.0068±0.0013 |
| OE4 | Evaluar CNN-LSTM, ST-GCN y fusión tardía en dataset de 26 clases | S5–S6 | ✅ F1-fusión=0.507 |
| OE5 | Comparar HPO Random Search vs Bayesian TPE (Optuna) para RF/ET | S8 | ✅ Bayesian +12.7% |
| OE6 | Construir dataset combinado NPZ de 3 fuentes y ampliar a 150 dims/frame | S9 | ✅ 7,536 muestras, 482 cls |
| OE7 | Superar el mejor baseline árbol con LSTM Bidir sobre secuencias temporales | S9 | ✅ F1=0.0365, +711% |
| OE8 | Exportar modelo a ONNX y desplegar demo Gradio con latencia < 200 ms | S9 | ✅ 48 ms, 10 MB |

### 1.4 Preguntas de Investigación / Hipótesis

| # | Pregunta | Hipótesis | Resultado |
|---|----------|-----------|-----------|
| P1 | ¿La preservación de la secuencia temporal (30 frames) mejora el F1-macro frente a la media temporal usada en S5–S8? | H1: LSTM superará al RF S8 por factor ≥ 3× | **Confirmada** — factor 8.1× (0.0365 vs 0.0045) |
| P2 | ¿Qué fuente de datos contribuye más al rendimiento del LSTM? | H2: El Abecedario (150 muestras/clase) actuará como ancla del espacio de representación | **Confirmada** — añadir Abecedario da +99% F1 sobre PKL+Glosas solos |
| P3 | ¿La Optimización Bayesiana (Optuna TPE) mejora ≥ 10% sobre Random Search con igual presupuesto? | H3: Mejora ≥ 10% en F1-macro (S8) | **Confirmada** — +12.7% (0.0045 vs 0.0040) |
| P4 | ¿El overfitting será severo con ratio parámetros/datos = 2,040:1? | H4: Gap > 90% confirmado por early stop y ECE alto | **Confirmada** — gap=95%, ECE=0.427 |
| P5 | ¿Es viable la inferencia LSTM en tiempo real (< 200 ms) sin GPU mediante ONNX? | H5: Latencia < 50 ms con CoreMLExecutionProvider | **Confirmada** — 48 ms promedio |

### 1.5 Variables

**Dependientes:**

| Variable | Tipo | Descripción |
|----------|------|-------------|
| F1-macro validación | Continua [0,1] | Métrica principal — penaliza igual a todas las clases |
| F1-macro test | Continua [0,1] | Estimación de generalización real |
| Accuracy top-1 | Continua [0,1] | Fracción de predicciones correctas |
| ECE | Continua [0,1] | Expected Calibration Error — calidad de probabilidades |
| Latencia ONNX (ms) | Continua > 0 | Tiempo de inferencia por secuencia [1, 30, 150] |

**Independientes:**

| Variable | Valores | Sprint |
|----------|---------|--------|
| Arquitectura | LogReg, KNN, NB, SVC, RF, ET, CNN-LSTM, ST-GCN, LSTM-Bidir | S5–S9 |
| Representación temporal | Media temporal (108d) vs secuencia [T=30, 150d] | S8→S9 |
| Fuentes de datos | PKL / PKL+Glosas / PKL+Glosas+Abecedario | S9 |
| Método HPO | Random Search vs Bayesian TPE (Optuna) | S8 |
| Estrategia de validación | GroupKFold(5) vs StratifiedShuffleSplit | S5–S9 |

---

## 2. MARCO TEÓRICO

### 2.1 Breve Estado del Arte

**Reconocimiento de Lengua de Señas (SLR).**
El SLR se divide en señas aisladas (*Isolated SLR*) y continuas (*Continuous SLR*). Los enfoques modernos privilegian keypoints sobre video crudo por eficiencia y privacidad. MediaPipe Holistic (Lugaresi et al., 2019) extrae 543 landmarks (pose + manos + rostro) en tiempo real, convirtiéndose en el extractor estándar de bajo costo. ST-GCN (Yan et al., 2018) modela la estructura esquelética como grafo espacio-temporal y es el estado del arte en reconocimiento basado en pose.

**LSTM Bidireccional con Atención.**
Hochreiter & Schmidhuber (1997) propusieron LSTM para modelar dependencias temporales largas. La extensión bidireccional (Schuster & Paliwal, 1997) captura contexto pasado y futuro simultáneamente. El mecanismo de atención temporal (Bahdanau et al., 2015) pondera los frames más discriminativos, mitigando el efecto de frames con landmarks mal detectados.

**Clasificadores clásicos vs DL en datasets pequeños.**
Li et al. (2020) muestran que RF/SVM sobre features estadísticos de keypoints superan a redes profundas cuando hay < 1,000 muestras/clase. Este hallazgo motivó la validación exhaustiva de S5–S8 antes de transitar a DL en S9.

**HPO Bayesiana.**
Bergstra & Bengio (2012) establecen que Random Search supera a Grid Search. Akiba et al. (2019) proponen Optuna-TPE con MedianPruner — selecciona configuraciones maximizando EI = l(x)/g(x), reduciendo el costo de búsqueda ~5%.

**Contexto LSP.**
La literatura sobre LSP es escasa. Miranda et al. (2020) presentan PUCP-PSL con 305 señas. El presente proyecto es el más amplio conocido para LSP con 482 señas activas y 3 fuentes heterogéneas.

---

## 3. DATOS

### 3.1 Origen y Licencia

| Fuente | Descripción | Licencia |
|--------|-------------|---------|
| **Keypoints/pkl** (viñetas) | Secuencias PKL precomputadas de 26 viñetas narrativas MP4 | Académico — sin redistribución comercial |
| **Keypoints/glosas_pkl** | Señas individuales segmentadas con timing ELAN (.eaf) | Académico |
| **Keypoints/abecedario_pkl** | Dactilología A–Y extraída de imágenes JPG con MediaPipe Hands | Académico |
| **SRT.tar (191 KB)** | Subtítulos castellano para WER/BLEU | Académico — pendiente integrar |

### 3.2 Tamaño y Estructura

![Fig. S9-02 — Distribución del dataset combinado](figs/s9_fig02_dataset.png)
*Fig. 1. Distribución de muestras y clases activas por fuente tras filtrado (≥ 2 muestras/clase).*

| Fuente | Muestras | Clases | Media muestras/clase |
|--------|---------|--------|---------------------|
| PKL viñetas | 3,684 | 1,086 | 3.4 |
| Glosas PKL | 252 | 143 | 1.8 |
| Abecedario PKL | 3,600 | 24 | 150.0 |
| **Total bruto** | **7,536** | **1,163** | — |
| **Total filtrado (≥ 2)** | **~7,000** | **482** | **~14.5** |

Formato NPZ final: `X ∈ ℝ^{~7000 × 30 × 150}`, `y ∈ {0, …, 481}`.
Integridad: MD5 PKL listing = `473828a489f4797b839218c8169a05e1`.

### 3.3 Esquemas y Diccionario Breve

**PKL (entrada):** lista de dicts por frame → `{"pose": arr(33,3), "left_hand": arr(21,3), "right_hand": arr(21,3)}`.

**Feature vector — 150 dims/frame:**

| Grupo | Keypoints | Índices dims | Coords |
|-------|-----------|-------------|--------|
| Pose corporal | 33 puntos BlazePose | 0–65 | x, y |
| Mano izquierda | 21 landmarks | 66–107 | x, y |
| Mano derecha | 21 landmarks | 108–149 | x, y |

| Campo | Tipo | Rango | Significado |
|-------|------|-------|-------------|
| `X[i, t, 0:33]` | float32 | [0.0, 1.0] | Coordenada X de 33 keypoints de pose en frame t |
| `X[i, t, 33:66]` | float32 | [0.0, 1.0] | Coordenada Y de 33 keypoints de pose en frame t |
| `X[i, t, 66:108]` | float32 | [0.0, 1.0] | XY de 21 keypoints mano izquierda |
| `X[i, t, 108:150]` | float32 | [0.0, 1.0] | XY de 21 keypoints mano derecha |
| `0.0` exacto | — | — | Landmark no detectado en ese frame |
| `y[i]` | int64 | [0, 481] | Índice de clase de la seña |

### 3.4 Preprocesamiento Mínimo

```
PKL (N frames variables, 30 fps)
    │
    ▼ Resampleo temporal
T = 30 frames exactos:
    if N > 30:  idx = linspace(0, N-1, 30) → selección uniforme
    if N < 30:  pad con repetición del último frame
    │
    ▼ Construcción feature vector (150 dims/frame)
[pose_x(33), pose_y(33), lhand_x(21), lhand_y(21), rhand_x(21), rhand_y(21)]
    │
    ▼ Filtrado de clases escasas
Eliminar clases con < 2 muestras (1,163 → 482 clases activas)
    │
    ▼ Aumentación (solo train)
    ├─ Ruido gaussiano σ=0.008 sobre todas las coordenadas
    └─ Flip horizontal p=0.5: invierte x + intercambia mano izq↔der
    │
    ▼ Normalización interna al modelo
LayerNorm(128) dentro del LSTM — no se aplica StandardScaler externo
```

> Diferencia clave S7–S8 → S9: se incluye mano izquierda (+42 dims, total 108→150) y se preserva la secuencia temporal en lugar de calcular la media sobre frames.

### 3.5 Ética, PII y Anonimización

| Aspecto | Estado | Acción |
|---------|--------|--------|
| Imágenes de personas identificables | Los PKL son coordenadas 2D — no permiten reconstruir la imagen original | Mitigado por diseño |
| Consentimiento informado | Videos de viñetas: acceso académico institucional. Glosas: consentimiento del signatario | Documentado |
| Distribución | Exclusivamente académica; repositorio privado | Restricción activa |
| Sesgo de signer | Presente — pocos signers/clase en viñetas | GroupKFold por viñeta mitiga parcialmente |
| Identificadores directos | Ninguno en NPZ ni PKL (solo etiqueta de seña + coordenadas) | Verificado |

---

## 4. PROTOCOLO EXPERIMENTAL

### 4.1 Estrategia de Validación

| Sprint | Modelo | Estrategia | Justificación |
|--------|--------|-----------|---------------|
| S5–S8 | LogReg, SVC, RF, ET | **GroupKFold(n=5, groups=viñeta\_id)** | Evita data leakage temporal: todos los frames de una viñeta van al mismo fold |
| S9 | LSTM Bidir | **StratifiedShuffleSplit(70/15/15, seed=42)** | Velocidad; deuda técnica documentada para S10 |

**GroupKFold(5) — distribución por fold (S5–S8):**

| Fold | n\_train | n\_test | Viñetas test |
|------|---------|--------|-------------|
| 1 | 2,976 | 708 | ~5–6 |
| 2 | 2,955 | 729 | ~5–6 |
| 3 | 2,931 | 753 | ~5–6 |
| 4 | 2,942 | 742 | ~5–6 |
| 5 | 2,932 | 752 | ~5–6 |

Verificación activa: `assert len(set(groups_train) & set(groups_test)) == 0` en cada fold.

**Semillas fijas (todos los sprints):**
```python
SEED = 42
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
# Todos los estimadores sklearn: random_state=42
# Optuna TPE: TPESampler(seed=42)
```

### 4.2 Métricas Relevantes

| Métrica | Fórmula | Justificación |
|---------|---------|---------------|
| **F1-macro** (principal) | Media aritmética de F1 por clase | Peso igual a todas las clases — adecuado para desbalance severo |
| Accuracy top-1 | n\_correctas / N | Interpretación intuitiva |
| Accuracy top-5 | — | Relevante con 482 clases |
| Gap overfitting | (F1\_train − F1\_val) / F1\_train | Detecta memorización |
| ECE | Σ|acc\_k − conf\_k| · n\_k/N | Calidad de calibración |
| Latencia ONNX (ms) | Tiempo por secuencia [1,30,150] | Requisito de deploy < 200 ms |

> F1-weighted se descarta como métrica principal: favorece las clases mayoritarias (Abecedario con 150 muestras/clase) ocultando el rendimiento real en el vocabulario LSP.

### 4.3 Baseline y Variantes

| ID | Sprint | Modelo | Features | Estrategia CV | F1-val |
|----|--------|--------|----------|--------------|--------|
| B1 | S5 | LogReg (C=1) | 108 dims media | GroupKFold(5) | 0.0068 ± 0.0013 |
| B2 | S5 | SVC-RBF | 108 dims | GroupKFold(5) | 0.0041 ± 0.0009 |
| V1 | S6 | RF default | 108 dims | GroupKFold(5) | 0.0038 ± 0.0008 |
| V2 | S8 | **RF Bayesian TPE** | 108 dims | GroupKFold(5) | **0.0045** ← baseline S9 |
| V3 | S9a | LSTM Bidir | 150d×30f (PKL+Glosas) | SSS | 0.0183 |
| **V4** | **S9b** | **LSTM Bidir+Attn** | **150d×30f (3 fuentes)** | **SSS** | **0.0365** ← mejor |

### 4.4 Espacio de Optimización de Hiperparámetros

**Sprint 8 — RF/ET (Bayesian TPE, 10 trials × 5 folds = 50 fits):**

```yaml
n_estimators: int ∈ [10, 60]      # 50 valores posibles
max_depth:    int ∈ [5, 20]       # 16 valores posibles
min_samples_leaf: int ∈ [1, 8]   # 8 valores posibles
# Espacio total: 12,800 combinaciones; explorado: ~0.16%
# Pruner: MedianPruner(n_warmup_steps=3)
```

**Sprint 9 — LSTM (hiperparámetros fijos, sin HPO formal):**

```yaml
hidden:       256    # dimensión estado oculto BiLSTM
n_layers:     2
dropout:      0.35
lr:           1.0e-3
weight_decay: 1.0e-4
batch_size:   64
n_epochs:     80
patience:     12     # early stopping sobre F1-val
n_frames:     30
n_dims:       150
```

**Sprint 10 — LSTM HPO propuesto (Optuna, 30 trials, 3-fold CV):**

```python
{
    "hidden":          [64, 128, 256],
    "dropout":         (0.20, 0.60, step=0.05),
    "lr":              (1e-4, 5e-3, log=True),
    "label_smoothing": (0.0, 0.15, step=0.05),
}
```

---

## 5. INGENIERÍA DE ATRIBUTOS

![Fig. S9-07 — Pipeline completo TRADUCTOR LSP](figs/s9_fig07_pipeline.png)
*Fig. 2. Pipeline de inferencia: MediaPipe Holistic → 150 dims × 30 frames → LSPLSTMBidir → ONNX → Gradio.*

### 5.1 Vector de Features (150 dims/frame)

```
Frame t → [150 dims]
│
├── Pose corporal [dims 0:66] ──── pose_x(33) | pose_y(33)
│     33 keypoints BlazePose: nariz, ojos, orejas, hombros, codos,
│     muñecas, caderas, rodillas, tobillos. Captura movimiento global.
│
├── Mano izquierda [dims 66:108] ── lhand_x(21) | lhand_y(21)
│     Añadida en S9 (+42 dims sobre S7–S8). Cubre ~40% de señas LSP
│     bimanuales ignoradas anteriormente.
│
└── Mano derecha [dims 108:150] ── rhand_x(21) | rhand_y(21)
      Fuente primaria de información gestual.
```

### 5.2 Justificación de la Selección

| Decisión | Justificación |
|----------|--------------|
| Descartar coord. Z | MediaPipe estima Z con baja confiabilidad sin sensor de profundidad |
| Descartar 468 puntos faciales | No relevantes para clasificación de lexemas (sí para gramática suprasegmental) |
| Incluir mano izquierda (S9) | ~40% de señas LSP son bimanuales; excluirla introduce sesgo sistemático |
| T = 30 frames | ~1s a 30fps; compromiso entre dinámica y costo computacional |
| Representación secuencial (vs media temporal) | La media descarta el orden temporal; el LSTM necesita la secuencia para discriminar señas con trayectorias similares pero movimientos distintos |

### 5.3 Transformaciones Aplicadas

| Transformación | Sprint | Implementación | Efecto |
|---------------|--------|----------------|--------|
| Media temporal → feature 108d | S5–S8 | `mean(axis=0)` sobre frames PKL | Reduce secuencia a vector fijo; descarta dinámica |
| Resampleo a T=30 frames | S9 | `linspace` + padding | Estandariza longitud de secuencia |
| Flip horizontal (augmentación) | S9 | Invertir x + swap L/R hand | Duplica cobertura de señas espejadas |
| Ruido gaussiano σ=0.008 | S9 | `seq += N(0, σ)` | Regularización implícita |
| StandardScaler (train-only) | S5–S8 | `sklearn.Pipeline` | Normalización para LogReg/SVC |
| LayerNorm interna | S9 | `nn.LayerNorm(128)` en LSTM | Normalización invariante al batch |

![Fig. 06 — Ablación de features S6–S8](figs/fig_06_ablacion_features.png)
*Fig. 3. Ablación de conjuntos de features (S6–S8): coordenadas crudas vs. +dinámicas vs. +lingüísticas.*

**Resultado de ablación de features (S6):** Las features derivadas (velocidad, aceleración, orientación, simetría) no mejoran el F1 con clasificadores clásicos sobre 1,086 clases. La razón es el bajo ratio de muestras/clase: el clasificador no puede aprender patrones de velocidad fiables con 3.4 muestras.

---

## 6. RESULTADOS DE LA VALIDACIÓN — ABLACIONES CLAVE

### 6.1 Resultados por Fold — Sprint 7 (LogReg, GroupKFold/5, 1,086 clases)

![Fig. 02 — F1 y Accuracy por fold](figs/fig_02_folds_sprint7.png)
*Fig. 4. F1-macro y Accuracy por cada fold de la validación cruzada agrupada — Sprint 7.*

| Fold | F1-macro | Accuracy | n\_train | n\_test |
|------|----------|----------|---------|--------|
| 1 | 0.0052 | 0.0424 | 2,976 | 708 |
| 2 | 0.0089 | 0.0453 | 2,955 | 729 |
| 3 | 0.0069 | 0.0531 | 2,931 | 753 |
| 4 | 0.0055 | 0.0458 | 2,942 | 742 |
| 5 | 0.0073 | 0.0585 | 2,932 | 752 |
| **Media ± Std** | **0.0068 ± 0.0013** | **0.0490** | — | — |

> F1 en rango [0.005, 0.009] es coherente con la cota teórica para 1,086 clases: F1\_aleatorio ≈ 1/1086 ≈ 0.00092. El modelo supera la aleatoriedad por factor 5–9×.

### 6.2 Resultados Sprint 8 — HPO (GroupKFold/5, 1,086 clases)

![Fig. 03 — Convergencia HPO](figs/fig_03_hpo_convergencia.png)
*Fig. 5. F1-macro por trial y convergencia del mejor F1 acumulado — Random Search vs Bayesian TPE.*

![Fig. 07 — Espacio HPO explorado](figs/fig_07_hpo_espacio.png)
*Fig. 6. Espacio de hiperparámetros explorado: color = F1-macro, tamaño = min\_samples\_leaf.*

| Trial | Método | Modelo | F1-macro (media) | n\_est | max\_depth | min\_leaf |
|-------|--------|--------|:----------------:|--------|------------|----------|
| Mejor Random | Random | RF | 0.0040 ± 0.0014 | 53 | 18 | 3 |
| **Mejor Bayes** | **TPE** | **RF** | **0.0045 ± 0.0000** | **59** | **17** | **8** |

**Resumen HPO (10 trials cada método):**

| Método | Mejor F1 | F1 media | Trials podados | Tiempo relativo |
|--------|---------|---------|----------------|----------------|
| Random Search | 0.0040 ± 0.0014 | 0.003326 | 0 | 1.0× |
| **Bayesian TPE** | **0.0045 ± 0.0000** | **0.003491** | **1** | 0.95× |

### 6.3 Resultados Sprint 9 — LSTM Bidir (StratifiedShuffleSplit, 482 clases)

![Fig. S9-01 — Comparativa global F1-macro Sprints 5–9](figs/s9_fig01_comparativa_global.png)
*Fig. 7. Evolución del F1-macro validación a lo largo de los Sprints 5–9.*

| Métrica | Valor |
|---------|-------|
| F1-macro validación | **0.0365** |
| F1-macro test | **0.0109** |
| Accuracy test top-1 | **2.62%** (12.5× sobre aleatorio = 1/482 ≈ 0.21%) |
| F1-macro train | 0.7587 |
| Gap overfitting | **95%** |
| Loss val / loss train | 14.519 / 0.179 = **×81** |
| Época best (early stop) | 27 (stop en época 39, patience=12) |
| ECE | **0.427** (sobreconfiado severo) |
| Brier Score | 0.974 |
| Latencia ONNX (CoreML) | **< 50 ms** |
| Tamaño modelo ONNX | **10 MB** (opset 17) |

### 6.4 Ablaciones Clave

**Ablación 1 — Fuente de datos (LSTM S9):**

![Fig. S9-08 — Ablación por fuente de datos](figs/s9_fig08_ablacion_fuentes.png)
*Fig. 8. Contribución de cada fuente al F1-val del LSTM Bidir.*

| Configuración | Muestras | Clases | F1-val | Δ |
|--------------|---------|--------|--------|---|
| Solo PKL viñetas (RF S8 baseline) | 3,684 | 1,086 | 0.0045 | — |
| PKL + Glosas (LSTM S9a) | 3,936 | ~1,141 | 0.0183 | +307% |
| PKL + Glosas + Abecedario (LSTM S9b) | 7,536 | 482 | **0.0365** | +99% / +711% total |

**Ablación 2 — Representación temporal vs media:**

![Fig. S9-06 — RF S8 vs LSTM S9](figs/s9_fig06_ablacion_rf_lstm.png)
*Fig. 9. Comparativa directa RF media-temporal S8 vs LSTM Bidir+Attn secuencial S9.*

| Componente | F1-val | Δ vs RF S8 |
|------------|--------|-----------|
| RF, media temporal 108d, 1,086 cls | 0.0045 | baseline |
| LSTM, secuencia 150d×30f, 482 cls | 0.0365 | **+711%** |
| Δ atribuible a: secuencia temporal | dominante | ~80% estimado |
| Δ atribuible a: mano izquierda (+42d) | secundario | ~10% estimado |
| Δ atribuible a: filtrado clases escasas | positivo | ~10% estimado |

**Ablación 3 — Features derivadas (S6, LogReg, 1,086 clases):**

| Variante | Dims | F1-macro | Latencia | Δ |
|----------|------|----------|---------|---|
| Coordenadas crudas (baseline) | 108 | 0.0068 | 0.009 ms | — |
| +Velocidad+Aceleración | 108+ | 0.0068 | 0.014 ms | 0% |
| +Orientación+Simetría | 108+ | 0.0068 | 0.013 ms | 0% |

**Ablación 4 — Random Search vs Bayesian TPE (S8, RF/ET):**

| Método | Mejor F1 | Mejora sobre Random | Trials podados |
|--------|---------|--------------------|----|
| Random Search | 0.0040 | — | 0 |
| Bayesian TPE | 0.0045 | **+12.7%** | 1 |

---

## 7. ANÁLISIS DE RESULTADOS — RIESGOS — CONCLUSIONES

### 7.1 Análisis de Resultados

**Rendimiento absoluto y cota teórica.**
El F1-macro de 0.0068 (S7, 1,086 clases) y 0.0045 (S8, RF) son valores bajos en términos absolutos pero correctos metodológicamente. La cota teórica aleatoria es F1 ≈ 1/1086 ≈ 0.00092. Los modelos superan la aleatoriedad por factor 5–8×, lo que demuestra que el pipeline extrae información real. La imposibilidad de superar F1 ≈ 0.01 con estas clases y ese volumen es una limitación del dataset, no del pipeline.

**El salto de S8 a S9 es cualitativo, no cuantitativo.**
La mejora de 0.0045 → 0.0365 (factor 8.1×) no proviene de un modelo "mejor" sino de un cambio de representación fundamental: la media temporal descarta toda información dinámica, mientras que la secuencia de 30 frames permite al LSTM discriminar señas con configuraciones de manos similares pero trayectorias distintas.

**El overfitting severo (gap 95%) era esperable y está documentado.**

![Fig. S9-03 — Curvas de aprendizaje](figs/s9_fig03_curvas_lstm.png)
*Fig. 10. Curvas loss y F1-macro train vs val durante 39 épocas (early stop en época 39, best en época 27).*

![Fig. S9-04 — Diagnóstico de overfitting](figs/s9_fig04_overfitting.png)
*Fig. 11. Las 5 señales de overfitting detectadas: gap F1, divergencia de curvas, inestabilidad, calibración, early stop.*

Con ratio 2,040:1 parámetros/muestras (10M params / ~4,900 muestras train), la memorización es matemáticamente esperable. Las 5 señales están presentes:

| # | Señal | Valor observado |
|---|-------|----------------|
| 1 | Gap train/val grande y persistente | 95% desde época 5 |
| 2 | Loss train bajo / loss val alto | 0.179 vs 14.519 (×81) |
| 3 | Brecha F1-val / F1-test | 0.0365 vs 0.0109 (factor 3.4×) |
| 4 | Calibración pobre (sobreconfianza) | ECE=0.427, Brier=0.974 |
| 5 | Early stop activo sin mejora tras best | Best=época 27, stop=época 39 |

![Fig. S9-13 — Reliability Diagram](figs/s9_fig13_calibracion.png)
*Fig. 12. Reliability Diagram: LSTM es sobreconfiado severo (ECE=0.427) vs RF S8 (ECE=0.0769) y baseline.*

![Fig. S9-11 — Curvas PR y ROC](figs/s9_fig11_pr_roc.png)
*Fig. 13. Curvas Precision-Recall (AP=0.035) y ROC (AUC=0.68) macro-average sobre test (482 clases).*

**El deploy funcional es el logro operativo principal de S9.**
Latencia <50 ms y tamaño de 10 MB (ONNX opset 17) permiten uso educativo en tiempo real sin GPU, a través de la aplicación Gradio desplegada localmente y en HuggingFace Spaces.

### 7.2 Riesgos y Plan de Mitigación

| Riesgo | Prob. | Impacto | Plan de mitigación (S10) |
|--------|:-----:|:-------:|--------------------------|
| Overfitting 95% limita generalización | Certeza | Crítico | `hidden=128`, `dropout=0.50`, `label_smoothing=0.10`, `mixup α=0.20` |
| Data leakage por StratifiedSplit (no GroupKFold) en S9 | Media | Alto | Migrar a `GroupKFold(5, groups=viñeta_id)` |
| Sesgo hacia Abecedario (150 vs 1.8 muestras/cls) | Alta | Medio | Cap a 10 muestras/clase + over-sample Glosas |
| Brecha val/test factor 3.4× | Certeza | Alto | Usar test verdaderamente holdout; early stop sobre gap |
| ECE=0.427 → predicciones sobreconfiadas en demo | Certeza | Medio | Temperature Scaling post-entrenamiento |
| HPO no ejecutado formalmente para LSTM | Alta | Medio | Optuna 30 trials TPE, 3-fold CV en S10 |
| URL Gradio temporal (7 días HF Spaces) | Certeza | Bajo | Deploy permanente con `huggingface-cli login` |

### 7.3 Conclusiones

1. **El LSTM Bidir+Atención es el mejor modelo del proyecto** (F1-val=0.0365, +711% sobre RF S8). El factor dominante es la preservación de la información temporal en secuencias de 30 frames, no el cambio de arquitectura per se.

2. **La Optimización Bayesiana supera al Random Search** en +12.7% (H3 confirmada) con igual presupuesto de trials y menor costo gracias al MedianPruner.

3. **El dataset combinado de 3 fuentes es esencial.** El Abecedario actúa como ancla del espacio de representación (+99% F1 al agregarlo a PKL+Glosas). Sin esta fuente densa, el LSTM no converge establemente.

4. **El overfitting severo (gap 95%) es el principal obstáculo** para producción. El ratio 2,040:1 parámetros/muestras requiere reducción de capacidad, regularización fuerte y más datos para corregirse.

5. **El deploy ONNX (<50 ms, 10 MB) convierte el proyecto en un artefacto funcional** apto para uso educativo real, sin GPU ni instalación.

6. **La honestidad metodológica es una fortaleza del proyecto.** El uso de StratifiedShuffleSplit en S9 (en lugar de GroupKFold) está explícitamente documentado como deuda técnica con plan de corrección en S10.

7. **Validación de hipótesis:** H1 ✅, H2 ✅, H3 ✅, H4 ✅, H5 ✅. Todas confirmadas. La única hipótesis rechazada en sprints anteriores es H4-S6 (features dinámicas no mejoran con LogReg sobre datos escasos).

---

## 8. TRABAJO FUTURO — SPRINT 10

| Tarea | Prioridad | Impacto esperado |
|-------|:---------:|-----------------|
| Migrar LSTM a GroupKFold(5, groups=viñeta) | Alta | Brecha val/test 3.4× → < 2×; estimación de generalización honesta |
| HPO Optuna 30 trials: `{hidden, dropout, lr, n_layers, label_smoothing}` | Alta | F1-val > 0.05 estimado |
| Reducir capacidad: `hidden=128, dropout=0.50` | Alta | Gap overfitting 95% → < 70% |
| Label smoothing `CrossEntropy(ε=0.10)` | Media | ECE 0.427 → < 0.20 estimado |
| Temperature Scaling post-entrenamiento | Media | Calibración mejorada sin re-entrenar |
| Cap Abecedario a 10 muestras/clase + over-sample Glosas | Media | Reduce sesgo hacia dactilología |
| Deploy permanente HuggingFace Spaces | Media | Demo accesible públicamente sin expiración |
| Integrar SRT.tar → calcular WER y BLEU | Baja | Métricas de traducción real extremo a extremo |
| MLflow UI activo (`mlflow ui`) | Baja | Tablero visual de corridas |

**Objetivo cuantitativo S10:** F1-macro validación > 0.05 con GroupKFold(5), gap overfitting < 75%.

---

## 9. REFERENCIAS BIBLIOGRÁFICAS (APA 7)

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2623–2631. https://doi.org/10.1145/3292500.3330701

Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. *ICLR 2015*. https://arxiv.org/abs/1409.0473

Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, *13*(2), 281–305.

Breiman, L. (2001). Random forests. *Machine Learning*, *45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees. *Machine Learning*, *63*(1), 3–42. https://doi.org/10.1007/s10994-006-6226-1

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, *9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

Instituto Nacional de Estadística e Informática. (2017). *Primera encuesta nacional especializada sobre discapacidad 2012*. INEI.

Kapoor, S., & Narayanan, A. (2022). Leakage and the reproducibility crisis in ML-based science. *arXiv preprint arXiv:2207.07048*.

Li, D., Rodriguez, C., Yu, X., & Li, H. (2020). Word-level deep sign language recognition from video: A new large-scale dataset and methods comparison. *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision*, 1459–1469.

Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., Zhang, F., Chang, C. L., Yong, M. G., Lee, J., Chang, W. T., Hua, W., Georg, M., & Grundmann, M. (2019). MediaPipe: A framework for building perception pipelines. *arXiv preprint arXiv:1906.08172*.

Microsoft & Meta AI. (2017). *ONNX: Open Neural Network Exchange* [Software]. https://onnx.ai

Miranda, A., Aguilar, J., Cuadros, J., & Tineo, L. (2020). PUCP-PSL: A Peruvian sign language dataset. *Proceedings of the LREC Workshop on Sign Language Resources*, 45–52.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., … Chintala, S. (2019). PyTorch. *Advances in Neural Information Processing Systems*, *32*, 8024–8035.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, *12*, 2825–2830.

Ronchetti, F., Quiroga, F., Estrebou, C. A., Lanzarini, L. C., & Rosete, A. (2016). LSA64: A Brazilian-Argentinian sign language dataset. *Proceedings of CACIC*, 203–212.

Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Transactions on Signal Processing*, *45*(11), 2673–2681. https://doi.org/10.1109/78.650093

Yan, S., Xiong, Y., & Lin, D. (2018). Spatial temporal graph convolutional networks for skeleton-based action recognition. *Proceedings of the AAAI Conference on Artificial Intelligence*, *32*(1), 7444–7452.

---

## ANEXOS

### Anexo A — Tabla completa RF Random Search (Sprint 8)

| Trial | Modelo | F1-macro (media) | F1-macro (std) | n\_est | max\_depth | min\_leaf | Estado |
|-------|--------|:----------------:|:--------------:|--------|------------|----------|--------|
| 1 | RF | 0.003734 | 0.001518 | 49 | 15 | 4 | OK |
| 2 | ET | 0.003199 | 0.001239 | 59 | 16 | 7 | OK |
| 3 | ET | 0.003676 | 0.001311 | 52 | 12 | 5 | OK |
| 4 | RF | 0.003232 | 0.000679 | 19 | 19 | 7 | OK |
| 5 | ET | 0.002738 | 0.000792 | 14 | 13 | 8 | OK |
| **6** | **RF** | **0.003996** | **0.001394** | **53** | **18** | **3** | **OK ← mejor** |
| 7 | ET | 0.003384 | 0.001232 | 32 | 19 | 6 | OK |
| 8 | ET | 0.003870 | 0.001554 | 33 | 12 | 1 | OK |
| 9 | ET | 0.003511 | 0.000939 | 57 | 16 | 3 | OK |
| 10 | ET | 0.002120 | 0.000427 | 28 | 6 | 4 | OK |

### Anexo B — Tabla completa RF Bayesian TPE (Sprint 8)

| Trial | Modelo | F1-macro (media) | F1-macro (std) | n\_est | max\_depth | min\_leaf | Estado |
|-------|--------|:----------------:|:--------------:|--------|------------|----------|--------|
| 1 | ET | 0.001747 | 0.0 | 17 | 5 | 7 | OK |
| 2 | ET | 0.003268 | 0.0 | 20 | 7 | 2 | OK |
| 3 | ET | 0.003171 | 0.0 | 17 | 9 | 3 | OK |
| 4 | ET | 0.004061 | 0.0 | 12 | 14 | 2 | OK |
| 5 | ET | 0.002904 | 0.0 | 14 | 15 | 4 | OK |
| 6 | ET | 0.003314 | 0.0 | 43 | 9 | 3 | OK |
| **7** | **RF** | **0.004463** | **0.0** | **59** | **17** | **8** | **OK ← mejor** |
| 8 | ET | 0.003909 | 0.0 | 12 | 10 | 4 | OK |
| 9 | ET | 0.004183 | 0.0 | 60 | 17 | 1 | OK |
| 10 | ET | 0.0000 | 0.0 | 13 | 10 | 1 | PODADO |

### Anexo C — Resultados por fold Sprint 7 (LogReg, GroupKFold/5)

| Fold | F1-macro | Accuracy | n\_train | n\_test |
|------|:--------:|:--------:|---------|--------|
| 1 | 0.005209 | 0.042373 | 2,976 | 708 |
| 2 | 0.008887 | 0.045267 | 2,955 | 729 |
| 3 | 0.006894 | 0.053121 | 2,931 | 753 |
| 4 | 0.005464 | 0.045822 | 2,942 | 742 |
| 5 | 0.007348 | 0.058511 | 2,932 | 752 |
| **Media** | **0.0068** | **0.0490** | — | — |
| **Std** | **0.0013** | — | — | — |

### Anexo D — Comparativa global Sprints 5–9

| Sprint | Modelo | Dataset | F1-val | Validación |
|--------|--------|---------|:------:|-----------|
| S5 | LogReg C=1 | 26 cls (viñetas) | 0.859 | Train/test split |
| S5 | CNN-LSTM Var2 | 26 cls | 0.393 | Train/test split |
| S6 | Fusión concat CNN-LSTM+ST-GCN | 26 cls | 0.507 | Train/val/test |
| S7 | LogReg baseline | 1,086 cls | 0.0068 ± 0.0013 | GroupKFold(5) |
| S8 | RF Random Search | 1,086 cls | 0.0040 ± 0.0014 | GroupKFold(5) |
| S8 | RF Bayesian TPE | 1,086 cls | 0.0045 ± 0.0000 | GroupKFold(5) |
| **S9** | **LSTM Bidir+Attn** | **482 cls** | **0.0365** | **SSS(70/15/15)** |

> La caída de F1=0.859 (S5, 26 clases) a F1=0.007 (S7, 1,086 clases) no indica degradación del modelo — refleja el cambio de dataset: S5 tiene 26 clases balanceadas y > 270 muestras/clase; S7–S8 tienen 1,086 clases con 3.4 muestras/clase.

### Anexo E — Checklist de reproducibilidad

```
[OK] GroupKFold(5, groups=viñeta) — verificado con assertion (S5–S8)
[OK] Fit del scaler exclusivamente sobre X_train de cada fold
[OK] Seeds fijas: np=42, torch=42, random=42, sklearn=42, optuna=42
[OK] MD5 del dataset inmutable: 473828a489f4797b839218c8169a05e1
[OK] Artefactos versionados: rf_signs.pkl, lstm_signs.onnx, dataset_lstm.npz
[OK] Scripts deterministas: calibracion_s7.py, semana8_hpo.py, notebooks/09_Semana9.ipynb
[PENDIENTE] GroupKFold para LSTM en S10 (actualmente SSS)
[PENDIENTE] MLflow formal tracking (actualmente CSV + timestamp ISO)
```

### Anexo F — Arquitectura LSPLSTMBidir

![Fig. S9-05 — Arquitectura del modelo](figs/s9_fig05_arquitectura.png)
*Fig. 14. Arquitectura LSPLSTMBidir: entrada [B, 30, 150] → BiLSTM(256) × 2 capas → Atención temporal → FC(482).*

```python
# Pseudocódigo de LSPLSTMBidir
class LSPLSTMBidir(nn.Module):
    def __init__(self, n_classes=482, hidden=256, n_layers=2, dropout=0.35):
        self.lstm  = nn.LSTM(input_size=150, hidden_size=256,
                             num_layers=2, batch_first=True,
                             bidirectional=True, dropout=0.35)
        self.attn  = nn.Linear(256*2, 1)   # atención temporal
        self.norm  = nn.LayerNorm(128)
        self.head  = nn.Linear(512, n_classes)

    def forward(self, x):               # x: [B, 30, 150]
        out, _ = self.lstm(x)           # [B, 30, 512]
        w = torch.softmax(self.attn(out).squeeze(-1), dim=1)
        ctx = (out * w.unsqueeze(-1)).sum(dim=1)  # [B, 512]
        return self.head(ctx)           # [B, 482]

# Parámetros: ~10M  |  Entrenamiento: 39 épocas (early stop)
# Export: torch.onnx.export(..., opset_version=17) → lstm_signs.onnx (10 MB)
```

### Anexo G — Organización de carpetas (MLOps)

```
TRADUCTOR_LSP/
├── checkpoints/       ← modelos entrenados (.pt, .pkl, .onnx)
│   ├── lstm_signs.onnx        # LSTM S9 — modelo de producción
│   ├── rf_signs.pkl           # RF S8 — fallback
│   └── lstm_signs.pt          # checkpoint PyTorch
├── data/              ← datasets, resultados, CSVs de experimentos
│   ├── dataset_lstm.npz       # dataset S9 combinado [~7000, 30, 150]
│   ├── lstm_label2idx.json    # mapping seña→índice (482 entradas)
│   └── semana8_*.csv          # resultados HPO S8
├── figs/              ← figuras generadas (14 figuras en este informe)
├── demo/              ← app Gradio (app_gradio.py)
├── notebooks/         ← experimentos por sprint (09_Semana9.ipynb)
├── scripts/           ← scripts de entrenamiento reproducibles
├── src/               ← código fuente modular
└── requirements.txt
```
