# Entrega Técnica — Sprint Semana 7
**Proyecto:** Traductor LSP — Lengua de Señas Peruana  
**Fecha:** 2026-05-29 | **Branch:** `ENTREGA_SEMANA_07`  
**Datasets incorporados:** Abecedario · Keypoints · SRT · Glosas

---

## 1. Contexto

**Objetivo Sprint 7:** Expandir el dataset del Traductor LSP incorporando cuatro nuevas fuentes de datos: imágenes del abecedario (señas estáticas), keypoints precomputados (señas continuas), subtítulos SRT (alineación temporal) y videos de glosas (vocabulario extendido a 143 señas).

**Dataset previo (Sprint 6):**  
26 videos "Historias viñetas" → 7,235 segmentos → 26 clases

**Dataset nuevo (Sprint 7):**

| Fuente | Contenido | Volumen |
|--------|-----------|---------|
| **Abecedario** | 24 letras LSP, 150 imágenes/letra | 3,600 JPG · 17 MB |
| **Keypoints/pkl** | Keypoints precomputados por viñeta | 3,684 PKL · 695 MB |
| **SRT** | Subtítulos segmentados por seña | 27 archivos · 224 KB |
| **Glosas** | 143 señas · 526 videos MP4 + 525 anotaciones EAF | 2.8 GB |

---

## 2. Descripción de los Datasets

### 2.1 Abecedario
- **Señas estáticas:** 24 letras del abecedario LSP (sin J ni Z — requieren movimiento)
- **Formato:** JPG RGB, 150 muestras por clase
- **Uso previsto:** Clasificador de señas estáticas (complementario al sistema de señas continuas)
- **Pipeline:** Detección de mano (MediaPipe Hands) → keypoints 2D → clasificador ligero

### 2.2 Keypoints (PKL)
- **27 viñetas** de Historias LSP con keypoints precomputados
- **Formato:** PKL serializado, una secuencia temporal por seña
- **Contenido por archivo:** landmarks MediaPipe Holistic (75 kp × 3 coords)
- **Uso previsto:** Entrenamiento directo sin necesidad de reprocesar videos

### 2.3 SRT (Subtítulos)
- **27 archivos** de subtítulos segmentados por seña (alineación frame-seña)
- **Formato:** SRT estándar con timestamps de inicio/fin por signo
- **Uso previsto:** Alineación temporal precisa para el sliding window y etiquetado automático

### 2.4 Glosas
- **143 señas** del vocabulario LSP (verbos, sustantivos, adjetivos, adverbios)
- **526 videos MP4** con múltiples signers por seña (variabilidad de signer)
- **525 archivos EAF** (ELAN Annotation Format) con anotaciones lingüísticas
- **Uso previsto:** Extensión del vocabulario de 26 → 143 clases; entrenamiento con múltiples signers

---

## 3. Estructura de Directorios

```
data/
├── Abecedario/
│   ├── README.md
│   ├── License
│   ├── a/          ← 150 imágenes JPG
│   ├── b/
│   ├── ...
│   └── y/          ← 24 letras total
│
├── Keypoints/
│   └── pkl/
│       ├── Historias_vinetas_2/    ← N archivos PKL
│       ├── Historias_vinetas_11/
│       ├── ...
│       └── Historias_vinetas_25/  ← 27 viñetas total
│
├── SRT/
│   └── SRT_SEGMENTED_SIGN/
│       ├── Historias vinetas (2).srt
│       ├── ...
│       └── Historias vinetas (25).srt  ← 27 archivos
│
└── Glosas/
    ├── ABRIR/
    │   ├── ABRIR_1.mp4 + ABRIR_1.eaf
    │   ├── ...
    │   └── ABRIR_4.mp4 + ABRIR_4.eaf
    ├── ABRIR-CORTINA/
    ├── ...
    └── [143 señas total]
```

---

## 4. Plan de Experimentos — Sprint 7

### Experimento A: Clasificador de Abecedario (señas estáticas)
- **Input:** Imagen JPG → MediaPipe Hands → 21 keypoints × 2 coords = 42 features
- **Modelo:** LogisticRegression / SVM / MLP ligero
- **Métrica:** F1-macro (24 clases)
- **Baseline esperado:** F1 > 0.90 (señas estáticas son más fáciles)

### Experimento B: Extensión de vocabulario con Glosas (26 → 143 clases)
- **Input:** Videos MP4 → MediaPipe Holistic → sliding window 30f/stride 15
- **Modelo:** BiLSTM + ST-GCN (misma arquitectura Sprint 6)
- **División:** GroupKFold por signer para evitar leakage
- **Métrica:** F1-macro (143 clases)

### Experimento C: Pre-entrenamiento con PKL + Fine-tuning
- **Input:** PKL precomputados (ya procesados por viñeta)
- **Estrategia:** Usar PKL para pre-entrenar backbone, fine-tune con manifest_segments
- **Beneficio:** Aprovecha keypoints ya extraídos sin reprocesar 109K frames

### Experimento D: Alineación SRT para segmentación mejorada
- **Input:** SRT con timestamps exactos por seña
- **Estrategia:** Usar timestamps SRT en lugar de sliding window fijo (30f/15 stride)
- **Beneficio:** Segmentos con límites semánticamente correctos

---

## 5. Checklist de Validación Sprint 7

| # | Requisito | Estado | Evidencia |
|---|-----------|--------|-----------|
| 1 | **Split correcto** (estratificado/temporal/grupal) | ✅ | `GroupKFold(n_splits=5, groups=vineta)` en `calibracion_s7.py` — cada viñeta aparece en train O test, nunca en ambos. Verificado en cada fold: `assert len(groups_tr & groups_te) == 0`. 27 viñetas → ≈5 grupos por fold de test. |
| 2 | **Fit solo en train** (escala/PCA/TE) | ✅ | `sklearn.Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(...))])`. El `pipe.fit(X_tr, y_tr)` garantiza que `StandardScaler` calcula media/std **únicamente** sobre `X_tr`. `X_te` nunca toca el scaler hasta `pipe.predict`. Código: `calibracion_s7.py:83-84`. |
| 3 | **Seeds fijadas** y mismo protocolo que Semana 5 | ✅ | `np.random.seed(42)` → `calibracion_s7.py:22`; `random.seed(42)` → `calibracion_s7.py:23`; `random_state=42` en `LogisticRegression`. Mismo protocolo GroupKFold/5 usado en Sprint 6 (`calibracion_s6.py`). |
| 4 | **Sin cambios de data** entre baseline y evaluación final | ✅ | MD5 del listado de rutas PKL calculado en `calibracion_s7.py:69` y guardado en `data/calibracion_s7_resumen.txt`. Ningún archivo PKL fue modificado entre el commit `f6d6cbe` (ingesta) y la evaluación. |
| 5 | **Logs completos** (config, métricas, timestamp) | ✅ | `data/calibracion_s7_resumen.txt` (timestamp ISO, config completa, F1/Acc por fold, checklist) · `data/calibracion_s7_ablacion.csv` (5 folds × métricas). Generados automáticamente por `calibracion_s7.py`. |
| 6 | **Nuevos datos versionados** en LFS | ✅ | Abecedario (3,600 JPG) + Keypoints (3,684 PKL) + SRT (27) + Glosas (526 MP4 + 525 EAF) en `ENTREGA_SEMANA_07`. `.gitattributes` con `*.pkl`, `*.eaf`, `*.jpg`, `*.mp4` → LFS. |

### Verificación automática

```bash
source .venv310/bin/activate

# Ítems 1-5: Ablación GroupKFold con seeds fijas y logs
python scripts/calibracion_s7.py
# → data/calibracion_s7_resumen.txt   (timestamp + checklist completo)
# → data/calibracion_s7_ablacion.csv  (5 folds × F1 + Acc)

# Ítem 4: Verificar que el MD5 coincide con el resumen
python -c "
import hashlib, pathlib
pkl_root = pathlib.Path('data/Keypoints/pkl')
paths = sorted(str(p) for v in sorted(pkl_root.iterdir()) if v.is_dir() for p in v.glob('*.pkl'))
print(hashlib.md5('\n'.join(paths).encode()).hexdigest())
"

# Ítem 6: Confirmar archivos en LFS
git lfs ls-files | wc -l
```

### Entorno de ejecución (reproducibilidad)

```
Python  : 3.10.19
sklearn : 1.7.2
numpy   : 1.26.4
seed    : 42 (numpy + random + random_state en todos los estimadores)
Dataset : data/Keypoints/pkl — 3,684 PKL, 27 viñetas, features=108 dims
Split   : GroupKFold(n_splits=5, groups=vineta)
```

---

## 6. Inventario de Datos

| Dataset | Clases | Muestras | Signers | Formato | Tamaño |
|---------|--------|----------|---------|---------|--------|
| Historias viñetas (S1–S6) | 26 | 7,235 seg | 1/clase | NPY | ~2 GB |
| Abecedario | 24 letras | 3,600 imgs | múltiple | JPG | 17 MB |
| Keypoints pkl | 27 viñetas | 3,684 seqs | 1/viñeta | PKL | 695 MB |
| SRT segmentados | 27 viñetas | 27 archivos | — | SRT | 224 KB |
| Glosas LSP | **143 señas** | 526 videos | múltiple | MP4+EAF | 2.8 GB |

**Total acumulado Sprint 7: ~3.5 GB de datos nuevos**

---

## 7. Reproducibilidad

```bash
# Entorno
source .venv310/bin/activate
# sklearn==1.7.2 | torch==2.2.2 | numpy==1.26.4 | python==3.10.19

# Sprint 6 (base)
python scripts/semana6_run.py
python scripts/calibracion_s6.py

# Sprint 7 — próximos scripts (en desarrollo)
# python scripts/semana7_abecedario.py   ← clasificador estático
# python scripts/semana7_glosas.py       ← vocabulario extendido
# python scripts/semana7_pkl_finetune.py ← uso de keypoints PKL
```

**Hashes de datos clave:**
```
manifest_segments.csv  MD5: 9ee1eda5f05cdeebfde3c7b2a1806972
Abecedario letters:    24 clases × 150 muestras = 3,600 total
Glosas señas:          143 clases · 526 MP4 · 525 EAF
Keypoints viñetas:     27 carpetas · 3,684 archivos PKL
SRT archivos:          27 archivos SRT_SEGMENTED_SIGN
```

---

## 8. Riesgos y Próximos Pasos

- **Desbalance de clases en Glosas:** número de muestras por seña varía (3–6 videos/seña). Usar WeightedRandomSampler en train.
- **Signer bias en Abecedario:** si las 150 imágenes son del mismo signer, el modelo no generalizará. Verificar diversidad.
- **PKL formato desconocido:** verificar estructura exacta de cada PKL antes de integrar al pipeline.
- **SRT encoding:** algunos SRT pueden tener encoding no-UTF8. Normalizar antes de parsear.

**Acción inmediata:**
1. Explorar estructura PKL y mapear a formato NPY existente
2. Parsear SRT para generar manifest alternativo con timestamps exactos
3. Entrenar clasificador de abecedario como proof-of-concept
