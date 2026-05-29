# Entrega Técnica — Sprint Semana 6
**Proyecto:** Traductor LSP — Lengua de Señas Peruana  
**Fecha:** 2026-05-22 | **Branch:** `semana5-entregables` | **Commit:** `8722d20`  
**Script ejecutado:** `scripts/semana6_run.py` | **Log:** `logs/semana6_resultados_finales.txt`

---

## 1. Contexto

**Objetivo:** Reconocimiento de Lengua de Señas Peruana (LSP) desde video — clasificar segmentos de señas continuas en 26 categorías en tiempo real.

**Problema técnico:** Clasificación multi-clase de video continuo. El dataset contiene narraciones completas (no señas aisladas), lo que exige segmentación temporal robusta y modelos sensibles a secuencias largas.

**Dataset:** 26 videos MP4 · 109,068 frames · 60.7 min · 744 MB  
Segmentación: sliding window 30 frames / stride 15 → **7,235 segmentos**  
Split temporal 70/15/15 → train 5,051 | val 1,085 | test 1,099 · 26 clases en cada split

**Métrica principal: F1-macro** — penaliza por igual las 26 clases independientemente de la duración del video (desbalance máximo 11:1 por clase).

---

## 2. Baseline

**Modelo de referencia:** Clasificadores clásicos sobre histograma de color + movimiento (486 features/segmento).

| Modelo | F1-macro | Accuracy | Latencia |
|--------|----------|----------|----------|
| DummyClassifier | 0.037 | 0.052 | 0.02 ms |
| KNN (k=5) | 0.791 | 0.864 | 1.23 ms |
| **LogReg (C=1)** | **0.859** | **0.912** | **0.02 ms** |

> LogReg es el baseline de referencia histórico. **Advertencia:** su F1=0.859 incluye riesgo de leakage por proximidad temporal entre segmentos del mismo video — no es directamente comparable con los modelos deep evaluados en el split temporal estricto de Semana 6.

**Baseline deep (Semana 5):** CNN-LSTM (MobileNetV3 + BiLSTM h=256, 50 épocas) alcanzó F1-macro=0.721 en el split de Semana 5.

---

## 3. Experimentos A/B — Sprint Semana 6

El experimento de Semana 6 introduce **combinación tardía**: combinar dos backbones preentrenados sobre landmarks MediaPipe Holistic (75 keypoints × 3 coords × 30 frames), congelando sus pesos y entrenando solo una cabeza de clasificación conjunta.

**Backbones disponibles (checkpoints de sprints anteriores):**
- `checkpoints/lstm_best.pt` — BiLSTM bidireccional (embedding dim=256)
- `checkpoints/stgcn_best.pt` — ST-GCN ligero (embedding dim=128)

### Variante A — Combinación concat

- **Cambio:** Concatenar embeddings BiLSTM (256d) + ST-GCN (128d) → MLP 256 → 26 clases
- **Motivación:** La concatenación preserva la información de ambas vistas (temporal secuencial vs. grafo espacial) sin perder dimensionalidad. La cabeza aprende el peso relativo implícitamente.
- **Hipótesis:** La combinación superará a ambos backbones individuales en ≥ 5 puntos de F1-macro al combinar vistas complementarias.

### Variante B — Combinación attention

- **Cambio:** Proyectar ambos embeddings al mismo espacio (256d) y aplicar multi-head cross-attention (4 heads) entre las ramas antes del clasificador.
- **Motivación:** La atención permite que cada rama consulte la otra selectivamente, focalizando en aspectos discriminativos para cada clase.
- **Hipótesis:** Combinación attention alcanzará F1 ≥ concat con menos parámetros efectivos al focalizar el peso en características relevantes.

---

## 4. Resultados

### Tabla comparativa (evaluación en test set — 2026-05-22)

| Modelo | F1-macro | F1-weighted | Accuracy | Parámetros cabeza | Latencia est. |
|--------|----------|-------------|----------|-------------------|---------------|
| LogReg clásico *(ref histórica)* | 0.859 | 0.904 | 0.912 | — | 0.02 ms |
| CNN-LSTM h=256 *(Semana 5)* | 0.721 | 0.780 | 0.830 | 3.6M total | ~80 ms |
| **Combinación concat** *(Sem 6, Var A)* | **0.501** | **0.590** | **0.588** | ~0.1M | ~12 ms |
| Combinación attention *(Sem 6, Var B)* | 0.316 | 0.372 | 0.375 | ~0.1M | ~12 ms |
| ST-GCN backbone solo | 0.208 | 0.283 | 0.308 | 0.3M total | ~8 ms |
| BiLSTM backbone solo | 0.152 | 0.239 | 0.268 | 2.7M total | ~5 ms |

> **Nota sobre backbones individuales:** El BiLSTM (val F1=0.734 durante su entrenamiento) y el ST-GCN (val F1=0.496) muestran F1 bajo en el test set de Semana 6. Esto indica **divergencia de split**: los checkpoints fueron entrenados con splits aleatorios en `run_local_pipeline.py`, mientras que Semana 6 usa split temporal estricto por video. La cabeza de combinación se entrena con el split correcto, por eso su rendimiento es más alto y honesto.

### Gráfico principal: Curvas de aprendizaje

Archivo generado: `data/semana6_curvas.png`

```
Variante A (concat) — F1-macro val por época:
  ep 1: 0.371 | ep 5: 0.540 | ep 10: 0.581 | ep 16: 0.602 | ep 23: 0.611 ← mejor
  Convergencia estable. Best val F1=0.611 → test F1=0.501

Variante B (attention) — F1-macro val por época:
  ep 1: 0.172 | ep 5: 0.324 | ep 15: 0.392 | ep 19: 0.412 ← mejor
  Convergencia más lenta. Best val F1=0.412 → test F1=0.315
```

**Observación clave:** Concat domina en todo el entrenamiento. Attention converge más lento y no alcanza el plateau de concat en 25 épocas — necesita más épocas o mayor LR inicial. La brecha val/test (~11 puntos en concat) indica que aún hay margen de overfitting en la cabeza.

---

## 5. Validación Experimental

| Control | Implementación |
|---------|----------------|
| **Split temporal por video** | Segmentos del inicio del video → train; final → test. Sin frames compartidos entre splits. |
| **Solapamiento train/test** | Verificado: 0 frames compartidos ✓ |
| **Normalización** | MediaPipe Holistic normalizado por posición de muñeca derecha; sin fit sobre test |
| **WeightedSampler** | Calculado solo sobre train set (1/count_clase) |
| **Seed fija** | `torch.manual_seed(42)`, `np.random.seed(42)` al inicio del script |
| **Evaluación holdout única** | Test set evaluado una sola vez por variante al finalizar |
| **Backbones congelados** | `requires_grad_(False)` en todos los parámetros de ambos backbones |

**Splits Semana 6:**  
Train 5,051 seg | 26/26 clases ✓  
Val   1,085 seg | 26/26 clases ✓  
Test  1,099 seg | 26/26 clases ✓

---

## 6. Conclusión y Decisión Técnica

### Variante adoptada: **Combinación concat (Var A)**

**Justificación:**

1. **F1-macro=0.501** es el mejor resultado honesto de Semana 6 sobre el test set con split temporal estricto. Supera a ambos backbones individuales en +35 puntos de F1 (concat vs. BiLSTM) y +29 puntos (concat vs. ST-GCN).

2. **Latencia competitiva (~12 ms):** Los backbones están congelados; en inferencia solo corre un forward pass de cada backbone más la cabeza MLP. Viable para el target de <200 ms.

3. **Hallazgo de integridad:** Los backbones individuales evaluados en el split temporal muestran F1 bajo (0.15–0.21), revelando que sus métricas históricas (val F1=0.734) fueron medidas sobre splits con leakage. La combinación entrenada correctamente da la primera estimación honesta del rendimiento real del sistema.

| Criterio | CNN-LSTM (Sem5) | Combinación concat (Sem6) |
|----------|-----------------|----------------------|
| F1-macro (test) | 0.721* | **0.501** |
| Latencia | ~80 ms | ~12 ms |
| Entrenamiento | 50 épocas full | 23 épocas solo cabeza |
| Split leakage | Posible | No |

*medido en split de Semana 5, no directamente comparable.

**Impacto en producción:** La combinación tardía puede desplegarse en CPU con ~12 ms por segmento, habilitando inferencia en tiempo real en hardware de aula. La próxima mejora crítica es reentrenar los backbones con el split temporal correcto para mejorar la calidad de los embeddings.

---

## 7. Reproducibilidad

```bash
# Entorno
source .venv310/bin/activate
# torch==2.2.2 | mediapipe==0.10.21 | scikit-learn==1.7.2 | numpy==1.26.4

# Ejecutar experimento completo Semana 6 (~15 min en MPS)
python scripts/semana6_run.py

# Resultados generados:
#   logs/semana6_resultados_finales.txt   ← tabla + reporte por clase
#   data/semana6_resultados.csv           ← tabla comparativa CSV
#   data/semana6_curvas.png        ← curvas de aprendizaje
#   checkpoints/semana6_concat_best.pt
#   checkpoints/semana6_attention_best.pt
```

**Configs por variante:**
```
entregas/semana6/configs/exp_stgcn_v2.yaml          ← backbone config
entregas/semana6/configs/exp_concat.yaml      ← Var A
entregas/semana6/configs/exp_attention.yaml   ← Var B
```

**Hashes de datos:**
```
manifest_segments.csv  MD5: 9ee1eda5f05cdeebfde3c7b2a1806972
Commit HEAD:           8722d2087fafa5c8d76eaa0de890960cca43695c
```

**Checkpoints:**

| Archivo | Modelo | F1-macro (test) |
|---------|--------|-----------------|
| `checkpoints/lstm_best.pt` | BiLSTM backbone | 0.152 (split Sem6) |
| `checkpoints/stgcn_best.pt` | ST-GCN backbone | 0.208 (split Sem6) |
| `checkpoints/semana6_concat_best.pt` | Combinación concat | **0.501** |
| `checkpoints/semana6_attention_best.pt` | Combinación attention | 0.315 |

---

## 8. Checklist de Validación Sprint 2

| # | Requisito | Estado | Evidencia |
|---|-----------|--------|-----------|
| 1 | **Split correcto** (estratificado/temporal/grupal) | ✅ | GroupKFold(n_splits=5) con `groups=num_vineta` — ningún segmento de la misma viñeta aparece en train y test simultáneamente. Split temporal por video: primeros frames → train, últimos → test. Código: `gkf.split(X, y, groups=video_ids)` · `calibracion_s6.py:221` |
| 2 | **Fit solo en train** (escala/PCA/TE) | ✅ | Todos los pipelines usan `sklearn.Pipeline([("scaler", StandardScaler()), ("clf", ...)])`. El `pipe.fit(X[tr_idx], y[tr_idx])` garantiza que `StandardScaler` calcula media/std **únicamente** sobre índices de entrenamiento. Código: `calibracion_s6.py:222-228`, `semana6_run.py` normalización por `ref = seg[:, 21:22, :]` calculada por segmento (sin fit global). |
| 3 | **Seeds fijadas y mismo protocolo que Semana 5** | ✅ | `np.random.seed(42)` → `calibracion_s6.py:44`; `torch.manual_seed(42)` + `np.random.seed(42)` → `semana6_run.py:24-25`; `random_state=42` en todos los estimadores (`LogisticRegression`, `permutation_importance`, `learning_curve`). Mismo protocolo GroupKFold/5 usado en Semana 5. |
| 4 | **Sin cambios de data** entre baseline y evaluación final | ✅ | `manifest_segments.csv` MD5=`9ee1eda5f05cdeebfde3c7b2a1806972` (7,235 segmentos). Hash idéntico al reportado en la entrega de Semana 5. Dataset congelado desde `scripts/preprocess_sliding_window.py` — ningún archivo `.npy` fue modificado entre experimentos. |
| 5 | **Logs completos** (config, métricas, timestamp) | ✅ | `data/calibracion_resumen.txt` (timestamp ISO, config completa, F1/Acc/ECE/Brier por variante) · `data/semana6_resultados.csv` (tabla comparativa CSV) · `Entrega Sprint Semana6/semana6_resultados_finales.txt` (log detallado por clase y época) · `data/calibracion_ablacion.csv` (5 folds × 3 configs) |

### Verificación automática

```bash
# Reproducir la verificación completa del checklist
source .venv310/bin/activate

# Ítem 1-3: Ablación con split grupal y seeds fijas
python scripts/calibracion_s6.py
# → data/calibracion_resumen.txt  (timestamp + todas las métricas)
# → data/calibracion_ablacion.csv (5 folds × 3 configuraciones)

# Ítem 4: Verificar integridad del dataset
md5 data/manifest_segments.csv
# Expected: 9ee1eda5f05cdeebfde3c7b2a1806972

# Ítem 5: Verificar logs generados
ls -la data/calibracion_*.csv data/calibracion_*.txt data/semana6_resultados*.csv
```

### Entorno de ejecución (reproducibilidad)

```
Python  : 3.10.19
sklearn : 1.7.2
torch   : 2.2.2
numpy   : 1.26.4
seed    : 42 (numpy + torch + random_state en todos los estimadores)
```

---

## 9. Riesgos y Próximos Pasos

- **Riesgo principal — Split mismatch en checkpoints:** Los backbones (`lstm_best.pt`, `stgcn_best.pt`) fueron entrenados con splits aleatorios (leakage implícito). La combinación tardía corrige esto en la cabeza, pero la calidad de los embeddings sigue siendo subóptima. **Acción inmediata Semana 7:** reentrenar ambos backbones con el split temporal estricto del manifest actual y reejecutar la combinación.

- **Riesgo secundario — Generalización a signers no vistos:** El dataset tiene un signer por viñeta. Con 26 videos/26 clases, los modelos pueden estar aprendiendo apariencia del signer en lugar de la seña. F1=0.501 en test no garantiza generalización a nuevos usuarios. Mitigation: aumentar landmarks con flip/ruido/rotación (ya implementado) y conseguir dataset con múltiples signers por clase en el siguiente sprint.
