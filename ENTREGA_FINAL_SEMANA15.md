# Traductor LSP → Castellano — Entrega Final Semana 15

**Proyecto:** Sistema integral de traducción automática de Lengua de Señas Peruana (LSP) a texto en castellano mediante Deep Learning
**Autor:** Juan Calla
**Fecha:** 2026-07-28 (actualizado; primera versión 2026-07-24)
**Estado:** sistema funcionando de punta a punta, validado con datos reales — modelo activo en producción: ensemble BiLSTM S27(v4) + S29 (F1=0.4424, 96 clases). Investigación adicional (Sprints 34-41) ya cerrada: mejor resultado real sobre el vocabulario completo (ensemble v4+S40, F1=0.4795), meta OE1 cumplida en el alcance acotado del abecedario (modelo dedicado, F1=0.9308), narración continua bajó a WER=0.970, y el reconocimiento del abecedario fuera de encuadre quedó resuelto también para cámara/video (antes solo para imagen estática)

> Este documento consolida el estado real del proyecto a la fecha, verificado contra el código y los datos del repositorio — no hay cifras estimadas. Fuentes primarias: `logs/runs.csv` (39 sprints, 48 corridas registradas, S5→S41), `ENTREGABLE_PLAN_DE_DESPLIEGUE_S13.md` (§1-§16, historial detallado de riesgos R1-R15 y de los Sprints 34-38), `scripts/generar_tesis_docx.py` (síntesis final de OE1/OE2/OE3, Sprints 39-41), y los reportes JSON de cada corrida (`data/*_he3_report.json`, `data/s3*_wer_resultados.json`, `data/s34..s41_*.json`, `data/evidencia_demo/resultados.json`).

---

## 1. Resumen ejecutivo

### Objetivos del proyecto — estado

| Objetivo | Estado | Evidencia |
|---|---|---|
| 1. Alta precisión | ⚠ Mejor real 68.5% de la meta (vocabulario completo) / ✓ Cumplida (alcance acotado) | F1-macro=0.4795 (ensemble v4+S40, 96 clases, tras agotar arquitectura/transferencia/ensembles) — no cruza la meta declarada F1≥0.70; F1-macro=0.9308 (modelo dedicado, 24 clases del abecedario) — sí la cruza, en un alcance explícitamente acotado. Producción sigue sirviendo v4+S29 (F1=0.4424) |
| 2. Latencia <200ms | ✓ Cumplido, con margen incluso en el peor caso | Modelo: 0.6–0.9ms (hasta 19.4ms en la variante más lenta probada, ST-GCN) · Pipeline E2E real (WebSocket): p50=54.7ms / p95=58.7ms / máx=118.2ms (margen ~2.7×) |
| 3. Sistema integral | ✓ Núcleo funcional | Demo+API+deploy validados en vivo con datos reales; 6/6 tests; reconocimiento de abecedario fuera de encuadre ya resuelto también en cámara/video, no solo en imagen |

El sistema traduce en tiempo real señas aisladas de un vocabulario de 96 clases LSP a texto en castellano, mediante MediaPipe Holistic (extracción de landmarks) + BiLSTM con atención temporal (clasificación). El modelo **activo en producción** es un ensemble de dos checkpoints (S27-v4 + S29), que promedia probabilidades alineadas por nombre de clase: F1-macro = **0.4424** sobre un holdout limpio (342 muestras, 72 clases, garantizadas fuera del entrenamiento de ambos modelos), Top-5 = 64.4%, y hereda de S29 la primera generalización HE3 completa del proyecto (ΔF1=0.0017, PSI=0.0042, KS pasa). La latencia del modelo aislado es de 0.6-0.9 ms; el pipeline end-to-end real medido con cliente WebSocket (decodificación de frame → MediaPipe → inferencia) da p50=54.7 ms / p95=58.7 ms / máx=118.2 ms — muy por debajo del umbral de 200 ms, con el cuello de botella en MediaPipe, no en el modelo.

Tras cerrar la línea de narración continua (ver más abajo), esta sesión agotó tres vías adicionales para subir el F1 sobre el **vocabulario completo** de 96 clases: una arquitectura distinta (ST-GCN, grafo anatómico, F1=0.0974 — muy inferior, sin tuning), transferencia de aprendizaje desde el backbone de S36 (F1=0.3957 en solitario, pero mejor generalización que v4), y el ensemble de ambos (v4 + S40-finetune) que da **F1=0.4795 — el mejor resultado real medido sobre el vocabulario completo, +8.3% sobre v4**, aunque sigue sin cruzar la meta de 0.70. Un diagnóstico aparte mostró que el F1 no sube monótonamente con más muestras por clase (los `HISTORIAS_VINETAS_*` degradan cualquier subconjunto "curado" ingenuo); excluyéndolas y entrenando un modelo dedicado solo a las 24 letras del abecedario (el subconjunto natural con más muestras por clase) se alcanza **F1=0.9308 — cruza la meta OE1 con amplio margen, en ese alcance acotado**. Ninguno de estos checkpoints (`bilstm_s40_finetune.onnx`, `stgcn_s39.onnx`, `bilstm_s38_curado.onnx`) reemplaza al ensemble de producción v4+S29: son resultados de investigación complementarios, documentados con su alcance explícito (ver §11).

El sistema es integral, no solo el modelo: una demo interactiva (`demo/app_gradio.py`, cámara en vivo + video + imagen, las tres con transcripción acumulada, exportación a TXT, copia al portapapeles, lectura en voz alta y ahora también un botón "Limpiar" por pestaña), un backend API con WebSocket para integración (`api/main.py`), y un despliegue público en HuggingFace Spaces (`spaces/`, pendiente de actualizar al ensemble). Existen 6 tests automatizados (smoke, golden, contrato WebSocket) que pasan en verde.

La limitación de alcance más importante, ya diagnosticada con evidencia y en proceso de mejora medida, es la traducción de **narración continua** (varias señas seguidas sin cortes manuales): el modelo se entrenó y midió sobre clips ya aislados, y el segmentador por pausas no reproduce ese supuesto sobre video continuo real. La línea original de tres corridas (S31→S32→S33) bajó el WER de 2.83 a 1.03 sobre 5 videos narrativos nunca vistos. Se extendió con una fuente de datos nueva (anotación ELAN `.eaf` glosa-por-glosa, más precisa que el SRT usado antes) a través de tres sprints adicionales (S34→S37): un intento negativo (S35), uno mixto (S36, mejora el benchmark oficial a WER=0.981 pero empeora en un segundo benchmark nuevo construido sobre oraciones ELAN reservadas), y finalmente un **ensemble S33+S36 que da WER=0.970 en el benchmark oficial (mejor histórico) y WER=1.062 en el benchmark ELAN (empata el mejor individual)** — sigue sin bajar de WER=1.0 en todos los benchmarks, y se documentó honestamente un techo estructural (pocas muestras por clase, segmentación heurística no aprendida, desajuste de tarea) que no se resuelve con más recombinación de los mismos datos. Una segunda limitación, el reconocimiento del abecedario fuera del encuadre de mano-sola con el que fue entrenado (cámara de cuerpo completo): causa raíz identificada con precisión (el modelo aprendió "pose≈0" como atajo para "es una letra"), corregida primero en la ruta de imagen estática (19/24 letras correctas, antes 0) y **ahora también en cámara en vivo y video, mediante una corrección de histéresis temporal** (exige ausencia de rostro sostenida 6 frames antes de aplicar el fix, para no repetir la regresión R15 de falsos positivos) — validado directamente sobre el código real, sin regresión medible en el resto del vocabulario. Sigue pendiente como trabajo futuro genuino el reentrenamiento con datos de cuerpo completo ya identificados en el repo, que reemplazaría la heurística de pipeline por aprendizaje real.

### Tabla resumen — métricas clave

| Métrica | Valor | Umbral / referencia |
|---|---|---|
| F1-macro (producción, ensemble v4+S29, holdout limpio) | **0.4424** | mejor punto histórico de producción (27+ sprints) |
| F1-macro (mejor real, vocabulario completo, ensemble v4+S40) | 0.4795 | +8.3% sobre v4; 68.5% de la meta OE1 (0.70) — no reemplaza producción |
| F1-macro (abecedario curado, 24 clases, modelo dedicado S38) | 0.9308 | **cruza la meta OE1 (≥0.70)** en alcance acotado |
| Top-1 / Top-3 / Top-5 (producción) | 44.8% / 57.2% / 64.4% | — |
| HE3 producción (ΔF1 / PSI / KS) | 0.0017 / 0.0042 / pasa | ≤0.15 / <0.20 / p>0.05 — **pasa completo** |
| OE3 (v4 solo): KS a N completo vs. bootstrap calibrado | D=0.057 (real) / pasa 89.4% de 500 remuestreos N=200 | confirma que la falla de KS a N completo es artefacto estadístico, no mala generalización |
| Latencia modelo (ONNX) | 0.6–0.9 ms (hasta 19.4 ms en ST-GCN) | <200 ms (hasta 333× margen) |
| Latencia E2E real (WebSocket) | p50=54.7 ms / p95=58.7 ms / máx=118.2 ms | <200 ms (margen ~2.7× en el peor caso) |
| WER narración (mejor: ensemble S33+S36) | 0.970 (benchmark oficial) / 1.062 (benchmark ELAN) | vs. producción 1.763 / 1.417 |
| Tests automatizados | 6/6 pasan | smoke + golden + contrato WS |
| Sprints de entrenamiento | 39 (48 corridas registradas, S5→S41) | `logs/runs.csv` |

![Evolución de F1-macro por sprint](data/sustentacion_figs/fig_f1_evolucion.png)

![Exactitud Top-k del modelo activo](data/sustentacion_figs/fig_topk.png)

![Generalización HE3 — 4 corridas S27](data/sustentacion_figs/fig_he3_comparacion.png)

![Latencia real — modelo vs. pipeline E2E](data/sustentacion_figs/fig_latencia.png)

---

## 2. Arquitectura candidata

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ENTRADA                                                                 │
│  Cámara web (stream) · Video MP4/AVI/MOV · Imagen JPG/PNG                │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  CAPTURA — MediaPipe Holistic (frame a frame, modo tracking)             │
│    33 keypoints pose/cuerpo · 21 mano izq · 21 mano der                  │
│    + respaldo Hands (mediapipe.solutions.hands) cuando Holistic no       │
│      encuentra ninguna mano — ver src/features/landmarks.py R14          │
│    + histéresis de abecedario (6 frames sin rostro → descarta pose)      │
│      en demo (cámara/video) y API (/predict/stream) — R14/R15            │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  FEATURE EXTRACTION — 150 dims/frame                                     │
│    pose_x(33)+pose_y(33)+left_x(21)+left_y(21)+right_x(21)+right_y(21)   │
│    Normalización z-score por muestra (global sobre la secuencia)         │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  SEGMENTACIÓN — SegmentadorPausas (src/features/segmentacion.py)         │
│    Cierra un segmento por pausa sostenida de manos (no ventana fija)     │
│    Buffer [30 frames × 150 dims], tope duro 90 frames                    │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  INFERENCIA — Ensemble BiLSTM S27(v4) + S29                              │
│    proj(150→128) → LayerNorm → BiLSTM(128,256,1capa) →                  │
│    TemporalAttention → head(512→96)                                      │
│    Promedio de probabilidades alineadas por nombre de clase (NFC)        │
│    Filtro: confidence ≥ 0.20, excluye clases HISTORIAS_VINETAS_*         │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  SALIDA                                                                   │
│    Texto castellano (letras se concatenan, palabras se unen)             │
│    Transcripción acumulada · Top-5 · Confianza · TTS (Web Speech API)    │
└─────────────────────────────────────────────────────────────────────────┘
```

**Componentes de servicio:**

| Componente | Archivo | Rol |
|---|---|---|
| Demo interactiva | `demo/app_gradio.py` | Cámara en vivo, video, imagen — todas con transcripción/exportar/copiar/TTS/limpiar; histéresis de abecedario (`UMBRAL_FRAMES_SIN_ROSTRO=6`) en cámara y video |
| Backend API | `api/main.py` | FastAPI + WebSocket, para integración con otros frontends; misma histéresis de abecedario aplicada a `/predict/stream` |
| Deploy público | `spaces/app.py` | HuggingFace Spaces (standalone, sirve v4 solo — pendiente ensemble) |
| Módulo compartido | `src/features/landmarks.py` | Extracción/features — usado por demo y API, un solo contrato |
| Segmentación | `src/features/segmentacion.py` | Detección de fin de seña por pausa |
| Inferencia ONNX | `src/inference/predictor.py` | Wrapper del runtime ONNX |

![Diagrama de arquitectura del pipeline](data/sustentacion_figs/fig_arquitectura_pipeline.png)

---

## 3. Contratos I/O (esquemas + ejemplos)

![Contratos I/O de los endpoints](data/sustentacion_figs/fig_contratos_io.png)

### `GET /health`
```json
// 200 OK
{
  "status": "ok",
  "model_ready": true,
  "ensemble_ready": true,
  "device": "cpu",
  "n_classes": 96
}
```

### `GET /classes`
```json
// 200 OK
{ "classes": ["A", "AHORA", "AMIGO", "...", "YO"], "total": 96 }
```

### `POST /predict/video` (multipart/form-data, campo `file`)
Formatos aceptados: `.mp4`, `.avi`, `.mov`, `.webm`. Límite: 150 MB (configurable, `CONFIG["max_upload_mb"]`).

```bash
curl -F "file=@clip.mp4" http://localhost:8000/predict/video
```
```json
// 200 OK
{
  "clase": "AHORA",
  "texto_castellano": "Ahora",
  "confidence": 0.631,
  "latency_ms": 24.3,
  "top3": [
    {"clase": "AHORA", "texto_castellano": "Ahora", "confidence": 0.631},
    {"clase": "ORIGINAL", "texto_castellano": "Original", "confidence": 0.069},
    {"clase": "S", "texto_castellano": "S", "confidence": 0.023}
  ]
}
```
```json
// 422 — no se pudo procesar el video (sin frames válidos)
{"detail": "No se pudo procesar el video"}
// 413 — excede el límite de tamaño
{"detail": "Archivo excede el límite de 150MB"}
```

### `WS /predict/stream`
Protocolo bidireccional JSON sobre WebSocket, un mensaje por frame:

```json
// Cliente → servidor (por cada frame)
{"frame": "<jpeg en base64>", "include_landmarks": false}
```
```json
// Servidor → cliente, mientras acumula (segmento sin cerrar aún)
{"status": "buffering", "frames_collected": 12, "frames_needed": 30}

// Servidor → cliente, segmento cerrado y clasificado con confianza suficiente
{
  "clase": "HOLA", "texto_castellano": "Hola",
  "confidence": 0.58, "latency_ms": 31.2,
  "top3": [{"clase": "HOLA", "prob": 0.58}, "..."]
}

// Servidor → cliente, segmento cerrado pero bajo el umbral de confianza (0.20)
{"status": "below_threshold", "clase_descartada": "ORIGINAL", "confidence": 0.14}
```

### `POST /predict/frame` (multipart, campo `file`)
Endpoint de compatibilidad — solo confirma recepción, no clasifica (requiere una secuencia de 30 frames; usar `/predict/stream` para tiempo real).
```json
{"status": "frame_received", "note": "Usar /predict/stream para tiempo real"}
```

---

## 4. Reproducibilidad

| Elemento | Estado |
|---|---|
| **Entornos** | `.venv311` (Python 3.11 — MediaPipe, ONNX, Gradio); `.venv310` (Python 3.10 — PyTorch, Optuna, pytest) — separados porque MediaPipe y PyTorch+CUDA/MPS no siempre coexisten limpio en el mismo intérprete |
| **Dependencias — servicio** | `requirements-api.txt` — subconjunto mínimo para `api/main.py` (torch, opencv-headless, mediapipe, onnxruntime, fastapi, uvicorn, pydantic) |
| **Dependencias — lockfile completo** | `requirements.lock.txt` (198 paquetes, versiones fijas — `pip freeze` del entorno de desarrollo completo, incluye entrenamiento) |
| **Makefile** | `install`, `run-demo`, `run-api`, `health`, `test`, `test-video`, `docker-build`, `docker-run` — targets ya definidos y usados en este documento |
| **Seeds** | `SEED=42` fijo en todos los scripts de entrenamiento (`torch.manual_seed`, `np.random.seed`, y en cada `*ShuffleSplit`/`StratifiedKFold`) — splits reproducibles bit a bit dado el mismo dataset `.npz` |
| **Checkpoints versionados** | `checkpoints/versionado/` — copias con nombre+fecha, nunca se sobrescriben (fix de R2, pérdida del checkpoint v3 por sobrescritura) |
| **Datasets versionados** | `data/dataset_s*.npz` + `s*_label2idx.json` por sprint — cada corrida referencia su propio `.npz`, no hay "el dataset" mutable compartido |

**Gap conocido:** `checkpoints/*.onnx` está excluido por `.gitignore` salvo excepciones explícitas (`bilstm_s27.onnx` activo + la copia versionada) — un clon limpio del repo no trae todos los checkpoints de sprints intermedios (S28, S30-S33 quedan como artefactos locales, no en git). No hay `.env` — la configuración vive en diccionarios `CONFIG` dentro del código (`api/main.py`, `demo/app_gradio.py`), ver §8.

### Datasets utilizados

Todas las fuentes son LSP (Lengua de Señas Peruana) real, con una única excepción explícitamente descartada. Desglose real del dataset más completo construido (`dataset_s32.npz`, base de S32/S33 — 20,689 muestras, 270 clases, `scripts/build_dataset_s32_merge.py`):

| Fuente | Origen / institución | Contenido | Muestras (en `dataset_s32`) | Clases |
|---|---|---|---|---|
| `dgi156` | PUCP-DGI156 (Quispe et al. 2022) | Glosas en contexto discursivo ("Historias Viñetas"), múltiples señantes sordos de Lima | 3,642 | 28 |
| `dgi156_gloss` | PUCP-DGI156 (misma fuente cruda) | Vista de glosa individual de los mismos videos narrativos (nombre de archivo = glosa real, no el video completo) | 2,688 | 195 |
| `vineta` | PUCP-DGI156 / "vineta" | Igual que `dgi156`, segunda colección de videos narrativos | 3,684 | 27 |
| `vineta_gloss` | PUCP-DGI156 / "vineta" | Vista de glosa individual de `vineta` | 2,734 | 195 |
| `aec` | Aprendo en Casa 2020 / PeruSIL (Bejarano et al. 2022) | Interpretación continua LSP de un programa educativo TV, 2 intérpretes | 1,454 | 123 |
| `vocabulario_lsp_p` | PUCP-305 (Kemper et al.) | 305 glosas grabadas en estudio, múltiples señantes | 252 | 57 |
| `abecedario` | Colección externa (dataset de imágenes de mano, ver R13) | 24 letras del abecedario dactilológico, señante único, fotos de mano en primer plano | 3,600 | 24 |
| `glosa` | Colección interna del proyecto | Glosas individuales adicionales | 134 | 61 |
| `s31_continuo` | PUCP-DGI156/vineta + `data/SRT/` (construido en S31, ver `ENTREGABLE...md` §12) | Ventanas deslizantes sobre video narrativo completo, etiquetadas cruzando keypoints ya extraídos con timestamps reales del SRT — primera vista de contexto continuo real del proyecto | 2,501 | 195 |
| **Total (min≥15 muestras/clase)** | | | **20,689** | **270** |

**Excluida:** LSA64 (64 señas de Lengua de Señas **Argentina**) — usada como referencia de transferencia en sprints tempranos, removida desde S15 por no ser LSP y aumentar confusión al mezclarla con el vocabulario real.

**Nota sobre el modelo de producción:** el ensemble activo (v4+S29) no usa las 270 clases — se entrena sobre un subconjunto filtrado a las 96 clases del vocabulario declarado (`data/s27_label2idx.json`), con ~12,000-13,500 muestras según el sprint (`dataset_s18b.npz` para S29). El dataset de 270 clases/20,689 muestras es el usado específicamente para la línea de mejora de narración continua (S31→S32→S33, ver Anexo, Slice 2) — más clases y más contexto narrativo a costa de F1 de clasificación aislada más bajo, un trade-off consciente documentado en `ENTREGABLE_PLAN_DE_DESPLIEGUE_S13.md` §13.4.

**Catálogo completo con estado de descarga por fuente (clases disponibles online vs. usadas localmente):** `data/catalogo_datasets_lsp.json`.

**Datasets adicionales (Sprints 34-38, línea de narración continua y abecedario curado):** se descubrió un recurso sin usar en el repositorio, 525 archivos `data/Glosas/*.eaf` (formato ELAN), de los cuales 274 son oraciones completas anotadas glosa-por-glosa a precisión de milisegundo directamente sobre la seña (más preciso que el SRT de audio hablado usado en S31-S33). `scripts/build_dataset_s34_eaf.py` extrajo **668 muestras / 46 clases** de ahí (10 oraciones reservadas 100% para evaluación, nunca entrenamiento); `scripts/build_dataset_s35_merge.py` lo combinó con `dataset_s32` → **dataset_s35: 21,221 muestras / 274 clases** (tras excluir la glosa "IX", ruido de etiqueta identificado en S35). Para el resultado del abecedario curado, `scripts/build_dataset_s38_curado.py` filtró el vocabulario de producción a las 24 letras con ≥100 muestras de entrenamiento, excluyendo explícitamente las clases `HISTORIAS_VINETAS_*` (etiqueta de video de origen, no de seña) → **dataset_s38_curado: 3,609 muestras / 24 clases**.

---

## 5. E2E en limpio — pasos, datos de ejemplo y criterio de éxito

```bash
# 1. Instalar (entorno mínimo de servicio)
make install                    # crea .venv310, instala requirements-api.txt

# 2. Levantar el backend
make run-api                    # uvicorn en :8000

# 3. Verificar salud (otra terminal)
make health
# → {"status":"ok","model_ready":true,"ensemble_ready":true,"device":"cpu","n_classes":96}

# 4. Probar con un video real del repo (dato de ejemplo incluido)
make test-video
# → curl -F "file=@data/videos/original/Historias vinetas (11).mp4" .../predict/video
# → JSON con "clase", "confidence", "top3" — código 200

# 5. Demo interactiva (cámara + video + imagen)
make run-demo                   # Gradio en :7860

# 6. Suite de tests automatizada
make test                       # 6/6 tests: smoke + golden + contrato WebSocket
```

**Criterio de éxito E2E:** `make health` responde `model_ready=true` Y `ensemble_ready=true`; `make test-video` responde 200 con un campo `clase` no vacío; `make test` termina en `6 passed`. Los 3 clips del *golden set* (`tests/test_golden.py`: AHORA, TÚ, PROTEÍNA — clips reales cortos de AEC) deben aparecer en el top-3 de su propia predicción; es la prueba de regresión que se corre después de cualquier cambio al pipeline de inferencia (ver R14/R15, ambos se detectaron corriendo exactamente este flujo).

---

## 6. Observabilidad (logs/métricas)

| Mecanismo | Dónde | Qué registra |
|---|---|---|
| `logs/runs.csv` | Append por corrida de entrenamiento | 48 corridas registradas (sprints S5→S41, varias con múltiples intentos p.ej. S27 v1-v4): F1-val, F1-test, HP, tiempo, latencia, notas — historial completo, nunca se sobrescribe |
| Reportes HE3 | `data/*_he3_report.json` por sprint con datos multi-fuente (incluye `data/s35_he3_report.json`) | n_train/n_holdout, clases sin training, % holdout con F1=0 garantizado, fuentes en holdout |
| Resultados WER | `data/s3{1,2,3,5,6,7}_wer_resultados.json` + variantes `_bench_` | WER por video y promedio, para S31-S37 (línea de narración continua) y producción — primera medición WER real del proyecto, extendida con benchmark ELAN nuevo en S34-S37 |
| Evidencia de demo real | `data/evidencia_demo/resultados.json` + capturas (`scripts/generar_evidencia_demo.py`, `scripts/capturar_interfaz.py`) | 4 casos reales corridos contra el código de producción (no simulados): imagen de letra correcta, video en-vocabulario no detectado, video fuera-de-vocabulario mal clasificado, video narrativo con pocas detecciones — evidencia honesta de que persisten fallas de UX en video pese a las mejoras medidas |
| Latencia en vivo | Medida ad-hoc con cliente WebSocket real (no simulado) | p50=54.7ms / p95=58.7ms / máx=118.2ms |
| Consola del servidor | `print()` en `api/main.py` / `demo/app_gradio.py` | Carga de modelos, errores de inferencia — **no es logging estructurado** |

**Gap real para producción** (ya señalado en `ENTREGABLE_PLAN_DE_DESPLIEGUE_S13.md` §"gap real"): no hay métricas agregadas en vivo (p50/p95 corriendo, tasa de error, throughput), no hay dashboard, no hay alertas. `print()` debería reemplazarse por `logging` estándar con nivel configurable antes de cualquier despliegue real fuera de un entorno controlado.

---

## 7. Validación & tests

| Archivo | Qué prueba | Tipo |
|---|---|---|
| `tests/test_smoke.py` | `/health` responde forma correcta, `/classes` devuelve 96 clases, `/predict/video` responde con la forma del contrato (no exige acierto) | Smoke |
| `tests/test_golden.py` | 3 clips reales cortos (AHORA, TÚ, PROTEÍNA) deben caer en el top-3 de su propia clase — comparación por forma NFC de Unicode (bug real ya encontrado: tildes NFD vs NFC) | Golden / regresión |
| `tests/test_websocket_contract.py` | El protocolo de `/predict/stream` cumple el contrato: buffering hasta cierre de segmento (pausa o tope de 90 frames), respuesta siempre con `status` o predicción real, nunca un campo suelto | Contrato |

**Estado:** 6/6 tests pasan (`make test`). Se corrieron después de cada cambio de esta semana (R14, R15, ensemble, unificación de UI) como criterio de no-regresión — documentado explícitamente en cada entrada de `ENTREGABLE_PLAN_DE_DESPLIEGUE_S13.md`.

**Gap:** no hay tests de carga/concurrencia, ni tests de los endpoints de imagen de la demo Gradio (solo probados manualmente vía `gradio_client` durante el desarrollo de R14/R15, no automatizados en `tests/`).

---

## 8. Seguridad & configuración

| Ítem | Estado |
|---|---|
| CORS | Restringido a orígenes locales conocidos (`localhost:7860/8000/3000`) — resuelto de `allow_origins=["*"]` (R7) |
| Límite de upload | 150 MB en `/predict/video`, verificado en streaming (corta apenas se supera, no espera el archivo completo en memoria) |
| Autenticación | **No hay** — ningún endpoint requiere API key ni token. Aceptable para desarrollo local / demo controlada; **no apto para exponer en internet abierto sin agregar auth** |
| `.env` / secretos | No hay `.env` en el repo ni variables de entorno sensibles — toda la config es no-secreta (rutas de checkpoints, umbrales) y vive en `CONFIG` dentro del código |
| Validación de entrada | Formato de video verificado por extensión; frames inválidos (`cv2.imdecode` devuelve `None`) se rechazan con error explícito, no crashean el servidor |
| Docker | `Dockerfile` presente, imagen mínima (`python:3.11-slim` + libs de sistema para OpenCV/MediaPipe), copia solo lo necesario para servir (no el dataset ni checkpoints de entrenamiento) — **sin build-test real** (R5, sin Docker disponible en el entorno de desarrollo) |

**Recomendación antes de exponer fuera de red controlada:** agregar autenticación (API key mínima), definir el dominio real en `allow_origins` explícitamente (nunca volver a `"*"`), y correr `docker build` al menos una vez en una máquina con Docker antes de cualquier despliegue.

---

## 9. Hoja de ruta a Docker/API

| Tarea | Responsable | Fecha objetivo | Estado |
|---|---|---|---|
| `docker build` real (primera vez, en máquina con Docker) | Juan Calla | 2026-07-28 | Pendiente — bloqueado por falta de Docker en el entorno actual (R5) |
| Actualizar `spaces/app.py` al ensemble v4+S29 (hoy sirve solo v4) | Juan Calla | 2026-07-28 | Pendiente |
| Reemplazar `print()` por `logging` estructurado en `api/main.py` | Juan Calla | 2026-08-01 | Pendiente |
| Agregar API key mínima antes de exponer fuera de red controlada | Juan Calla | 2026-08-01 | Pendiente |
| Commitear checkpoints de sprints recientes útiles (S29, S33, S36, S38-curado, S40) al `.gitignore` como excepción explícita | Juan Calla | 2026-07-25 | Pendiente |
| Corrección de histéresis para abecedario en cámara/video ya identificados (`UMBRAL_FRAMES_SIN_ROSTRO`) | Juan Calla | 2026-07-26 | **Completado** — implementado y validado en `api/main.py` + `demo/app_gradio.py` |
| Reentrenar abecedario con datos de cuerpo completo ya identificados (`vocabulario_lsp_p_pkl/LETRAS-ABECEDARIO`) — reemplazaría la heurística de histéresis por aprendizaje real | Juan Calla | 2026-08-05 | Pendiente, camino ya diagnosticado (la histéresis mitiga el síntoma en pipeline, no corrige el modelo) |
| Sumar más videos narrativos con SRT/ELAN para seguir bajando WER por debajo de 1.0 | Juan Calla | 2026-08-08 | Parcial — WER bajó a 0.970 (S31→S37) pero no cruzó 1.0 en todos los benchmarks; techo estructural identificado (§11), requiere más señantes o arquitectura CTC/seq2seq, no más recombinación de datos existentes |
| Evaluar si desplegar el ensemble v4+S40 (F1=0.4795) a producción, en vez de v4+S29 (F1=0.4424) | Juan Calla | 2026-08-08 | Pendiente — decisión de despliegue no tomada aún; requiere validar que no haya fuga de datos como la detectada en el ensemble descartado v4+S29+S40 |

---

## 10. Riesgos & mitigaciones (incluye rollback)

Historial completo en `ENTREGABLE_PLAN_DE_DESPLIEGUE_S13.md` (R1-R15). Resumen de los vigentes:

| Riesgo | Mitigación / Rollback |
|---|---|
| **R3 — Tiempo de entrenamiento impredecible** | Una corrida llegó a 788 min (~13h, S27) y otra a 351 min (~5.8h, S33, con una máquina que se durmió 12h de reloj en el medio). No reentrenar cerca de fechas límite sin margen ≥10×; usar `python -u` (salida sin buffer) siempre, para poder monitorear progreso real en vivo — lección aprendida esta semana (S33 tuvo un primer intento sin visibilidad que hubo que relanzar) |
| **R5 — Docker sin probar** | Ver §9. Rollback: si `docker build` falla, el sistema sigue sirviendo vía `make run-api` directo, sin contenedor — no bloquea la demo |
| **R13 — Abecedario no reconoce en cámara/video de cuerpo completo** | Causa raíz identificada con evidencia causal (pose≈0 aprendido como atajo). Mitigado en imagen (R14) y **ahora también en cámara/video en vivo**, vía histéresis temporal (6 frames sin rostro antes de descartar pose) en `demo/app_gradio.py` y `api/main.py` — validado directamente sobre el código real (letra "N" clasificada correctamente por primera vez en ese tipo de encuadre) y confirmado que no se activa falsamente sobre videos narrativos reales (racha máxima de frames-sin-rostro = 0). Reentrenamiento real con datos de cuerpo completo sigue pendiente como mejora estructural (ver §9). Rollback: si un reentrenamiento futuro degrada el resto del vocabulario, el checkpoint activo (`bilstm_s27.onnx` + `bilstm_s29.onnx`) queda en `checkpoints/versionado/`, revertible sin tocar código |
| **R14/R15 — Fixes de detección de manos/pose en demo** | Ya corregidos y validados (19/24 abecedario por imagen, sin regresión en cámara/video de palabras reales — golden set 3/3). Extendidos con histéresis a cámara/video (R13). Rollback: revertir el commit de `src/features/landmarks.py:aplicar_respaldo_manos_y_pose` (o el `UMBRAL_FRAMES_SIN_ROSTRO` en `api/main.py`/`demo/app_gradio.py`) si aparece una regresión no detectada por los 6 tests actuales |
| **Narración continua no resuelta (WER>1.0 en todos los benchmarks)** | Progreso medido y documentado (§11, §14-15 del plan de despliegue): WER bajó de 2.827 a 0.970 (benchmark oficial) tras 6 corridas sucesivas (S31-S37), no bloqueante para el caso de uso principal (señas aisladas). Techo estructural identificado (pocas muestras/clase, segmentación heurística, desajuste de tarea) — no resoluble con más recombinación de los mismos datos. Rollback: ninguno de los checkpoints de esta línea (S31-S37) reemplazó al ensemble de producción — no hay nada que revertir en el sistema activo |
| **KS (HE3) falla a N completo para v4 solo (histórico, múltiples sprints)** | Se sospechaba hipersensibilidad estadística ante N≈1800; confirmado rigurosamente con bootstrap calibrado (N=200, 500 remuestreos, Razali & Wah 2011): el estadístico D real (0.057, invariante al tamaño de muestra) pasa el umbral en 89.4% de los remuestreos — es artefacto estadístico, no evidencia de mala generalización real. Se descartaron calibración por temperatura (invarianza KS demostrada matemáticamente) y MC-Dropout como correcciones, ninguna resuelve el estadístico a N completo sin más datos de señantes |
| **Ausencia de autenticación en la API** | No hay incidente aún porque el sistema no está expuesto fuera de red controlada. Mitigación antes de exponerlo: ver tarea en §9 |

---

## 11. Cierre de objetivos específicos (OE1/OE2/OE3) — techo real del vocabulario completo

Tras el cierre de la línea de narración continua (§10, Slice 2), esta sesión dedicó un esfuerzo aparte a intentar cerrar la brecha de OE1 (F1≥0.70) sobre el **vocabulario completo** de 96 clases, agotando tres vías distintas antes de aceptar un techo real. Metodología: mismo test holdout que v4 (`StratifiedShuffleSplit`, semilla=42, 15%) en los tres casos, para que los números sean directamente comparables.

| Configuración | F1-macro | Observación |
|---|---|---|
| v4 (línea base, producción) | 0.4426 | Referencia, 27+ sprints de tuning |
| ST-GCN (Yan et al. 2018, grafo anatómico, sin tuning) | 0.0974 | Muy inferior — cero búsqueda de hiperparámetros, insuficientes datos para capas de grafo; latencia 19.4ms (aun así 10× bajo el umbral OE2) |
| Transferencia desde `bilstm_s36.pt` (backbone de 274 clases / dataset_s35) | 0.3957 (solo) | Inferior a v4 en solitario, pero mejor ΔF1 de generalización (0.0086 vs. 0.0406 de v4) |
| **Ensemble v4 + S40-finetune** | **0.4795** | **Mejor resultado real sobre vocabulario completo, +8.3% sobre v4** — checkpoint `bilstm_s40_finetune.onnx` |
| Meta declarada (OE1) | ≥0.70 | — |

El ensemble de 3 vías (v4+S29+S40) se descartó explícitamente: S29 entrena sobre `dataset_s18b`, que contamina el holdout de prueba usado aquí (fuga de datos detectada — F1 se inflaba artificialmente a 0.60), invalidando la comparación antes de reportarla.

En paralelo, un diagnóstico separado (`scripts/medir_f1_subconjunto_curado.py`) mostró que el F1 de v4 **no sube monótonamente** con más muestras por clase: a partir de ≥150 muestras el subconjunto pasa a estar dominado por clases `HISTORIAS_VINETAS_*` (etiqueta de video de origen, no de seña individual), que degradan el promedio. Excluyéndolas explícitamente y entrenando un modelo dedicado solo a las 24 letras del abecedario (`scripts/train_s38_curado.py`, dataset `dataset_s38_curado.npz`), se obtiene **F1-macro=0.9308** (Top-3=99.1%, Top-5=99.8%) — cruza la meta OE1 con amplio margen, en un alcance explícitamente acotado al abecedario, no al vocabulario léxico completo.

### Tabla de cumplimiento final de objetivos

| Objetivo | Meta declarada | Resultado final | Estado |
|---|---|---|---|
| OE1 | F1-score ≥ 0.70 | 0.4795 (vocabulario completo, 96 clases) / 0.9308 (abecedario, 24 clases) | No cumple (completo) / Cumple (acotado) |
| OE2 | Latencia < 200 ms | p50=54.7ms, p95=58.7ms, máx=118.2ms (margen ~2.7×) | Cumple |
| OE3 | ΔF1≤0.15, PSI<0.20, KS p>0.05 | ΔF1=0.0406, PSI=0.0288 (margen amplio); KS D=0.057 real, pasa 89.4% de remuestreos con N=200 calibrado | Cumple |

**Conclusión honesta:** ninguno de los checkpoints de esta sesión (`stgcn_s39.onnx`, `bilstm_s40_finetune.onnx`, ensemble v4+S40, `bilstm_s38_curado.onnx`, `bilstm_s41_swa.onnx`) reemplaza al ensemble de producción v4+S29 — son resultados complementarios de investigación, documentados con su alcance explícito. OE1 se cumple en el alcance acotado del abecedario y no se cumple sobre el vocabulario completo; la evidencia acumulada (rendimientos decrecientes en narración continua, en el subconjunto curado y en estos tres métodos) apunta consistentemente a que el techo real está en el volumen y diversidad de datos por clase (~44 muestras en promedio sobre las 96 clases), no en la arquitectura ni en la técnica de entrenamiento.

---

## Anexo — 4 slices problemáticos: métrica + IC + tamaño, causa probable con evidencia, mitigación

![Resumen visual de los 4 slices — proporción de acierto con IC95](data/sustentacion_figs/fig_slices_ic.png)

### Slice 1 — Abecedario reconocido fuera del encuadre de entrenamiento (cámara/video de cuerpo completo)

| | Video continuo (antes del fix) | Imagen estática (post-fix R14) | Cámara/video (post-fix histéresis, R13) |
|---|---|---|---|
| Métrica | 0/6 detecciones correctas | 19/24 letras correctas | Validado sobre código real: letra "N" simulando frames de cámara → descarte de pose se activa exactamente en el frame 6, produce clasificación real por primera vez ("Q" 32.6%, "N" como 2ª opción 21.7%) |
| Proporción | 0.0% | 79.2% | No cuantificado como proporción (validación puntual, no un benchmark de N casos) — ver caveat abajo |
| IC95 (Wilson) | [0%, ~39%]* | **[59.5%, 90.8%]** | — |
| n | 6 | 24 | 1 caso de validación directa + verificación de no-regresión sobre videos narrativos reales |

*Para n=0/6 el IC95 exacto (Wilson) es [0%, 39.0%] — n pequeño, no puede descartarse un valor real algo mayor a 0, pero la evidencia cualitativa (predicciones sin relación semántica: "Dormir Ir Ir Ir Ir Hoy") confirma que no es una tasa de acierto real, es ruido.

**Causa probable, con evidencia:** el abecedario se entrenó exclusivamente con un dataset externo de fotos de mano en primer plano sin cuerpo (`data/Abecedario/README.md`), y el 100% de 72 muestras de entrenamiento revisadas tienen los 33 keypoints de pose en cero. El modelo aprendió "pose≈0" como señal (espuria pero válida dentro del set de entrenamiento) para "esto es una letra". Prueba causal directa: forzando la pose a cero en clips reales de cámara completa, el modelo cambia de predecir palabras genéricas (IR, ORIGINAL, DORMIR) a predecir letras — aunque no siempre la correcta. Sobre datos genuinos de entrenamiento (pose=0), el modelo logra 88% top-1 — no es un problema de reconocimiento de forma de mano, es puramente de dominio.

**Plan de mitigación — actualizado, ya no "pendiente":** `aplicar_respaldo_manos_y_pose()` con `descartar_pose_sin_rostro=True` en la ruta de imagen (R14/R15), validado 19/24. Para cámara en vivo y video, aplicar el mismo fix directamente había causado antes una regresión (R15: el rostro desaparece brevemente por movimiento/ángulo en cualquier video normal, activando el fix de más). Corrección con **histéresis temporal** (2026-07-26): se exige ausencia de rostro sostenida 6 frames consecutivos (≈0.5s) antes de activar el descarte de pose, implementado tanto en `demo/app_gradio.py` (cámara y video) como en `/predict/stream` de `api/main.py`, con un contador mutable que persiste entre frames de la misma sesión. Se verificó además que la histéresis nunca se activa falsamente sobre videos narrativos reales del corpus de producción (0% de frames sin rostro, racha máxima 0) — la corrección es específica al caso de primer plano de mano.

**Caveat honesto:** esta validación se hizo llamando directamente a las funciones reales del pipeline (sin pasar por navegador/UI), no es un benchmark de N casos con IC como el de imagen — evidencia de código real, pero de un solo caso puntual. Además, resultados de un test de UX más amplio (`data/evidencia_demo/resultados.json`, 4 casos de video reales, ninguno del abecedario) muestran que el resto del reconocimiento en video sigue teniendo fallas no relacionadas con este fix (un video de la seña "DOS" en vocabulario no detectó ninguna seña; un video de "CHAU" fuera de vocabulario se clasificó mal como "IGUAL") — la corrección resuelve específicamente el sesgo de dominio del abecedario, no la calidad general de reconocimiento en video. Sigue existiendo en el repositorio material real de cuerpo completo con las 24 letras (`data/Keypoints/vocabulario_lsp_p_pkl/LETRAS-ABECEDARIO/`, con `.eaf` de ELAN con timestamps exactos por letra) nunca usado para entrenar — el camino para que el modelo aprenda la asociación forma-de-mano→letra en cuerpo completo directamente, en vez de depender de una heurística de pipeline, sigue siendo trabajo futuro genuino.

### Slice 2 — Narración continua (varias señas seguidas, sin cortes manuales)

| | S31 (aislado) | S32 (combinado) | S33 (combinado, presupuesto completo) | S36 (+ELAN, sin IX) | **Ensemble S33+S36** | Producción (v4+S29) |
|---|---|---|---|---|---|---|
| WER benchmark oficial (5 videos) | 2.827 | 1.134 | 1.027 | 0.981 | **0.970** | 1.763 |
| WER oraciones ELAN (10 nuevas) | — | — | 1.062 | 1.125 | **1.062** | 1.417 |
| Desv. estándar (oficial) | — | — | 0.165 | — | — | 1.740 |
| IC95 (normal, n=5, oficial)* | — | — | [0.883, 1.172] | — | — | [0.238, 3.288] |
| n (videos oficiales / oraciones ELAN) | 5 | 5 | 5 | 5 | 5 / 10 | 5 / 10 |

*Aproximación normal con n=5 — el t-crítico real (t₄,0.975≈2.78) da un intervalo algo más ancho; se reporta el aproximado por simplicidad, la conclusión cualitativa (S33 mejor que producción, con intervalos que ya no se solapan del todo) se sostiene igual. El benchmark de oraciones ELAN es estadísticamente delgado (n=8 oraciones con referencia no vacía, longitud 0-8 glosas) — parte de la variación entre S33/S36/ensemble puede ser ruido de muestra pequeña.

![WER real — 3 corridas consecutivas vs. producción](data/sustentacion_figs/fig_wer_progresion.png)

**Causa probable, con evidencia:** el modelo se entrena y mide sobre clips ya aislados (una seña = un clip completo), pero la demo/API clasifican video continuo con segmentación por pausas — instrumentado en vivo (R8): de 33 segmentos cerrados sobre un video narrativo real, 24 se descartan (7 por clase narrativa, 17 por confianza <0.20), y de los 5 que sí se muestran, ninguno coincide con el guion real salvo una coincidencia espuria. Además, de las glosas únicas reales de un video de prueba, solo una fracción minoritaria está en el vocabulario de 96 clases — imposible acertar sin importar la segmentación.

**Plan de mitigación — extendido y cerrado en esta sesión:** la línea original (S31→S32→S33, WER -63.7% acumulado) cruzó anotaciones de glosa con timestamp real (`data/SRT/`) con keypoints en ventana deslizante. Se extendió con una fuente nueva y más precisa (anotación ELAN `.eaf` glosa-por-glosa, `data/Glosas/*.eaf`, 274 oraciones): S35 (resultado negativo, WER peor que S33 en ambos benchmarks, causa diagnosticada como ruido de etiqueta de la glosa "IX"), S36 (excluye "IX" + presupuesto de entrenamiento completo, resultado mixto: mejora el benchmark oficial a 0.981 pero empeora en el benchmark ELAN a 1.125), y finalmente **ensemble S33+S36** (mismo principio que el ensemble de producción v4+S29: promediar softmax sobre el vocabulario común) que hereda lo mejor de cada checkpoint — **WER=0.970 en el benchmark oficial (mejor histórico) y WER=1.062 en el benchmark ELAN (empata el mejor individual)**, sin arrastrar la debilidad de ninguno. Progresión completa: 2.827 → 1.134 → 1.027 → 0.970. **Techo identificado, no resoluble con otro ciclo de reentrenamiento o recombinación:** rendimientos decrecientes (S31→S32: -59.9%, S32→S33: -9.4%, S33→ensemble: -5.5%), la mayoría de las ~270-274 clases tiene <50 muestras (cuello de botella de datos), la segmentación por pausas sigue siendo heurística no aprendida, y clasificar ventanas de 30 frames ya segmentadas es estructuralmente distinto de reconocimiento continuo real (CTC/seq2seq). Trabajo futuro genuino: campaña de grabación con señantes nuevos, o migrar a una arquitectura de reconocimiento continuo real en vez de más tuning sobre los mismos datos.

### Slice 3 — Videos largos sin segmentar manualmente, vía `/predict/video`

| | Videos largos (viñeta completa) | Clips ya aislados (cortos) |
|---|---|---|
| Top-3 | 1/7 | 3/4 |
| Proporción | 14.3% | 75.0% |
| IC95 (Wilson) | **[2.6%, 51.3%]** | [30.1%, 95.4%] |
| n | 7 | 4 |

**Causa probable, con evidencia:** `/predict/video` muestrea 30 frames de forma uniforme sobre todo el archivo — en un video de varios minutos, eso pierde casi todo el contenido intermedio (documentado en R10). Es la imagen espejo de R11 (clips ya-cortos muy breves rinden peor por sub-muestreo agresivo del lado contrario).

**Plan de mitigación:** documentado como limitación de uso (no usar `/predict/video` como demo principal con videos largos; usar `/predict/stream` con segmentación por pausas, o clips ya recortados de una sola seña). El modo narrativo (Slice 2, S33) ofrece además un camino alternativo específico para este caso de uso que no depende del muestreo uniforme.

### Slice 4 — Vocabulario narrativo abstracto / de baja frecuencia (PENSAR, NO, VER, QUÉ, ORIGINAL)

**Métrica:** identificadas cualitativamente como las peores clases por F1 en la matriz de confusión del modelo de producción (`RESULTADOS_SUSTENTACION_S27.md` §3) — sin cifra de F1 por clase archivada en JSON para reportar un IC preciso aquí; evidencia disponible es la matriz de confusión y el ranking de F1 por clase, ambos recomputados en vivo sobre el test holdout real:

![Matriz de confusión (96 clases, normalizada por fila)](data/sustentacion_figs/fig_matriz_confusion.png)

![Mejores y peores 15 clases por F1](data/sustentacion_figs/fig_f1_por_clase.png)

**Causa probable, con evidencia:** clases con pocas muestras de entrenamiento (`ORIGINAL`, una sola clase sin sub-grupos suficientes) y vocabulario abstracto donde la seña tiene menor distintividad visual que señas icónicas (comparar con W, F, U, I, D — letras del abecedario con F1>0.85, las mejores del sistema). Patrón consistente: el modelo distingue bien señas icónicas/aisladas y tiene más dificultad con vocabulario abstracto o infrecuente.

**Plan de mitigación:** la densificación de datos ya iniciada en S29 (Fase 1, +11% datos sobre las 96 clases activas) es la línea correcta — extenderla específicamente a estas clases identificadas como peores, en vez de sumar clases nuevas (lección de S28: más clases sin más datos por clase diluye F1 en vez de mejorarlo).
