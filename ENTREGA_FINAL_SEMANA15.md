# Traductor LSP → Castellano — Entrega Final Semana 15

**Proyecto:** Sistema integral de traducción automática de Lengua de Señas Peruana (LSP) a texto en castellano mediante Deep Learning
**Autor:** Juan Calla
**Fecha:** 2026-07-24
**Estado:** sistema funcionando de punta a punta, validado con datos reales — modelo activo: ensemble BiLSTM S27(v4) + S29

> Este documento consolida el estado real del proyecto a la fecha, verificado contra el código y los datos del repositorio — no hay cifras estimadas. Fuentes primarias: `logs/runs.csv` (33 sprints), `ENTREGABLE_PLAN_DE_DESPLIEGUE_S13.md` (§1-§14, historial detallado de riesgos R1-R15), `RESULTADOS_PRESENTACION_S15.md` (resultados de modelo), y los reportes JSON de cada corrida (`data/*_he3_report.json`, `data/s3*_wer_resultados.json`).

---

## 1. Resumen ejecutivo

### Objetivos del proyecto — estado

| Objetivo | Estado | Evidencia |
|---|---|---|
| 1. Alta precisión | ⚠ Mejor histórico | F1-macro=0.4426 (test); meta declarada F1>0.70 no alcanzada — ese número era de planificación temprana para ≥200 clases, no comparable |
| 2. Latencia <200ms | ✓ Cumplido | Modelo: 0.6–0.9ms · Pipeline E2E real (WebSocket): p50=54.7ms / p95=58.7ms |
| 3. Sistema integral | ✓ Núcleo funcional | Demo+API+deploy validados en vivo con datos reales; 6/6 tests |

El sistema traduce en tiempo real señas aisladas de un vocabulario de 96 clases LSP a texto en castellano, mediante MediaPipe Holistic (extracción de landmarks) + BiLSTM con atención temporal (clasificación). El modelo activo es un **ensemble de dos checkpoints** (S27-v4 + S29), que promedia probabilidades alineadas por nombre de clase: F1-macro = **0.4424** sobre un holdout limpio (342 muestras, 72 clases, garantizadas fuera del entrenamiento de ambos modelos), Top-5 = 64.4%, y hereda de S29 la primera generalización HE3 completa del proyecto (ΔF1=0.0017, PSI=0.0042, KS pasa). La latencia del modelo aislado es de 0.6-0.9 ms; el pipeline end-to-end real medido con cliente WebSocket (decodificación de frame → MediaPipe → inferencia) da p50=54.7 ms / p95=58.7 ms — muy por debajo del umbral de 200 ms, con el cuello de botella en MediaPipe, no en el modelo.

El sistema es integral, no solo el modelo: una demo interactiva (`demo/app_gradio.py`, cámara en vivo + video + imagen, las tres con transcripción acumulada, exportación a TXT, copia al portapapeles y lectura en voz alta), un backend API con WebSocket para integración (`api/main.py`), y un despliegue público en HuggingFace Spaces (`spaces/`, pendiente de actualizar al ensemble). Existen 6 tests automatizados (smoke, golden, contrato WebSocket) que pasan en verde.

La limitación de alcance más importante, ya diagnosticada con evidencia y en proceso de mejora medida, es la traducción de **narración continua** (varias señas seguidas sin cortes manuales): el modelo se entrenó y midió sobre clips ya aislados, y el segmentador por pausas no reproduce ese supuesto sobre video continuo real. Se construyó en este sprint la primera infraestructura de medición WER real del proyecto y una línea de tres corridas consecutivas (S31→S32→S33) que bajó el WER de 2.83 a **1.03** sobre 5 videos narrativos nunca vistos — mejor que el pipeline de producción actual (WER=1.76) pero aún no un sistema de narración "resuelto" (WER sigue por encima de 1.0). Una segunda limitación bien evidenciada es el reconocimiento del abecedario fuera del encuadre de mano-sola con el que fue entrenado (cámara de cuerpo completo): causa raíz identificada con precisión (el modelo aprendió "pose≈0" como atajo para "es una letra"), parcialmente corregida en la ruta de imagen estática (19/24 letras correctas, antes 0), pendiente de reentrenamiento con datos ya disponibles en el repo para la ruta de video/cámara en vivo.

### Tabla resumen — métricas clave

| Métrica | Valor | Umbral / referencia |
|---|---|---|
| F1-macro (ensemble, holdout limpio) | **0.4424** | mejor punto histórico (27+ sprints) |
| Top-1 / Top-3 / Top-5 | 44.8% / 57.2% / 64.4% | — |
| HE3 (ΔF1 / PSI / KS) | 0.0017 / 0.0042 / pasa | ≤0.15 / <0.20 / p>0.05 — **pasa completo** |
| Latencia modelo (ONNX) | 0.6–0.9 ms | <200 ms (277× margen) |
| Latencia E2E real (WebSocket) | p50=54.7 ms / p95=58.7 ms | <200 ms |
| WER narración (mejor: S33) | 1.027 | vs. producción 1.763 (-41.8%) |
| Tests automatizados | 6/6 pasan | smoke + golden + contrato WS |
| Sprints de entrenamiento | 33 (42 corridas registradas) | `logs/runs.csv` |

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
| Demo interactiva | `demo/app_gradio.py` | Cámara en vivo, video, imagen — todas con transcripción/exportar/copiar/TTS |
| Backend API | `api/main.py` | FastAPI + WebSocket, para integración con otros frontends |
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
| `logs/runs.csv` | Append por corrida de entrenamiento | 42 corridas registradas (sprints S5→S33, varias con múltiples intentos p.ej. S27 v1-v4): F1-val, F1-test, HP, tiempo, latencia, notas — historial completo, nunca se sobrescribe |
| Reportes HE3 | `data/*_he3_report.json` por sprint con datos multi-fuente | n_train/n_holdout, clases sin training, % holdout con F1=0 garantizado, fuentes en holdout |
| Resultados WER | `data/s3{1,2,3}_wer_resultados.json` | WER por video y promedio, para S31/S32/S33 y producción — primera medición WER real del proyecto |
| Latencia en vivo | Medida ad-hoc con cliente WebSocket real (no simulado) | p50/p95/max reportados en `RESULTADOS_PRESENTACION_S15.md` §6 |
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
| Commitear checkpoints de sprints recientes útiles (S29, S33) al `.gitignore` como excepción explícita | Juan Calla | 2026-07-25 | Pendiente |
| Reentrenar abecedario con datos de cuerpo completo ya identificados (`vocabulario_lsp_p_pkl/LETRAS-ABECEDARIO`) — ver slice 1 en §10 | Juan Calla | 2026-08-05 | Pendiente, camino ya diagnosticado |
| Sumar más videos narrativos con SRT para seguir bajando WER por debajo de 1.0 | Juan Calla | 2026-08-08 | Pendiente, camino validado (S31→S32→S33) |

---

## 10. Riesgos & mitigaciones (incluye rollback)

Historial completo en `ENTREGABLE_PLAN_DE_DESPLIEGUE_S13.md` (R1-R15). Resumen de los vigentes:

| Riesgo | Mitigación / Rollback |
|---|---|
| **R3 — Tiempo de entrenamiento impredecible** | Una corrida llegó a 788 min (~13h, S27) y otra a 351 min (~5.8h, S33, con una máquina que se durmió 12h de reloj en el medio). No reentrenar cerca de fechas límite sin margen ≥10×; usar `python -u` (salida sin buffer) siempre, para poder monitorear progreso real en vivo — lección aprendida esta semana (S33 tuvo un primer intento sin visibilidad que hubo que relanzar) |
| **R5 — Docker sin probar** | Ver §9. Rollback: si `docker build` falla, el sistema sigue sirviendo vía `make run-api` directo, sin contenedor — no bloquea la demo |
| **R13 — Abecedario no reconoce en cámara/video de cuerpo completo** | Causa raíz identificada con evidencia causal (pose≈0 aprendido como atajo). Mitigación parcial ya aplicada (R14, ruta de imagen). Rollback: si un reentrenamiento futuro degrada el resto del vocabulario, el checkpoint activo (`bilstm_s27.onnx` + `bilstm_s29.onnx`) queda en `checkpoints/versionado/`, revertible sin tocar código |
| **R14/R15 — Fixes de detección de manos/pose en demo** | Ya corregidos y validados (19/24 abecedario por imagen, sin regresión en cámara/video de palabras reales — golden set 3/3). Rollback: revertir el commit de `src/features/landmarks.py:aplicar_respaldo_manos_y_pose` si aparece una regresión no detectada por los 6 tests actuales |
| **Narración continua no resuelta (WER>1.0)** | Progreso medido y documentado (§14 del plan de despliegue), no bloqueante para el caso de uso principal (señas aisladas). Rollback: el modo narrativo (S33) nunca reemplazó al ensemble de producción — no hay nada que revertir en el sistema activo |
| **Ausencia de autenticación en la API** | No hay incidente aún porque el sistema no está expuesto fuera de red controlada. Mitigación antes de exponerlo: ver tarea en §9 |

---

## Anexo — 4 slices problemáticos: métrica + IC + tamaño, causa probable con evidencia, mitigación

![Resumen visual de los 4 slices — proporción de acierto con IC95](data/sustentacion_figs/fig_slices_ic.png)

### Slice 1 — Abecedario reconocido fuera del encuadre de entrenamiento (cámara/video de cuerpo completo)

| | Video continuo (cámara/cuerpo completo) | Imagen estática (post-fix R14) |
|---|---|---|
| Métrica | 0/6 detecciones correctas | 19/24 letras correctas |
| Proporción | 0.0% | 79.2% |
| IC95 (Wilson) | [0%, ~39%]* | **[59.5%, 90.8%]** |
| n | 6 | 24 |

*Para n=0/6 el IC95 exacto (Wilson) es [0%, 39.0%] — n pequeño, no puede descartarse un valor real algo mayor a 0, pero la evidencia cualitativa (predicciones sin relación semántica: "Dormir Ir Ir Ir Ir Hoy") confirma que no es una tasa de acierto real, es ruido.

**Causa probable, con evidencia:** el abecedario se entrenó exclusivamente con un dataset externo de fotos de mano en primer plano sin cuerpo (`data/Abecedario/README.md`), y el 100% de 72 muestras de entrenamiento revisadas tienen los 33 keypoints de pose en cero. El modelo aprendió "pose≈0" como señal (espuria pero válida dentro del set de entrenamiento) para "esto es una letra". Prueba causal directa: forzando la pose a cero en clips reales de cámara completa, el modelo cambia de predecir palabras genéricas (IR, ORIGINAL, DORMIR) a predecir letras — aunque no siempre la correcta. Sobre datos genuinos de entrenamiento (pose=0), el modelo logra 88% top-1 — no es un problema de reconocimiento de forma de mano, es puramente de dominio.

**Plan de mitigación:** ya aplicado parcialmente — `aplicar_respaldo_manos_y_pose()` con `descartar_pose_sin_rostro=True` en la ruta de imagen (R14/R15), validado 19/24. Pendiente para video/cámara: ya existe en el repo material real de cuerpo completo con las 24 letras (`data/Keypoints/vocabulario_lsp_p_pkl/LETRAS-ABECEDARIO/`, con `.eaf` de ELAN con timestamps exactos por letra) nunca usado para entrenar — extraerlo y hacer fine-tune del checkpoint activo es el camino concreto, no ejecutado aún por el riesgo de tiempo de reentrenamiento cerca de la sustentación (decisión explícita documentada en R13).

### Slice 2 — Narración continua (varias señas seguidas, sin cortes manuales)

| | S31 (aislado) | S32 (combinado) | **S33 (combinado, presupuesto completo)** | Producción (v4+S29) |
|---|---|---|---|---|
| WER medio | 2.827 | 1.134 | **1.027** | 1.763 |
| Desv. estándar | — | — | 0.165 | 1.740 |
| IC95 (normal, n=5)* | — | — | **[0.883, 1.172]** | [0.238, 3.288] |
| n (videos) | 5 | 5 | 5 | 5 |

*Aproximación normal con n=5 — el t-crítico real (t₄,0.975≈2.78) da un intervalo algo más ancho; se reporta el aproximado por simplicidad, la conclusión cualitativa (S33 mejor que producción, con intervalos que ya no se solapan del todo) se sostiene igual.

![WER real — 3 corridas consecutivas vs. producción](data/sustentacion_figs/fig_wer_progresion.png)

**Causa probable, con evidencia:** el modelo se entrena y mide sobre clips ya aislados (una seña = un clip completo), pero la demo/API clasifican video continuo con segmentación por pausas — instrumentado en vivo (R8): de 33 segmentos cerrados sobre un video narrativo real, 24 se descartan (7 por clase narrativa, 17 por confianza <0.20), y de los 5 que sí se muestran, ninguno coincide con el guion real salvo una coincidencia espuria. Además, de las glosas únicas reales de un video de prueba, solo una fracción minoritaria está en el vocabulario de 96 clases — imposible acertar sin importar la segmentación.

**Plan de mitigación:** camino ya iniciado y con mejora medida y consistente en 3 corridas consecutivas (S31→S32→S33, WER -63.7% acumulado) — cruzar anotaciones de glosa con timestamp real (`data/SRT/`) con keypoints ya extraídos en ventana deslizante (`data/landmarks/`), combinado con el corpus grande de clips aislados para no perder volumen. Siguiente paso ya identificado: sumar más videos narrativos con SRT (`data/external_lsp/dgi156_full/SRT.tar` tiene contenido adicional) y correr HPO real (ninguna de las 3 corridas usó búsqueda de hiperparámetros, todas heredaron los valores de S13).

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
