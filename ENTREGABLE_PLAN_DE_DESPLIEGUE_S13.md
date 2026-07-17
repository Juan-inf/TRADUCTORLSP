# Entregable — Plan de Despliegue (S13)

**Proyecto:** Traductor LSP → Castellano
**Fecha:** 2026-07-17
**Alcance:** plan de despliegue del sistema tal como existe hoy — backend FastAPI (`api/main.py`), demo Gradio (`demo/app_gradio.py`), despliegue público (`spaces/`), y el modelo activo BiLSTM S27 (v4). No introduce componentes nuevos; documenta lo que ya está construido y validado esta semana, y lo que falta para producción real.

---

## 1. Resumen ejecutivo

El sistema traduce en tiempo real señas aisladas de un vocabulario de 96 clases LSP a texto en castellano, con latencia end-to-end medida de **p50=54.7ms / p95=58.7ms** (umbral objetivo: 200ms) y F1-macro=0.4426 con generalización verificada a señantes no vistos (ΔF1=0.0406, 3.7× de margen sobre el umbral de HE3). Existen tres superficies funcionales y validadas con datos reales esta semana: una demo interactiva (`demo/app_gradio.py`, cámara + video + TTS), un backend API con WebSocket para integración (`api/main.py`), y un despliegue público en HuggingFace Spaces (`spaces/`).

El sistema **no está listo para producción sin trabajo adicional**: no hay build de Docker verificado (falta entorno con Docker para probarlo), el checkpoint del modelo no está versionado en git, no hay suite de tests automatizados, no hay observabilidad estructurada (solo `print()`), la configuración de CORS/seguridad es la de desarrollo (`allow_origins=["*"]`), y el backend API todavía no incorpora las mejoras de calidad que sí tiene la demo (ver R9, §10) — "validado" en la tabla de §2 significa que no se cae, no que tenga la misma calidad de reconocimiento que la demo. Este documento detalla el estado exacto de cada componente y el camino concreto a producción.

**Limitación de alcance conocida y documentada:** el sistema reconoce señas aisladas, no narración continua ni gramática conectada (ver `ROADMAP_FRONTEND_Y_EVALUACION.md` y `SUSTENTACION_RESUMEN_FINAL.md`) — no es parte de este plan de despliegue, es una limitación de capacidad del modelo, no de infraestructura.

---

## 2. Arquitectura candidata

```
  Cliente (navegador / cámara)
       │
       │  captura frame (JPEG base64)
       ▼
  api/main.py — FastAPI
       │  endpoints: /health  /classes  /predict/video  WS /predict/stream
       ▼
  MediaPipe Holistic  →  landmarks [75,3]
       │
       ▼
  src/features/landmarks.py
       │  kp_seq_to_features()  →  [30,150]
       │  normalize_sample()    →  z-score
       ▼
  ONNXPredictor  →  checkpoints/bilstm_s27.onnx  (96 clases)
       │
       ▼
  Cliente  ←  {clase, texto_castellano, confidence, latency_ms, top3}
```

Rutas alternativas (mismo modelo y mismo módulo de landmarks, sin pasar por `api/main.py`):
- `demo/app_gradio.py` — Gradio standalone, cámara + video + TTS, para uso local/demo. Incluye además `src/features/segmentacion.py` (segmentación por pausas) que `api/main.py` no usa todavía.
- `spaces/app.py` — copia autónoma para HuggingFace Spaces (deploy público), no importa `src/` — tiene su propia copia inline de landmarks/normalización.

> ⚠️ **Divergencia real entre `demo/` y `api/`, encontrada al escribir este plan:** los 3 ajustes de esta semana (segmentación por pausas, filtro de clases `HISTORIAS_VINETAS_*`, umbral de confianza calibrado a 0.20) están **solo en `demo/app_gradio.py` y `spaces/app.py`**. `api/main.py` sigue con ventana fija de 30 frames sin filtrar, y su `CONFIG["confidence_threshold"]=0.40` es una variable **que nunca se usa** para filtrar nada — el WebSocket devuelve una predicción cada 30 frames sin importar la confianza, incluyendo clases narrativas. Alguien integrando contra el backend hoy tendría peor experiencia que la demo. Ver riesgo **R9** en §10.
>
> **Confirmado por código, no solo por lectura:** `notebooks/ENTREGABLE_PLAN_DE_DESPLIEGUE_S13.ipynb` §2 busca en vivo las cadenas `SegmentadorPausas` y `CONF_UMBRAL` en ambos archivos — resultado real: `api/main.py` → `SegmentadorPausas=False, CONF_UMBRAL=False`; `demo/app_gradio.py` → `SegmentadorPausas=True, CONF_UMBRAL=True`.

**Componentes y su estado:**

| Componente | Archivo | Estado |
|---|---|---|
| Extracción de landmarks | `src/features/landmarks.py` | ✅ compartido entre `api/` y `demo/` |
| Segmentación por pausas | `src/features/segmentacion.py` | ✅ calibrado con datos reales |
| Modelo | `checkpoints/bilstm_s27.onnx` (3.9MB) | ✅ entrenado, validado — ⚠️ no está en git |
| Backend API | `api/main.py` | ✅ validado con servidor real + cliente WebSocket real |
| Demo | `demo/app_gradio.py` | ✅ validado en vivo, corriendo en `:7860` |
| Deploy público | `spaces/` | ✅ actualizado al modelo S27 |
| Contenedor | `Dockerfile`, `docker-compose.yml` | ⚠️ preparado, **sin build-test real** |

---

## 3. Contratos I/O (esquemas + ejemplos)

### `GET /health`
```json
// Response 200
{"status": "ok", "model_ready": true, "device": "cpu", "n_classes": 96}
```

### `GET /classes`
```json
// Response 200
{"classes": ["A", "AHORA", "APRENDER", "..."], "total": 96}
```

### `POST /predict/video` (multipart/form-data, campo `file`)
```json
// Response 200 — capturado en vivo con `make test-video` al escribir este plan
{
  "clase": "DIEZ",
  "texto_castellano": "DIEZ",
  "confidence": 0.0984,
  "latency_ms": 4662.9,
  "top3": [
    {"clase": "DIEZ", "texto_castellano": "DIEZ", "confidence": 0.0984},
    {"clase": "ORIGINAL", "texto_castellano": "ORIGINAL", "confidence": 0.0908},
    {"clase": "IGUAL", "texto_castellano": "IGUAL", "confidence": 0.0593}
  ]
}
// Response 422 si el video no se puede procesar; 503 si el modelo no cargó
```
Nota: `api/main.py` no filtra por confianza (§2, R9) — a diferencia de la demo, esta respuesta se devuelve tal cual aunque la confianza sea baja (9.8% en este ejemplo real).

**Segunda corrida en vivo, sobre el mismo video (notebook, ejecutada después):**
```json
{
  "clase": "ORIGINAL", "texto_castellano": "ORIGINAL",
  "confidence": 0.0947, "latency_ms": 7089.8,
  "top3": [
    {"clase": "ORIGINAL", "confidence": 0.0947},
    {"clase": "DIEZ", "confidence": 0.0940},
    {"clase": "IGUAL", "confidence": 0.0377}
  ]
}
```
El top-1 cambió de `DIEZ` a `ORIGINAL` entre corridas — la diferencia de confianza entre ambos es de 0.0007 (prácticamente empatados). **Hallazgo real:** cuando el top-3 está así de cerca, el top-1 de `/predict/video` no es estable entre ejecuciones sobre el mismo archivo; `top3` es más confiable que `clase` sola como salida para este tipo de casos límite.

### `WS /predict/stream`
```
Cliente → {"frame": "<base64 JPEG>", "include_landmarks": false}

Servidor (mientras junta 30 frames) → {"status": "buffering", "frames_collected": 12, "frames_needed": 30}

Servidor (cada 30 frames) → {
  "clase": "DOS", "texto_castellano": "Dos", "confidence": 0.149,
  "latency_ms": 46.4, "top3": [...]
}

Servidor (error) → {"error": "frame inválido"}
```
Latencia real medida sobre este contrato (cliente WebSocket real, no simulado): p50=54.7ms, p95=58.7ms, max=118.2ms.

---

## 4. Reproducibilidad

| Ítem | Estado real |
|---|---|
| Entorno | `.venv310/` (Python 3.10.20 — torch, optuna; usado para entrenamiento e inferencia backend) y `.venv311/` (Python 3.11.15 — MediaPipe, Gradio) |
| Seeds | `SEED = 42` fijo en `scripts/train_s27.py` y en todos los splits (`StratifiedShuffleSplit`, `GroupShuffleSplit`). **No determinístico al 100%** — `DataLoader` usa `WeightedRandomSampler` y el augmentation usa `np.random` sin seed por worker; confirmado esta semana que dos corridas con la misma config (v3, v4) no reproducen bit-a-bit el mismo resultado. |
| Lockfile | ✅ `requirements.lock.txt` — generado con `pip freeze` sobre el `.venv310` real que corrió todos los tests y benchmarks de esta semana (198 paquetes con versión exacta). Es el entorno completo de desarrollo, no el mínimo de `requirements-api.txt` — para una imagen de producción más liviana, usar `requirements-api.txt` (ya usado por el `Dockerfile`) y congelarlo aparte cuando se pruebe el build de Docker (R5). |
| Makefile | No existía — se agrega `Makefile` en este entregable (ver raíz del repo) con los comandos ya validados esta semana. |

---

## 5. E2E en limpio (pasos + datos ejemplo + éxito)

```bash
# 1. Entorno
python3.10 -m venv .venv310 && .venv310/bin/pip install -r requirements-api.txt

# 2. Modelo (no está en git — copiar manualmente checkpoints/bilstm_s27.onnx
#    y data/s27_label2idx.json desde donde se entrenó, ver §10 riesgo R1)

# 3. Levantar backend
.venv310/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# 4. Verificar salud
curl http://localhost:8000/health
# Éxito esperado: {"status":"ok","model_ready":true,"n_classes":96}

# 5. Probar con dato de ejemplo real del repo
curl -F "file=@data/videos/original/Historias vinetas (11).mp4" \
     http://localhost:8000/predict/video
# Éxito esperado: JSON con "clase" dentro de las 96 etiquetas de data/s27_label2idx.json
```

**Criterio de éxito E2E:** `/health` responde `model_ready:true`, y `/predict/video` sobre cualquier archivo de `data/videos/original/*.mp4` devuelve una `clase` perteneciente al vocabulario (no error 422/503) en menos de 10 segundos para un video de ~1 minuto.

---

## 6. Observabilidad (logs/métricas)

**Lo que existe hoy:**
- `logs/runs.csv` — historial de entrenamientos (F1, ΔF1, PSI, KS, latencia por corrida). Es observabilidad de *entrenamiento*, no de *inferencia en producción*.
- `print()` a stdout en `api/main.py` y `demo/app_gradio.py` (carga de modelo, errores) — sin niveles de log, sin structured logging, sin destino persistente.
- Latencia por request: el campo `latency_ms` se devuelve en cada respuesta de `/predict/video` y `/predict/stream`, pero no se agrega ni se persiste — se pierde si no lo captura el cliente.

**Gap real para producción:** no hay métricas agregadas (p50/p95 en vivo, tasa de error, throughput), no hay alertas, no hay dashboard. Recomendación mínima antes de producción: reemplazar `print()` por `logging` estándar con nivel configurable, y agregar un contador simple de latencias (los benchmarks de esta semana ya muestran cómo medirlo — ver `SUSTENTACION_RESUMEN_FINAL.md` §objetivo latencia).

---

## 7. Validación & tests (smoke, golden)

**Implementado y verificado (`tests/`, 6/6 pasan):**
- `tests/test_smoke.py` — `/health` responde `model_ready:true` + `n_classes:96`; `/classes` devuelve el vocabulario completo; `/predict/video` respeta el contrato de campos de §3.
- `tests/test_golden.py` — 3 clips reales de una sola seña (AHORA, TÚ, PROTEÍNA) verificados en `top3`. **Elegidos con evidencia, no al azar**: se probó primero con 7 videos de viñeta completa y solo 1/7 acertó en `top3` — el muestreo uniforme a 30 frames sobre un clip de varios minutos pierde demasiado contexto. Los clips cortos (la tarea real para la que el modelo fue medido, F1=0.4426) sí son representativos. Este archivo también documenta un bug real de normalización Unicode encontrado al escribirlo (tildes en forma decompuesta en `s27_label2idx.json` — ver comentarios en el archivo).
- `tests/test_websocket_contract.py` — protocolo de `/predict/stream`: `status:buffering` en frames 1-29, predicción completa en el frame 30, y manejo de frame inválido.

Correr con: `.venv310/bin/python -m pytest tests/ -v` (requiere `pytest`, agregado a `requirements.lock.txt`).

**Gap restante:** sin CI configurado (no corre automáticamente en cada cambio) — recomendado antes de producción: GitHub Actions o similar ejecutando `pytest tests/` en cada push.

---

## 8. Seguridad & config (.env, límites)

| Ítem | Estado real | Riesgo |
|---|---|---|
| CORS | `allow_origins=["*"]` en `api/main.py` | Alto para producción pública — abierto a cualquier origen |
| Secretos/config | No existe `.env` ni gestión de secretos — todo hardcodeado en `CONFIG` dict | Bajo hoy (no hay credenciales que proteger), pero no escala |
| Límites de tamaño de archivo | `/predict/video` no valida tamaño máximo de upload | Medio — un archivo muy grande puede agotar memoria/tiempo |
| Rate limiting | No existe | Medio — sin protección ante abuso/DoS básico |
| Autenticación | No existe — todos los endpoints son públicos | Alto si se expone fuera de una red controlada |

**Recomendación antes de producción real:** restringir `allow_origins` a dominios conocidos, agregar límite de tamaño en `UploadFile`, y decidir si el caso de uso (institución educativa/salud) requiere autenticación básica (API key) antes de exponer el backend fuera de la red local.

**Confirmado por lectura real del archivo (no de memoria):** `notebooks/ENTREGABLE_PLAN_DE_DESPLIEGUE_S13.ipynb` §8 extrae `allow_origins` de `api/main.py` con una expresión regular sobre el código fuente real → `["*"]`. También confirma en vivo que `.env` no existe en el repo y que no hay ninguna cadena `max_size`/`MAX_UPLOAD` en `api/main.py`.

---

## 9. Hoja de ruta a Docker/API (tareas, responsables, fechas)

| Tarea | Responsable | Fecha objetivo | Estado |
|---|---|---|---|
| `tests/` con smoke + golden + contrato WebSocket (§7) | Juan Calla | — | ✅ Hecho — 6/6 tests pasan |
| `requirements.lock.txt` congelado (§4) | Juan Calla | — | ✅ Hecho |
| Portar a `api/main.py` los 3 ajustes de esta semana (segmentación, filtro narrativas, umbral 0.20) — ver R9 | Juan Calla | Antes de integrar el backend con un cliente real | ⚠️ Pendiente — encontrado al revisar este plan |
| Build de Docker real (`docker build` + `docker run`) | Juan Calla | Próxima sesión con Docker disponible | ⚠️ Pendiente — bloqueado por falta de Docker en este entorno |
| Commitear checkpoint a git (`checkpoints/bilstm_s27.onnx`) | Juan Calla | Antes del próximo despliegue | ⚠️ Pendiente — decisión explícita ya solicitada |
| Congelar `requirements-api.txt` (mínimo de producción) a versiones exactas | Juan Calla | Antes de producción | Pendiente |
| Logging estructurado en `api/main.py` (§6) | Juan Calla | 1 semana | Pendiente |
| Restringir CORS y límite de upload (§8) | Juan Calla | Antes de exponer fuera de red local | Pendiente |
| CI (correr `pytest tests/` en cada push) | Juan Calla | 1 semana | Pendiente |

---

## 10. Riesgos & mitigaciones (incluye rollback)

Todos los riesgos abajo son **reales, encontrados esta semana durante el desarrollo**, no hipotéticos.

| Riesgo | Encontrado | Mitigación / Rollback |
|---|---|---|
| **R1 — Checkpoint no versionado en git** | `checkpoints/*.onnx` excluido por `.gitignore` — el repo no es autosuficiente en un clon limpio | Commitear el `.onnx` (3.9MB, chico) con excepción explícita al `.gitignore`, o publicarlo como artefacto versionado aparte (release de GitHub, storage externo) |
| **R2 — Pérdida de checkpoint por sobrescritura** | El checkpoint de la corrida v3 (mejor KS, D=0.049) se perdió al sobrescribirlo la corrida v4 — solo quedó la métrica en `logs/runs.csv`, no los pesos | Versionar checkpoints por nombre de corrida (`bilstm_s27_v4.onnx`), no siempre el mismo nombre fijo |
| **R3 — Tiempo de entrenamiento impredecible** | Una corrida con `--skip-hpo` tardó 788 min (~13h) vs. el rango histórico de 66-120 min — causa no confirmada | No reentrenar cerca de fechas límite sin margen de +10× sobre el tiempo esperado; considerar entrenar en una máquina dedicada sin otros procesos compitiendo por CPU |
| **R4 — Umbral de confianza mal calibrado** | `CONF_UMBRAL=0.30` ocultaba predicciones correctas del modelo (confianza real de aciertos: media 0.55, pero con casos correctos desde 0.21) | Ya corregido a 0.20 esta semana, calibrado con 15 ejemplos reales — recalibrar si cambia el modelo |
| **R5 — Docker sin probar** | Sin acceso a Docker en el entorno de desarrollo de esta semana | Probar `docker build` en la primera oportunidad con Docker disponible, antes de cualquier despliegue real en contenedor |
| **R6 — Cambios de "optimización" con regresión silenciosa** | `model_complexity=0` en MediaPipe (para velocidad) degradó detección de mano derecha (3/10→1/10 frames) y dejó de mostrar traducciones — revertido tras detectarlo | Cualquier cambio de "performance" debe validarse contra los mismos videos de referencia antes de aceptarse, no solo medirse en velocidad |
| **R7 — CORS abierto** | `allow_origins=["*"]` — configuración de desarrollo | Restringir antes de exponer el backend fuera de una red controlada (§8) |
| **R8 — Sin reconocimiento continuo** | El sistema no traduce narración fluida sin pausas (verificado contra SRT real) — limitación de capacidad, no de infraestructura | Fuera de alcance de este plan de despliegue; documentado en `ROADMAP_FRONTEND_Y_EVALUACION.md` como investigación aparte |
| **R9 — `api/main.py` no tiene las mejoras de esta semana** | Encontrado al escribir este plan: la segmentación por pausas, el filtro de clases narrativas y el umbral de confianza calibrado (0.20) solo se aplicaron en `demo/app_gradio.py` y `spaces/app.py`. El WebSocket de `api/main.py` sigue con ventana fija de 30 frames y `confidence_threshold` como config muerta (nunca se usa para filtrar) | Portar los 3 ajustes a `api/main.py` antes de usarlo como backend de integración real — no asumir que "backend validado" (§2) significa "misma calidad que la demo"; hoy solo significa "no se cae" |
| **R10 — Videos largos vía API rinden peor de lo esperado** | Al armar el golden set: 7 videos de viñeta completa probados contra `/predict/video`, solo 1/7 acertó en `top3` (14%) — peor que el F1=0.4426 general, porque el muestreo uniforme a 30 frames sobre un video de varios minutos pierde casi todo el contenido intermedio | No usar `/predict/video` como demo principal con videos largos — funciona mucho mejor con clips cortos de una sola seña (3/4 aciertos en la prueba equivalente), que es además la tarea real para la que se midió F1. Documentado en `tests/test_golden.py`. |

**Rollback general:** todos los cambios de esta semana están en el working tree sin commitear a `main`/rama estable — `git diff`/`git checkout` revierte cualquier cambio de código individualmente. El modelo activo (v4) es el único artefacto no versionado (ver R1) — su "rollback" es simplemente no sobrescribir el archivo local hasta decidir qué versión conservar.

**R1 y R2 confirmados por comandos reales (no de memoria):** `notebooks/ENTREGABLE_PLAN_DE_DESPLIEGUE_S13.ipynb` §10 corre `git check-ignore -v checkpoints/bilstm_s27.onnx` en vivo → confirma `.gitignore:23:checkpoints/*.onnx`. El mismo notebook lista los archivos `bilstm_s27.*` en disco → solo existen `bilstm_s27.onnx` y `bilstm_s27.pt` (v4); ningún rastro de v1/v2/v3, confirmando que no son recuperables sin reentrenar.
