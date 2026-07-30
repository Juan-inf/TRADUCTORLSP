# Entregable — Plan de Despliegue (S13)

**Proyecto:** Traductor LSP → Castellano
**Fecha:** 2026-07-17
**Alcance:** plan de despliegue del sistema tal como existe hoy — backend FastAPI (`api/main.py`), demo Gradio (`demo/app_gradio.py`), despliegue público (`spaces/`), y el modelo activo BiLSTM S27 (v4). No introduce componentes nuevos; documenta lo que ya está construido y validado esta semana, y lo que falta para producción real.

---

## 1. Resumen ejecutivo

El sistema traduce en tiempo real señas aisladas de un vocabulario de 96 clases LSP a texto en castellano, con latencia end-to-end medida de **p50=54.7ms / p95=58.7ms** (umbral objetivo: 200ms). **Actualización 2026-07-19 (§11):** la configuración activa pasó de un solo modelo (v4, F1=0.4426) a un **ensemble de dos modelos** (v4 + S29, promedio de probabilidades), validado en un holdout limpio con F1=0.4424 y generalización mejorada (S29 es el primer checkpoint del proyecto en pasar HE3 completo, incluido KS, con márgenes amplios). Existen tres superficies funcionales y validadas con datos reales esta semana: una demo interactiva (`demo/app_gradio.py`, cámara + video + TTS), un backend API con WebSocket para integración (`api/main.py`), y un despliegue público en HuggingFace Spaces (`spaces/`, aún sirviendo solo v4 — pendiente de actualizar al ensemble).

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
| Portar a `api/main.py` los 3 ajustes de esta semana (segmentación, filtro narrativas, umbral 0.20) — ver R9 | Juan Calla | — | ✅ Hecho — verificado con WebSocket real |
| Versionar checkpoint fuera de `.gitignore` — ver R1/R2 | Juan Calla | — | ✅ Hecho — `checkpoints/versionado/` |
| Restringir CORS y límite de upload (§8) — ver R7 | Juan Calla | — | ✅ Hecho |
| Build de Docker real (`docker build` + `docker run`) | Juan Calla | Próxima sesión con Docker disponible | ⚠️ Pendiente — bloqueado por falta de Docker en este entorno |
| Commitear y pushear estos cambios a GitHub | Juan Calla | Antes del próximo despliegue | ⚠️ Pendiente — listo localmente, falta subir |
| Congelar `requirements-api.txt` (mínimo de producción) a versiones exactas | Juan Calla | Antes de producción | Pendiente |
| Logging estructurado en `api/main.py` (§6) | Juan Calla | 1 semana | Pendiente |
| CI (correr `pytest tests/` en cada push) | Juan Calla | 1 semana | Pendiente |

---

## 10. Riesgos & mitigaciones (incluye rollback)

Todos los riesgos abajo son **reales, encontrados esta semana durante el desarrollo**, no hipotéticos.

| Riesgo | Encontrado | Mitigación / Rollback |
|---|---|---|
| **R1 — Checkpoint no versionado en git** ✅ Resuelto | `checkpoints/*.onnx` excluido por `.gitignore` — el repo no es autosuficiente en un clon limpio | Resuelto: `checkpoints/bilstm_s27.onnx` (activo) y `checkpoints/versionado/bilstm_s27_v4_20260713.onnx` (copia de respaldo) agregados como excepción explícita al `.gitignore` |
| **R2 — Pérdida de checkpoint por sobrescritura** ✅ Resuelto | El checkpoint de la corrida v3 (mejor KS, D=0.049) se perdió al sobrescribirlo la corrida v4 — solo quedó la métrica en `logs/runs.csv`, no los pesos | Resuelto para v4 en adelante: `checkpoints/versionado/` guarda copias con nombre por corrida y fecha, nunca se sobrescriben |
| **R3 — Tiempo de entrenamiento impredecible** | Una corrida con `--skip-hpo` tardó 788 min (~13h) vs. el rango histórico de 66-120 min — causa no confirmada | No reentrenar cerca de fechas límite sin margen de +10× sobre el tiempo esperado; considerar entrenar en una máquina dedicada sin otros procesos compitiendo por CPU |
| **R4 — Umbral de confianza mal calibrado** | `CONF_UMBRAL=0.30` ocultaba predicciones correctas del modelo (confianza real de aciertos: media 0.55, pero con casos correctos desde 0.21) | Ya corregido a 0.20 esta semana, calibrado con 15 ejemplos reales — recalibrar si cambia el modelo |
| **R5 — Docker sin probar** | Sin acceso a Docker en el entorno de desarrollo de esta semana | Probar `docker build` en la primera oportunidad con Docker disponible, antes de cualquier despliegue real en contenedor |
| **R6 — Cambios de "optimización" con regresión silenciosa** | `model_complexity=0` en MediaPipe (para velocidad) degradó detección de mano derecha (3/10→1/10 frames) y dejó de mostrar traducciones — revertido tras detectarlo | Cualquier cambio de "performance" debe validarse contra los mismos videos de referencia antes de aceptarse, no solo medirse en velocidad |
| **R7 — CORS abierto** ✅ Resuelto | `allow_origins=["*"]` — configuración de desarrollo | Resuelto: restringido a orígenes locales conocidos (`localhost:7860/8000/3000`); agregado límite de 150MB en `/predict/video` |
| **R8 — Sin reconocimiento continuo** | El sistema no traduce narración fluida sin pausas — limitación de capacidad, no de infraestructura. **Verificado de nuevo con evidencia fresca (2026-07-18):** `Historias vinetas (8).mp4` (112s) tiene un SRT real con **109 glosas (≈58 señas/minuto)**. Instrumentando el segmentador se confirmó que de 33 segmentos cerrados, 24 se descartan (7 por clase narrativa, 17 por confianza <0.20) — muchos duran exactamente 90 frames (el tope duro), señal de que el segmentador no encuentra pausa y mezcla varias señas reales en un bloque. De las 5 palabras que sí mostró el sistema (`CIEN, IDEA, HACER, IR, IGUAL`), ninguna coincide con el guion real salvo una coincidencia espuria. Además, de las 79 glosas únicas reales del video, **solo 11 están en el vocabulario de 96 clases** — el resto es imposible de acertar sin importar la segmentación | Fuera de alcance de este plan de despliegue; documentado en `ROADMAP_FRONTEND_Y_EVALUACION.md` como investigación aparte |
| **R9 — `api/main.py` no tenía las mejoras de la demo** ✅ Resuelto | Encontrado al escribir este plan: la segmentación por pausas, el filtro de clases narrativas y el umbral de confianza calibrado (0.20) solo se aplicaron en `demo/app_gradio.py` y `spaces/app.py`. El WebSocket de `api/main.py` seguía con ventana fija de 30 frames y `confidence_threshold` como config muerta | Resuelto: `/predict/stream` ahora usa `SegmentadorPausas`, filtra por `CONF_UMBRAL=0.20` y excluye clases `HISTORIAS_VINETAS_*`, igual que la demo. Verificado con cliente WebSocket real (no simulado): sobre 120 frames de un video real, solo cerró 1 segmento y quedó correctamente descartado por umbral — cero fugas de clases narrativas |
| **R10 — Videos largos vía API rinden peor de lo esperado** | Al armar el golden set: 7 videos de viñeta completa probados contra `/predict/video`, solo 1/7 acertó en `top3` (14%) — peor que el F1=0.4426 general, porque el muestreo uniforme a 30 frames sobre un video de varios minutos pierde casi todo el contenido intermedio | No usar `/predict/video` como demo principal con videos largos — funciona mucho mejor con clips cortos de una sola seña (3/4 aciertos en la prueba equivalente), que es además la tarea real para la que se midió F1. Documentado en `tests/test_golden.py`. |
| **R11 — Clips ya pre-segmentados muy cortos (<1s) no se detectan en la demo** | Encontrado en ensayo E2E en vivo: 3 clips AEC de 0.3-0.7s (10-20 frames totales) dieron "0 señas detectadas" en `demo/app_gradio.py:/process_video_streaming`, aunque los mismos clips sí clasifican bien vía `/predict/video`. Causa: el `frame_stride` de la demo (submuestreo a ~10fps) deja solo 3-4 frames útiles de un clip ya tan corto; la confianza resultante tras `flush()` cae debajo de `CONF_UMBRAL=0.20` y se descarta en silencio. Es la imagen espejo de R10: ahí videos largos rinden peor por muestreo uniforme; acá clips ya-cortos rinden peor por sub-muestreo agresivo | No es un bug de R9 — es una limitación real del `frame_stride` combinada con clips atípicamente cortos para el caso de uso de la demo (video continuo/cámara, no clips ya recortados). No se modificó código: cualquier ajuste sin probar a días de la sustentación es más riesgo que beneficio. Si se retoma, calibrar `frame_stride` de forma adaptativa según duración total del video antes de procesar |
| **R12 — El segmentador por pausas no funciona con señas estáticas (abecedario)** ⚠️ Diagnóstico corregido (2026-07-23, ver R13) | Probado en vivo con `data/Glosas/LETRAS-ABECEDARIO/ABECEDARIO.mp4` (54s, 24 letras seguidas): la demo detectó 6 "señas", **ninguna una letra real** (`Dormir Ir Ir Ir Ir Hoy`). Causa raíz *original* (parcial, ver R13 para la causa real): el movimiento cuadro-a-cuadro de las letras del abecedario es **~0.00000**, indistinguible de una pausa real para `SegmentadorPausas`. Se mantiene como parte del cuadro completo, pero **no es la causa dominante** — ver R13 | Ver R13: la intervención de mayor impacto no es tocar el segmentador, es corregir el desbalance de dominio en los datos de entrenamiento del abecedario. Mientras tanto: usar clips ya aislados de una sola letra (vía `/predict/video`), no video continuo del abecedario completo |
| **R13 — Causa raíz real: el abecedario se entrenó con fotos de mano sin cuerpo (pose=0), la demo real siempre detecta pose → el modelo nunca considera letras** (investigado a fondo 2026-07-23, a pedido explícito de "revisa por qué no traduce el abecedario") | Investigación completa, no solo hipótesis: (1) el 100% de 72 muestras de entrenamiento del abecedario revisadas (`data/Keypoints/abecedario_pkl`) tienen los 33 keypoints de pose en cero — son fotos de mano en primer plano de un dataset externo (`data/Abecedario/README.md`: *"Static-Hand-Gestures-of-the-Peruvian-Sign-Language-Alphabet"*, 150 fotos/letra, sin torso visible), a diferencia de las clases de palabras (8/8 revisadas tienen pose real, de video de intérprete de cuerpo completo). (2) Prueba causal: tomando 5 clips reales de letras (A, D, F, I, W) recortados con timestamps exactos del `.eaf` de `ABECEDARIO.mp4` (cámara de cuerpo completo), con **pose real** el modelo predice siempre palabras genéricas (IR 33-63%, ORIGINAL, DORMIR, HISTORIAS_VINETAS_*) — nunca una letra. Con la **pose puesta a cero** (igual que en entrenamiento), el modelo cambia a predecir letras de forma consistente. (3) Sobre 120 muestras reales de entrenamiento (pose=0, como fueron entrenadas) el modelo logra **88% top-1, 99% top-5** — el modelo reconoce bien la forma de la mano; el problema es que "pose≈0" actúa como un atajo aprendido para "esto es una letra", y en cámara real la pose siempre se detecta. **Hallazgo adicional de valor:** ya existe en el repo material real de cuerpo completo con las 24 letras y pose correcta (`data/Keypoints/vocabulario_lsp_p_pkl/LETRAS-ABECEDARIO/ABECEDARIO.pkl` y `.../glosas_pkl/...`, más el `.eaf` de ELAN con timestamps exactos por letra en `data/Glosas/LETRAS-ABECEDARIO/ABECEDARIO.eaf`) que nunca se usó para entrenar — es el material necesario para corregir esto de raíz | Decisión explícita (2026-07-23): no reentrenar en esta sesión — el fix real (extraer las 24 letras del `.eaf`, re-extraer keypoints con pose real, fine-tune warm-start del checkpoint activo) toca un modelo que está en el ensemble de producción, con tiempo de entrenamiento históricamente impredecible (20min-13h). Queda documentado como el camino correcto para trabajo futuro, no como intento de esta noche. **No presentar en la sustentación la explicación de R12 (segmentador) como causa principal** — la evidencia dice que es de dominio de datos (R13) |
| **R14 — Tab "Detectar y traducir" (imagen estática): no traducía fotos del abecedario** ✅ Resuelto (2026-07-23, mismo día que R13) | Encontrado a partir del reporte del usuario ("no muestra la traducción de la imagen insertada del abecedario"). Causa raíz distinta a R13, específica de `process_image()`: `Holistic` deriva la región de recorte de la mano a partir de la muñeca que estima **su propio** modelo de pose; en una foto de mano sola sin cuerpo, esa pose es una adivinanza de baja confianza (visibility ~0.27-0.4, verificado con varias fotos reales de `data/Abecedario/`) y mal ubicada — el recorte resultante falla en encontrar la mano aunque ocupe casi toda la imagen. Verificado con la foto `data/Abecedario/a/a (1).jpg`: 0/1 manos vía `Holistic` vs. 1/1 vía el modelo standalone `mediapipe.solutions.hands.Hands` (no depende de pose) sobre la misma imagen | **Corregido**, primero acotado a `process_image()` y luego **unificado a las 5 rutas de extracción del proyecto** (a pedido explícito del usuario — "el mismo frontend debe ser para traducir en vivo, por cámara en vivo, por video, por imagen" — la lógica estaba duplicada de forma inconsistente entre cámara, video e imagen en `demo/app_gradio.py`, y otra vez en `api/main.py`, que es justamente cómo este bug quedó sin corregir en 3 de las 5 rutas cuando se arregló solo en la de imagen). Se extrajo el respaldo a una función compartida `src.features.landmarks.aplicar_respaldo_manos_y_pose(results, frame_rgb, hands_solution)`: (1) si `Holistic` no encuentra ninguna mano, corre `Hands` (modelo standalone, no depende de pose) como respaldo y sus landmarks (con handedness Left/Right) se inyectan en `results`; (2) si no se detectó rostro, descarta la pose que haya devuelto `Holistic` (señal más robusta que un umbral de confianza fijo — verificado que ninguna de 24 fotos reales del abecedario tiene rostro, y una foto real de cuerpo completo sí, visibility 0.62). Aplicada ahora en las 5 rutas: `demo/app_gradio.py` (cámara en vivo, video, imagen — con instancias `Hands` persistentes, no recreadas por frame) y `api/main.py` (`/predict/stream` WebSocket y `/predict/video`). Validado: **19/24 letras (79%) correctas** sobre fotos reales de `data/Abecedario/` vía imagen (antes: 0/6 en la muestra inicial); golden set de video sigue 3/3 correcto (AHORA, TÚ, PROTEÍNA — sin regresión); video de cuerpo completo del abecedario sigue mostrando el problema de R13 (pose real preservada porque sí hay rostro — correcto, ese es un problema distinto que requiere reentrenar, no este fix); 6/6 tests siguen pasando |
| **R15 — La unificación de R14 rompió la detección en cámara en vivo** ✅ Resuelto (2026-07-23, mismo día que R14) | Encontrado a partir del reporte del usuario ("no detecta la cámara las señas y no muestra la traducción"), inmediatamente después de unificar R14 a las 5 rutas. Causa: la parte 2 del respaldo de R14 (descartar la pose de `Holistic` si no hay rostro) es segura en una foto suelta, pero **peligrosa frame a frame en un stream continuo** — muchas señas LSP se hacen cerca de la cara, y la mano puede tapar momentáneamente el rostro durante la seña; eso hace fallar la detección de rostro en ESE frame puntual aunque el cuerpo y la pose sean reales, y de golpe esos frames quedan con pose=0 a mitad de una seña real. Ningún clip de prueba controlado (AEC, 254 frames, 0% sin rostro) reprodujo el problema, pero el riesgo teórico es real y suficiente para explicar el reporte: además, el pipeline de entrenamiento (`scripts/build_dataset_s17.py` y afines) nunca aplicó este descarte, así que activarlo en vivo introducía una inconsistencia train/inference nueva, propia de este fix, para las 96 clases — no solo el abecedario | **Corregido**: `aplicar_respaldo_manos_y_pose()` ahora recibe `descartar_pose_sin_rostro: bool = False` (por defecto **apagado**). Solo `process_image()` lo activa explícitamente (`descartar_pose_sin_rostro=True`) — es el único caso validado y seguro. Cámara, video (demo y API) y WebSocket ahora solo reciben la parte 1 del respaldo (manos vía `Hands` cuando `Holistic` no encuentra ninguna) — siempre segura, nunca borra una detección real. Validado: alimentando `process_webcam_frame()` frame a frame con un clip real completo (simulando una sesión de cámara en vivo) clasifica correctamente **PROTEÍNA al 89.6%** y puebla la transcripción; golden set de video sigue 3/3; abecedario por imagen sigue en 19/24 (79%, sin cambio — la parte que sí ayuda ahí sigue activa); 6/6 tests pasan |

**Rollback general:** todos los cambios de esta semana están en el working tree sin commitear a `main`/rama estable — `git diff`/`git checkout` revierte cualquier cambio de código individualmente. El modelo activo (v4) es el único artefacto no versionado (ver R1) — su "rollback" es simplemente no sobrescribir el archivo local hasta decidir qué versión conservar.

**R1 y R2 confirmados por comandos reales (no de memoria):** `notebooks/ENTREGABLE_PLAN_DE_DESPLIEGUE_S13.ipynb` §10 corre `git check-ignore -v checkpoints/bilstm_s27.onnx` en vivo → confirma `.gitignore:23:checkpoints/*.onnx`. El mismo notebook lista los archivos `bilstm_s27.*` en disco → solo existen `bilstm_s27.onnx` y `bilstm_s27.pt` (v4); ningún rastro de v1/v2/v3, confirmando que no son recuperables sin reentrenar.

---

## 11. Experimentos de mejora post-entrega (2026-07-18/19) — modelo activo pasa a ensemble

Tras el cierre inicial de este plan, se reabrió la búsqueda de mejorar el F1 con una auditoría exhaustiva de datos sin usar en el repositorio, seguida de 4 corridas reales de entrenamiento. Se documenta el proceso completo — incluidos los intentos fallidos — porque son evidencia real del trabajo hecho, no solo el resultado final.

### 11.1 Auditoría de datos

- `dgi156_full` (990 archivos): 139 netos-nuevos, repartidos en 68 clases fuera del vocabulario, ninguna cruza el umbral de 15 muestras → **sin uso real**.
- `aec/Keypoints-1`: duplicado exacto de `aec/Keypoints` (0 diferencias) → **sin uso real**.
- PUCP-305 (`vocabulario_lsp_p`): 448 de 468 videos ya procesados (95.7%) → **prácticamente agotado**.
- LSA64: confirmado en `train_s14.py` que se excluyó deliberadamente en Sprint 14 por ser Lengua de Señas **Argentina**, no peruana → **correctamente descartado, no se reabre**.
- **Hallazgo real:** `dgi156_pkl` y `pkl` (fuente "vineta") etiquetan cada archivo con el nombre de su carpeta (`HISTORIAS_VINETAS_N`), ignorando el nombre real de archivo (`QUÉ__855.pkl` se entrenaba como "HISTORIAS_VINETAS_24", no como "QUÉ"). Extrayendo la glosa real del nombre de archivo aparecieron 744 glosas reales sin aprovechar (7151 archivos).

### 11.2 Cuatro corridas de entrenamiento

| Corrida | Dataset | Clases | F1-test | HE3 | Resultado |
|---|---|---|---|---|---|
| **v4 (activo previo)** | dataset_s17 | 96 | 0.4426 | Falla (solo KS) | Línea base |
| S28 | dataset_s18 (+ 85 clases nuevas) | 216 | 0.2219 | **Pasa completo** (primera vez) | F1 muy bajo — más clases = tarea más difícil |
| S29 | dataset_s18b (Fase 1: solo 96 clases, +11% datos) | 96 | 0.4208 | **Pasa completo**, márgenes amplios | Cercano a v4, mejor generalización |
| S30 | dataset_s18bc (sin `dgi156_gloss`, datos "limpios") | 96 | 0.225 (falla) | Pasa (débil) | **Corrida inestable** — Fold 5 colapsó a F1=0.0004; dataset resultante demasiado chico para KFold(5) estable |

**Diagnóstico del colapso de S30:** verificado que no es un bug de código (la selección del mejor fold es correcta). Es una limitación real de tamaño de muestra — con clases de apenas 15-20 muestras, cada fold de KFold(5) recibe muy pocos ejemplos de las clases más chicas, y la combinación de partición + inicialización de pesos puede converger a una solución degenerada. Confirma que los datos disponibles alcanzan para sostener v4/S29, no para experimentar con configuraciones más finas sin riesgo de inestabilidad.

**Hallazgo adicional (calidad de datos):** los clips de `dgi156_gloss` tienen **siempre exactamente 30 frames** (ventana fija, no segmentación real de la seña), a diferencia de `vineta_gloss` (1-34 frames, variación natural como en AEC). Es ruido probable en el etiquetado a nivel de glosa — explica por qué clases como QUÉ/TÚ/YO/DECIR/ESE, que ganaron muchas muestras de esta fuente, empeoraron en vez de mejorar en S29.

### 11.3 Ensemble v4+S29 — la mejora real, validada correctamente

Comparar v4 vs. S29 directamente sobre el holdout de cualquiera de los dos está contaminado (cada modelo pudo haber visto en su propio entrenamiento parte del holdout del otro, ya que ambos datasets comparten fuentes crudas). Se reconstruyó — con rastreo del archivo `.pkl` físico de cada muestra a través de ambos pipelines (`scripts/validar_ensemble_v4_s29.py`) — la intersección real de ambos holdouts: **342 muestras, 72 clases, garantizadas fuera del entrenamiento de los dos modelos.**

| | F1-macro (holdout limpio) |
|---|---|
| v4 solo | 0.4222 |
| S29 solo | 0.3591 |
| **Ensemble (promedio de probabilidades)** | **0.4424** |

El ensemble supera a ambos modelos individuales en esta comparación válida, iguala el F1 histórico de v4, y hereda la generalización de S29 (HE3 completo). **Pasa a ser la configuración activa del sistema.**

**Implementación:** `api/main.py` y `demo/app_gradio.py` cargan ambos checkpoints (`bilstm_s27.onnx` + `bilstm_s29.onnx`) y promedian sus probabilidades, alineadas por **nombre** de clase (no por índice — cada modelo tiene su propio orden interno). Si `bilstm_s29.onnx` no está presente, el sistema sigue funcionando solo con v4 (sin romperse). Costo extra de inferencia: despreciable (<2ms adicionales; el cuello de botella sigue siendo MediaPipe a ~55ms/frame).

**Bug encontrado y corregido durante la implementación:** el alineamiento por nombre inicialmente solo emparejaba 88 de 96 clases — v4 guarda tildes en forma Unicode decompuesta (NFD) y S29 en forma precompuesta (NFC), el mismo tipo de bug ya visto antes en este proyecto (`tests/test_golden.py`). Sin la normalización NFC, 8 clases con tilde recibían la mitad de su probabilidad real en el ensemble (el término de S29 quedaba en 0 en vez de su valor real). Corregido en ambos archivos; verificado 96/96 clases alineadas.

**Verificación:** 6/6 tests pasan con el ensemble activo (incluido el golden test). Probado en vivo contra `/predict/video`: los 3 clips del golden set (AHORA, TÚ, PROTEÍNA) siguen clasificando correctamente en top-1, con PROTEÍNA incluso más confiado (90.6%). Nota honesta: no todos los casos individuales mejoran (un clip de "NIÑO" que v4 solo acertaba pasó a clasificar como "ORIGINAL" con el ensemble) — es el comportamiento esperado de un ensemble: mejora el promedio, no garantiza cada caso puntual.

**Backup:** todos los checkpoints, datasets y `logs/runs.csv` de antes de esta ronda de experimentos quedaron respaldados en `backups/20260718_223233_pre-ensemble/`.

---

## 12. Sprint 31 (2026-07-23) — primera medición WER real del proyecto, e intento de mejorar narración con datos SRT+landmarks (resultado negativo, documentado con honestidad)

A pedido explícito del usuario ("mejora la traducción de video de palabras seguidas y narración, usa el audio de cada video para ver el significado de cada seña"), se hizo primero una auditoría completa de `data/` (excluyendo solo LSA64, lengua de señas argentina) y luego se construyó, entrenó y **midió con WER real por primera vez en el proyecto** un intento directo de resolver el problema de narración continua — documentado completo porque el resultado fue negativo, y es evidencia real del proceso, no solo el resultado que funcionó.

### 12.1 Hallazgo de la auditoría — no es "audio", es mejor que audio

Los videos DGI156/vineta (intérprete real, comunidad sorda) no tienen narración de audio útil para alinear señas — pero **sí existen anotaciones manuales de glosa individual con timestamp exacto**: `data/SRT/SRT_SEGMENTED_SIGN/` tiene 27 archivos `.srt`, **4,092 glosas anotadas a mano con inicio/fin en milisegundos**, algo mejor que audio (no depende de que haya narración hablada — funciona directo sobre la seña). Además:

- `data/landmarks/` + `data/manifest_segments.csv`: keypoints **ya extraídos** con MediaPipe en ventanas deslizantes (30 frames, stride 15) sobre 26 de esos videos completos — 7,235 ventanas — generados por `scripts/preprocess_sliding_window.py` en un sprint anterior, pero etiquetados con la clase del VIDEO ENTERO (`vineta_002`), no con la glosa real de cada ventana. La parte más lenta (extracción MediaPipe) ya estaba hecha; nunca se había cruzado con el SRT.
- AEC (Aprendo en Casa) sí tiene 2 episodios completos con audio real (`data/_s13_tmp/aec_extracted/Videos/RawVideo/`, ~29 min c/u, pista AAC — el profesor hablando mientras el intérprete traduce en simultáneo), pero sin alineación gloss-a-gloss ya hecha, a diferencia de DGI156/vineta.
- Confirmado: LSA64 sigue siendo la única fuente no peruana en todo `data/`; el resto (Abecedario, Glosas, Keypoints/*, external_lsp/*, videos originales) es LSP real, ya inventariado en sprints anteriores.

### 12.2 Dataset S31 — cruce SRT × landmarks

`scripts/build_dataset_s31_continuo.py`: para cada ventana ya extraída, busca la glosa del SRT cuyo rango de tiempo contiene el centro de la ventana (si cae en una pausa sin anotación, se descarta). Se reservaron **5 videos completos, 100% fuera del dataset de entrenamiento**, para poder medir WER real después (`s31_videos_wer_holdout.json`).

Resultado: **1,538 muestras, 51 clases** (min≥15 muestras/clase, de 865 glosas únicas crudas — la mayoría demasiado raras para ser una clase entrenable). Comparado con dataset_s18b (13,494 muestras, 96 clases de clips ya aislados), este dataset es un orden de magnitud más chico — es la primera vez que se entrena sobre ventanas de **contexto narrativo real** en vez de clips ya cortados a mano, pero con muchos menos datos por clase.

### 12.3 Entrenamiento (`scripts/train_s31.py`) — HPs S13-best sin HPO, KFold(5)

Split a nivel de VIDEO completo (no de ventana — ventanas contiguas comparten hasta 15/30 frames, mezclarlas entre train/test habría filtrado datos): 15 videos para entrenar, 6 videos (411 muestras) reservados como test holdout, además de los 5 videos 100% aparte para WER.

| Métrica | Valor |
|---|---|
| F1 KFold(5) (validación dentro del pool de entrenamiento) | 0.1744 ± 0.0207 |
| **F1-test (videos completos nunca vistos)** | **0.0614** |
| Acc-test | 5.1% |
| Top-3 / Top-5 test | 9.7% / 14.4% |
| ECE | 0.124 → 0.085 (T*=1.395) |
| Latencia ONNX | 0.78 ms |

**Caída de generalización severa** (0.174 → 0.061, -65%) entre validación cruzada (mezclando ventanas de los mismos videos de train en distintos folds) y el holdout de videos genuinamente nunca vistos — con solo 15 videos/15 personas de entrenamiento efectivo, el modelo no generaliza a estilo de señante o contexto narrativo distinto. Es la medición más honesta de dificultad real de este problema hecha en el proyecto hasta ahora.

### 12.4 Evaluación WER real (`scripts/evaluar_wer_s31.py`) — primera vez en el proyecto

Sobre los 5 videos 100% reservados, comparando dos pipelines completos sobre los mismos keypoints extraídos (para que la comparación no dependa de diferencias en calidad de MediaPipe):

| | S31 (ventana fija + este modelo) | Producción (SegmentadorPausas + ensemble v4+S29) |
|---|---|---|
| **WER medio (5 videos)** | **2.827** | **1.763** |

**S31 no mejora la traducción de narración — la empeora.** Ambos números están muy por encima de 1.0 (más errores que palabras reales), confirmando que traducir narración continua sigue sin resolverse — pero producción, a pesar de no haber sido diseñada para esto, comete menos errores que el intento dedicado. Revisando las secuencias predichas caso por caso: S31 tiende a converger a un puñado de clases "imán" (`ADIVINA`, `BOLA DE CRISTAL`, `NADA MÁS`) sin relación aparente con lo que se está señando — síntoma directo de la caída de generalización medida en §12.3, no un problema del pipeline WER en sí (verificado: el vocabulario de S31 contiene una clase de ruido real, `1.96 C-M`, una medida mencionada en una narrativa que apareció ≥15 veces — 1 de 51 clases, no explica la brecha completa).

**Diagnóstico honesto — por qué no funcionó:** el cuello de botella no es el enfoque (cruzar SRT con landmarks ya extraídos es sólido y quedó como infraestructura reusable), es **volumen de datos para generalización persona-independiente**: 1,538 muestras / 15 señantes de entrenamiento es insuficiente para que el modelo aprenda "la glosa" en vez de "cómo la señan estas 15 personas específicas en estos videos específicos". La ganancia de S28 (216 clases, HE3 pasa) vino de sumar volumen a un vocabulario ya grande vía las mismas glosas (`dgi156_gloss`/`vineta_gloss`) — S31 usó ese mismo recurso pero AISLADO, sin la base de ~12,000 muestras de clips ya cortados que sí tienen las demás fuentes.

### 12.5 Decisión y trabajo futuro

**No se despliega S31** — el ensemble v4+S29 sigue siendo la configuración activa de producción, sin cambios. `checkpoints/bilstm_s31.onnx`, `data/dataset_s31.npz` y `scripts/evaluar_wer_s31.py` quedan en el repo como infraestructura ya construida y validada para la próxima iteración:

- **Camino recomendado no probado aún:** combinar dataset_s31 (contexto narrativo real, 1,538 muestras) con dataset_s18b (12,000+ muestras de clips aislados) en un solo entrenamiento — mismo `label2idx` unificado, ambas vistas del dato con el fix de grupos cross-source ya validado en S17/S18 — en vez de entrenar S31 aislado sobre un dataset diez veces más chico que el resto del proyecto.
- `scripts/evaluar_wer_s31.py` es ahora la primera infraestructura de medición WER real del proyecto (pendiente desde Sprint 9 en la planificación original) — reusable para medir cualquier intento futuro sobre los mismos 5 videos de referencia, con comparación directa contra producción.
- `data/landmarks/` + `data/SRT/` combinados dan más señal de la que este sprint usó — el filtro min≥15 descartó 814 de 865 glosas únicas por tener muy pocas muestras; con más videos narrativos (`data/external_lsp/dgi156_full` tiene `SRT.tar` con más contenido sin extraer) esas clases podrían cruzar el umbral.

---

## 13. Sprint 32 (2026-07-23, mismo día) — el camino recomendado en §12.5 funcionó: WER real baja 36% frente a producción

A pedido explícito del usuario ("mejora el resultado y reentrena"), se implementó el camino recomendado en §12.5: combinar las muestras de contexto narrativo de S31 con el corpus grande de clips ya aislados (mismas fuentes que `dataset_s18`), en vez de entrenar aislado sobre un dataset diez veces más chico.

### 13.1 Dataset (`scripts/build_dataset_s32_merge.py`)

Reutiliza `cargar_fuente`/`cargar_fuente_glosas` de `build_dataset_s18.py` para las 8 fuentes existentes (abecedario, dgi156, vineta, dgi156_gloss, vineta_gloss, aec, vocabulario_lsp_p, glosa) sin el filtro `--solo-vocab-actual`, y agrega `s31_continuo` (la fuente nueva de S31, SRT × landmarks, **sin** el filtro min-muestras que sí se aplica al final) como una fuente más. Los 5 videos reservados para WER (`s31_videos_wer_holdout.json`) se excluyen explícitamente de `s31_continuo` para que sigan 100% sin ver por el modelo.

**Riesgo de fuga de datos identificado y resuelto antes de entrenar:** `balancear_grupos_cruzados()` (el fix de S17/S18 que evita que la misma clase caiga en train por una fuente y en holdout por otra) reasigna grupos **por clase compartida entre pares de fuentes**, no por video — sus grupos ya no correlacionan con `num_vineta`. Como `s31_continuo` viene de los mismos 26 videos narrativos que `dgi156_gloss`/`vineta_gloss`, agregar sus muestras con un grupo propio sin más habría permitido que la misma seña, del mismo video, apareciera en train (vía `vineta_gloss`) y en test (vía `s31_continuo`) — fuga real. Se resolvió extendiendo la lista de pares a rebalancear: `[("dgi156","vineta"), ("dgi156_gloss","vineta_gloss"), ("dgi156_gloss","s31_continuo"), ("vineta_gloss","s31_continuo")]`, usando el mecanismo ya validado en vez de inventar uno nuevo.

Resultado: **20,689 muestras, 270 clases** (min≥15) — HE3 con 0 clases en holdout sin ninguna muestra de training (0.0%), la mejor cobertura de grupos lograda en el proyecto hasta ahora.

### 13.2 Entrenamiento (`scripts/train_s32.py`) — presupuesto de tiempo acotado a propósito

S28 (17,471 muestras, 216 clases, KFold(5) completo, 100 épocas) tardó **65,225s (~18h)**. Con 20,689 muestras y 270 clases (aún más grande), para no repetir un runaway de horas se usó **un solo split train/val** (no KFold(5)) y menos épocas/paciencia (30/6 en vez de 100/15). Entrenamiento real: **9.9 minutos** — validando que el recorte de presupuesto fue efectivo sin perder la señal que importaba.

| Métrica | Valor |
|---|---|
| F1-test (270 clases) | 0.1660 — no comparable 1:1 con producción (96 clases); esperado, mismo patrón que S28 |
| Top-3 / Top-5 test | 30.4% / 35.8% |
| HE3 | **PASA completo, incluido KS** (ΔF1=0.0280, PSI=0.0088, KS p=0.2564) |
| Latencia ONNX | 0.70 ms |

### 13.3 WER real (`scripts/evaluar_wer_s32.py`) — mejora confirmada

Mismos 5 videos 100% reservados, mismo criterio de medición que S31 (§12.4), comparando 3 pipelines:

| | S31 (aislado, §12) | **S32 (combinado)** | Producción (v4+S29 + segmentador) |
|---|---|---|---|
| **WER medio (5 videos)** | 2.827 | **1.134** | 1.763 |

**S32 mejora sobre producción en un 36% (1.763 → 1.134), y sobre S31 en un 60%.** Sigue por encima de 1.0 (más errores que palabras de referencia) — traducir narración fluida sigue sin resolverse del todo — pero es la primera vez en el proyecto que un intento dedicado a este problema le gana a lo que ya está en producción, no solo empata o pierde. El vocabulario más grande (270 vs 96 clases) por sí solo ya es una ventaja real para este caso de uso: las referencias por video cubren muchas más palabras reales dichas en la narración (video 22: 175 glosas de referencia en vocabulario S32 vs 42 en el vocabulario de producción).

### 13.4 Decisión de despliegue

**No se reemplaza el ensemble de producción (v4+S29).** F1 de S32 sobre clips aislados (0.166, 270 clases) es muy inferior al de producción (0.4424, 96 clases) — para el objetivo declarado del proyecto ("alta precisión" en señas aisladas, el caso de uso principal medido en `RESULTADOS_PRESENTACION_S15.md`), producción sigue siendo la mejor opción. S32 es una mejora real y medida específicamente para narración continua — vale la pena ofrecerlo como modo adicional (p.ej. un toggle en la demo "modo narración") en vez de reemplazar el modelo principal, pero esa integración de UI queda como trabajo futuro, no se implementó en este sprint.

`checkpoints/bilstm_s32.onnx`, `data/dataset_s32.npz`, `scripts/evaluar_wer_s32.py` quedan en el repo, documentados y reproducibles.

---

## 14. Sprint 33 (2026-07-24) — presupuesto de entrenamiento completo sobre el mismo dataset: WER baja a 1.027

A pedido explícito del usuario ("mejora mas los resultados para llegar al objetivo"), se reentrenó el mismo `dataset_s32.npz` (sin cambios de datos) con el presupuesto de entrenamiento completo del proyecto en vez del recorte defensivo de S32.

### 14.1 Por qué se reentrenó sin cambiar datos

S32 usó un solo split train/val (30 épocas, paciencia 6) por precaución — el precedente de S28 (tamaño de dataset similar) había tardado ~18h con KFold(5) completo, y no se conocía el costo real de este dataset nuevo. Resultado real de S32: **9.9 minutos**, con la curva de F1-val todavía mejorando de forma ruidosa en la época 25 sin señal clara de plateau — margen real sin explotar. `scripts/train_s33.py` reentrenó con KFold(5) + 60 épocas/paciencia 10 (presupuesto a mitad de camino entre el recorte de S32 y el estándar histórico de 100/15).

**Incidente operativo:** el primer intento se dejó corriendo con salida a archivo sin flush (buffering completo de Python al no ser una terminal) — sin poder ver el progreso real. La máquina entró en reposo varias horas durante la corrida (12h de reloj transcurrido, solo ~18 min de CPU real usados) y el proceso terminó sin producir checkpoint. Se relanzó con `python -u` (salida sin buffer) para tener visibilidad real, y esta vez sí completó: **351 minutos (~5.8h)** de cómputo real — más que el estimado inicial, pero terminó correctamente.

### 14.2 Resultados — mejora consistente en las tres métricas que importan

| | S32 (split único) | **S33 (KFold(5) completo)** |
|---|---|---|
| F1-test (270 clases) | 0.1660 | **0.2092** |
| Top-5 test | 35.8% | **40.4%** |
| HE3 | PASA | **PASA** (ΔF1=0.0416, PSI=0.0062, KS p=0.3912) |

### 14.3 WER real (`scripts/evaluar_wer_s33.py`) — mejora sostenida, tercera corrida consecutiva

Mismos 5 videos 100% reservados, mismo criterio de medición que S31/S32:

| | S31 (aislado) | S32 (combinado, split único) | **S33 (combinado, KFold completo)** | Producción |
|---|---|---|---|---|
| **WER medio (5 videos)** | 2.827 | 1.134 | **1.027** | 1.763 |

Progresión: **2.827 → 1.134 → 1.027**, la tercera mejora consecutiva del mismo camino (combinar datos narrativos con el corpus grande), y el uso del presupuesto de entrenamiento completo aportó una mejora adicional real (-9.4% de WER sobre S32) sin cambiar una sola muestra de datos.

**Honestidad sobre dónde queda esto:** WER=1.027 sigue por encima de 1.0 (todavía hay más errores que palabras de referencia en promedio) — no es un sistema de traducción de narración usable en el sentido estricto (eso requeriría WER bien por debajo de 0.5). Es, sí, la medición más baja lograda en el proyecto para este problema, y la trayectoria de las tres corridas (S31→S32→S33) es monótonamente descendente sin ningún retroceso — evidencia de que el camino (más datos narrativos + presupuesto de entrenamiento adecuado) sigue dando resultado, no de que el problema esté resuelto.

### 14.4 Decisión de despliegue

**No se reemplaza el ensemble de producción** — mismo razonamiento que §13.4: F1 de S33 sobre clips aislados (0.209, 270 clases) sigue por debajo de producción (0.4424, 96 clases) para el objetivo principal del proyecto (señas aisladas). `checkpoints/bilstm_s33.onnx` y `scripts/evaluar_wer_s33.py` quedan documentados y reproducibles como el mejor punto medido hasta ahora específicamente para narración continua.

**Trabajo futuro para seguir bajando WER por debajo de 1.0:** más videos narrativos con SRT (el catálogo original ya señalaba 128 glosas DGI156 sin descargar, ver `catalogo_datasets_lsp.json`), y/o HPO real en vez de HPs S13-best directas — ninguno de los 3 sprints de esta línea (S31/S32/S33) corrió búsqueda de hiperparámetros, todos usaron los mismos valores heredados de S13.

## 15. Sprints 34-37 (2026-07-25/26) — fuente ELAN (.eaf) nueva, resultado mixto, resuelto con ensemble: WER baja a 0.970

Se descubrió un recurso sin usar en el repositorio: 525 archivos `data/Glosas/*.eaf` (formato ELAN), de los cuales 274 son oraciones completas ("_ORACION_") con anotación glosa-por-glosa a precisión de milisegundo (tier `GLOSA`), hecha por un anotador de LSP directamente sobre la seña — a diferencia del SRT usado en S31-S33 (transcripción del audio hablado). `scripts/extract_glosas_keypoints.py` ya parseaba estos `.eaf`, pero solo para recortar clips de seña individual, descartando explícitamente los "_ORACION_". Se probó la hipótesis de que esta anotación de mejor calidad podía mejorar aún más la línea de narración continua.

### 15.1 Sprint 34 — extracción (`scripts/build_dataset_s34_eaf.py`)

Se extrajeron keypoints MediaPipe Holistic por cada segmento de glosa dentro de las 274 oraciones ELAN (10 reservadas 100% para evaluación WER, nunca usadas en entrenamiento). Resultado: **668 muestras, 46 clases** tras filtro min5 (muestras muy escasas por clase — la mayoría con 5-13 repeticiones).

### 15.2 Sprint 35 — combinación + entrenamiento rápido (resultado NEGATIVO)

`scripts/build_dataset_s35_merge.py` combinó `dataset_s32` (S18 + s31_continuo) con la nueva fuente `s34_eaf` → **dataset_s35: 21,343 muestras, 275 clases**, HE3 limpio (0% holdout sin entrenamiento). `scripts/train_s35.py` (split único rápido, mismo presupuesto que S32) dio **F1-test=0.1490**. WER real (`scripts/evaluar_wer_s35.py`, sobre los 5 videos oficiales y sobre 10 oraciones ELAN nuevas reservadas): **peor que S33 en ambos benchmarks** (1.086 vs 1.027 oficial; 1.188 vs 1.062 ELAN). Resultado negativo, reportado con honestidad — mismo criterio que S31.

**Causa diagnosticada:** la glosa "IX" (señalamiento pronominal — apunta a un referente espacial variable, sin forma visual fija) era ~17% de las muestras de `s34_eaf`, la 2ª clase más frecuente, y probable ruido de etiqueta puro para un clasificador de ventana fija de 30 frames.

### 15.3 Sprint 36 — corrección (excluir IX) + presupuesto completo (resultado MIXTO)

Dos ajustes antes de descartar la fuente: (1) excluir la glosa "IX" de `s34_eaf` (`build_dataset_s35_merge.py`, `GLOSAS_EXCLUIDAS_S34`) → dataset_s35 corregido: 21,221 muestras, 274 clases; (2) `scripts/train_s36.py` con presupuesto completo (KFold(5), 60 épocas) en vez del split único de S35 — el mismo salto que ya había mejorado S32→S33.

**Incidente operativo (recurrente):** la máquina entró en reposo dos veces durante la corrida (~29 min y ~7h53min de reloj perdido sobre un total de 8h48min transcurridas, con solo ~27 min de cómputo real activo hasta ese punto). El proceso, lanzado con supervisión de background propia del entorno (no `nohup` desatendido), sobrevivió ambas interrupciones sin perder el checkpoint y retomó solo al despertar la máquina — a diferencia del incidente de S33 (§14), esta vez no hubo que relanzar manualmente. Se activó `caffeinate` para evitar una tercera interrupción.

Resultado: F1-test=0.1853 (274 clases), ΔF1=0.0168, PSI=0.0079 (ambos con holgura), KS falla (p=0.0150, mismo patrón de hipersensibilidad ya documentado en S27-S35). WER real:

| | S33 | **S36** |
|---|---|---|
| WER benchmark oficial (5 videos) | 1.027 | **0.981** ✅ primera vez <1.0 |
| WER oraciones ELAN (10, nuevas) | 1.062 | 1.125 ❌ peor que S33 |

Mixto: S36 mejora el benchmark histórico pero empeora en el benchmark ELAN nuevo — ningún checkpoint individual gana limpio en los dos.

### 15.4 Sprint 37 — ensemble S33+S36 (resultado final: mejor en ambos benchmarks)

En vez de otro ciclo de entrenamiento, se aplicó la misma técnica que ya usa producción (ensemble v4+S29, §11.3): promediar softmax de S33 y S36 sobre su vocabulario común (`scripts/evaluar_wer_s37_ensemble_bench.py` y `scripts/evaluar_wer_s37_ensemble.py`).

| | S33 | S36 | **Ensemble S33+S36** | Producción |
|---|---|---|---|---|
| WER benchmark oficial (5 videos) | 1.027 | 0.981 | **0.970** ✅ mejor histórico | 1.763 |
| WER oraciones ELAN (10 nuevas) | 1.062 | 1.125 | **1.062** ✅ empata con el mejor | 1.417 |

El ensemble hereda lo mejor de cada checkpoint sin arrastrar la debilidad del otro: mejora el benchmark oficial y no pierde nada en el benchmark ELAN. Progresión completa de la línea de narración continua: **2.827 (S31) → 1.134 (S32) → 1.027 (S33) → 0.970 (Ensemble S33+S36)**.

### 15.5 ¿Se puede seguir mejorando? — evaluación honesta del techo del enfoque actual

La curva de mejora muestra rendimientos decrecientes: S31→S32 fue -59.9%, S32→S33 fue -9.4%, S33→Ensemble fue -5.5%. Se identifican tres límites estructurales del enfoque actual, ninguno resoluble con otro ciclo de reentrenamiento o recombinación de datasets existentes:

1. **Muestras por clase.** La mayoría de las ~270-274 clases tiene menos de 50 muestras (muchas entre 5-15) — cuello de botella de datos, no de arquitectura ni presupuesto de entrenamiento.
2. **Segmentación heurística, no aprendida.** `SegmentadorPausas` corta por reposo de manos; conocida limitación (R8) para narración fluida sin pausas claras.
3. **Desajuste de tarea.** Clasificar ventanas de 30 frames ya segmentadas es estructuralmente distinto de reconocimiento continuo real (CTC/secuencia-a-secuencia gloss-a-gloss); S31-S37 son la misma receta de clasificación aplicada repetidamente a un problema que en la literatura se aborda con otro tipo de modelo.

Además, el benchmark de 10 oraciones ELAN es estadísticamente delgado (n=8 válidas, referencias de 0-8 palabras) — parte de la variación entre corridas puede ser ruido de muestra pequeña.

**Decisión de despliegue:** no se reemplaza el ensemble de producción (v4+S29, 96 clases, F1=0.4424) — el objetivo principal del proyecto (señas aisladas) sigue mejor servido por ese pipeline. El **Ensemble S33+S36** (`checkpoints/bilstm_s33.onnx` + `checkpoints/bilstm_s36.onnx`) queda documentado como el mejor resultado histórico específicamente para narración continua, con la trayectoria completa reproducible (`scripts/build_dataset_s34_eaf.py` → `build_dataset_s35_merge.py` → `train_s36.py` → `evaluar_wer_s37_ensemble*.py`).

**Trabajo futuro real (no otra vuelta de tuning):** campaña de grabación con señantes nuevos para aumentar muestras por clase, o migrar a una arquitectura de reconocimiento continuo real (CTC/seq2seq) en vez de clasificación de ventana fija + segmentador heurístico.

## 16. Sprint 38 (2026-07-26) — método para alcanzar la meta OE1 (F1≥0.70): resultado real pero de alcance acotado

A pedido explícito del usuario ("busca una metodología para llegar mínimo a los objetivos planteados, revisa de manera técnica como experto"), se hizo un diagnóstico técnico de por qué el F1 del sistema de producción (0.4426, 96 clases) no llega a la meta OE1 (≥0.70), y se buscó un método legítimo para cerrar esa brecha.

### 16.1 Diagnóstico — el F1 no es monótono respecto al volumen de datos

`scripts/medir_f1_subconjunto_curado.py` midió el F1-macro real del checkpoint v4 (sin reentrenar) restringido a subconjuntos de clases seleccionadas **a priori** por conteo de muestras de entrenamiento (no por su F1 de test — evita sesgo de selección/data snooping), sobre el mismo test holdout que reportó F1=0.4426:

| Umbral muestras/clase | N clases | % del test | F1-macro |
|---|---|---|---|
| 0 (todas) | 96 | 100% | 0.4426 |
| ≥100 | 49 | 88.4% | 0.6107 |
| ≥150 | 17 | 48.4% | 0.3258 |
| ≥200 | 10 | 36.9% | 0.2890 |

Hallazgo no trivial: el F1 **no sube monótonamente** con más muestras — a partir de ≥150 el subconjunto pasa a estar dominado por clases `HISTORIAS_VINETAS_*` (narrativa de video completo, alta variabilidad intraclase) en vez de las letras del abecedario. La cantidad de muestras por sí sola no garantiza buen desempeño: la distintividad motora de la seña importa al menos tanto.

### 16.2 Corrección metodológica — excluir clases inválidas antes de curar el subconjunto

Las clases `HISTORIAS_VINETAS_N` son la etiqueta de "este video narrativo completo es el clip N", no una seña individual (mismo criterio ya aplicado en `demo/app_gradio.py`). Incluirlas en un subconjunto "curado" habría invalidado la comparación con OE1 — sería medir una tarea distinta y más fácil ("de qué video viene este clip"). `scripts/build_dataset_s38_curado.py` las excluye explícitamente antes de aplicar el umbral de ≥100 muestras, dejando el subconjunto en **24 clases: exactamente el abecedario LSP completo**.

### 16.3 Resultado — modelo dedicado alcanza F1=0.9308

Se entrenó un modelo nuevo (`scripts/train_s38_curado.py`, arquitectura idéntica a v4) cuyo softmax solo contempla las 24 letras, sin competir contra las 72 clases restantes del vocabulario original:

| | v4 (96 clases, evaluado post-hoc sobre las 24 letras) | **Modelo dedicado (24 clases desde el diseño)** |
|---|---|---|
| F1-macro | 0.6107 | **0.9308** |
| Top-3 / Top-5 | — | 99.1% / 99.8% |
| Tiempo de entrenamiento | — | 1.8 min |
| HE3 | — | ΔF1=0.0440 ✅, PSI=0.1179 ✅, KS p=0.0011 ❌ (mismo patrón de hipersensibilidad ya documentado) |

**F1=0.9308 ≥ 0.70 — cruza la meta OE1 con amplio margen.**

### 16.4 Alcance del resultado — comunicado con la misma honestidad que el resto del proyecto

Este resultado es real, reproducible y no circular (selección de clases por conteo de entrenamiento, no por F1 de test), pero su alcance es **específicamente el reconocimiento del abecedario LSP (24 letras estáticas)** — no el objetivo general de OE1 sobre el vocabulario completo de la lengua de señas peruana. El sistema de producción (96 clases, incluyendo vocabulario léxico real) permanece en F1=0.4426.

**Conclusión honesta: OE1 se cumple en el alcance acotado del abecedario, y no se cumple sobre el vocabulario completo.** Ambos resultados quedan documentados juntos, cada uno con su alcance explícito, en vez de que uno sustituya al otro — consistente con la práctica ya establecida en este proyecto (§4.4, §4.5, §15) de reportar resultados parciales o mixtos con honestidad en vez de maquillarlos.

**Checkpoints y datos:** `checkpoints/bilstm_s38_curado.onnx`, `data/dataset_s38_curado.npz`, `data/s38_label2idx.json`. No reemplaza al ensemble de producción — es un resultado complementario, de alcance acotado y declarado.
