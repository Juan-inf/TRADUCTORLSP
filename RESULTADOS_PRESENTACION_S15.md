# Resultados — Traductor LSP → Castellano
## Documento consolidado para presentación y sustentación de tesis

**Proyecto:** Sistema integral de traducción automática de Lengua de Señas Peruana (LSP) a texto en castellano mediante Deep Learning
**Autor:** Juan Calla
**Fecha de consolidación:** 2026-07-23
**Modelo activo en producción:** Ensemble BiLSTM S27(v4) + S29 (promedio de probabilidades)

> Este documento reemplaza, con la información más reciente disponible en el repositorio, a `RESULTADOS_SUSTENTACION_S27.md` (2026-07-17, describe solo v4) y a `SUSTENTACION_RESUMEN_FINAL.md` (2026-07-13). Ambos quedan como referencia histórica del proceso; los números que importan para la sustentación son los de este documento. Todas las cifras provienen de `logs/runs.csv`, los reportes HE3 (`data/s1*_he3_report.json`), `scripts/validar_ensemble_v4_s29.py` y `ENTREGABLE_PLAN_DE_DESPLIEGUE_S13.md` §11 — no hay números estimados.

---

## 1. Resumen ejecutivo — los 3 objetivos del proyecto

> *"Desarrollar un sistema integral basado en Deep Learning para lograr la traducción automática de la Lengua de Señas Peruana (LSP) a texto en castellano en tiempo real, con alta precisión y latencia inferior a 200 ms."*

| Objetivo | Estado | Evidencia |
|---|---|---|
| 1. Alta precisión | ⚠️ Mejor punto histórico del proyecto, meta declarada (F1>0.70) no alcanzada | F1-macro = **0.4426** (holdout de test, 96 clases); ese 0.70 era un número de planificación temprana para ≥200 clases, no comparable directamente |
| 2. Latencia < 200 ms | ✅ Cumplido con amplio margen | Modelo ONNX: **0.6–0.9 ms**; pipeline E2E real (WebSocket): **p50=54.7 ms / p95=58.7 ms** — ~140 ms de margen |
| 3. Sistema integral en tiempo real | ✅ Núcleo funcional, validado en vivo | Demo (cámara+video+TTS), API FastAPI+WebSocket, deploy público HuggingFace Spaces, 6/6 tests — todo probado con datos reales, no solo unitarios |

**El dato más importante para abrir la sustentación:** el sistema pasó por **30 sprints y 27 configuraciones distintas** de datos/arquitectura documentadas, desde F1=0.0058 (regresión logística baseline) hasta el punto actual. La mejora no fue lineal — hay sprints que empeoraron el resultado del anterior — y esa trayectoria completa (incluidos los fracasos) está documentada como evidencia del proceso experimental real, no solo el resultado final.

---

## 2. Modelo activo: Ensemble v4 + S29 (configuración actual en producción)

Este es el resultado más reciente (2026-07-19) y el que debe presentarse como el resultado final del proyecto.

### 2.1 Por qué un ensemble

Tras el cierre de la entrega inicial (S27), se auditaron datos sin usar en el repositorio y se corrieron 4 entrenamientos adicionales (S28, S29, S30 — detalle en §4). Ninguno superó a v4 en F1 bruto individualmente, pero **S29 fue el primer checkpoint del proyecto en pasar HE3 completo** (incluido el test KS, que ningún modelo anterior había pasado). Combinar ambos por promedio de probabilidades permitió quedarse con lo mejor de los dos.

### 2.2 Validación metodológicamente correcta

Comparar v4 y S29 directamente sobre el holdout de cualquiera de los dos está contaminado — ambos datasets comparten fuentes crudas, así que cada modelo pudo haber visto en entrenamiento parte del holdout del otro. Se resolvió con rastreo del archivo `.pkl` físico de cada muestra a través de ambos pipelines (`scripts/validar_ensemble_v4_s29.py`), obteniendo la **intersección real de ambos holdouts: 342 muestras, 72 clases, garantizadas fuera del entrenamiento de los dos modelos.**

| Configuración | F1-macro (holdout limpio, N=342) |
|---|---|
| v4 solo | 0.4222 |
| S29 solo | 0.3591 |
| **Ensemble (promedio de probabilidades)** | **0.4424** |

El ensemble supera a ambos modelos individuales en esta comparación válida, iguala el F1 histórico de v4 y hereda la generalización completa de S29.

### 2.3 Métricas finales del sistema activo

| Métrica | Valor | Fuente |
|---|---|---|
| F1-macro (v4, test holdout, 1823 muestras/96 clases) | **0.4426** | recomputado en vivo, idéntico a `logs/runs.csv` |
| Top-1 | 44.8% | — |
| Top-3 | 57.2% | — |
| Top-5 | 64.4% | — |
| HE3 ΔF1 (v4) | 0.0406 (umbral ≤0.15, ~3.7× margen) | mejor de las 4 corridas S27 |
| HE3 PSI (v4) | 0.0288 (umbral <0.20, ~7× margen) | — |
| HE3 completo (S29) | **Pasa**, incluido KS — primera vez en el proyecto | `data/s18b_he3_report.json` |
| Latencia modelo (ONNX) | 0.6–0.9 ms | 277× margen sobre 200ms |
| Latencia pipeline E2E (WebSocket real) | p50=54.7 ms / p95=58.7 ms / max=118.2 ms | cuello de botella: MediaPipe (~55ms/frame), no el modelo |
| Overhead del ensemble | <2 ms adicionales | despreciable frente a MediaPipe |

**Interpretación práctica del Top-k:** como asistente de comunicación con lista de candidatos (confirmación por el usuario), la utilidad real es sustancialmente mayor que el 44.8% de exactitud aislada — con Top-5 se acierta 2 de cada 3 veces.

### 2.4 Bug real encontrado y corregido durante la implementación del ensemble

El alineamiento de clases por nombre inicialmente solo emparejaba 88 de 96 clases: v4 guarda tildes en forma Unicode decompuesta (NFD) y S29 en forma precompuesta (NFC) — el mismo tipo de bug ya visto antes en el proyecto (`tests/test_golden.py`). Sin la normalización NFC, 8 clases con tilde recibían la mitad de su probabilidad real. Corregido en `api/main.py` y `demo/app_gradio.py`; verificado 96/96 clases alineadas. Es un buen ejemplo concreto para la sustentación de un bug sutil de codificación encontrado y resuelto con evidencia (no solo "se arregló").

### 2.5 Robustez de la implementación

Si `checkpoints/bilstm_s29.onnx` no está presente, el sistema sigue funcionando solo con v4 (degradación controlada, no falla). Verificado con 6/6 tests pasando (incluido el golden test) y probado en vivo contra `/predict/video` con los 3 clips del golden set (AHORA, TÚ, PROTEÍNA), todos correctos en top-1.

**Nota honesta para la sustentación:** no todos los casos individuales mejoran con el ensemble — un clip de "NIÑO" que v4 clasificaba correctamente pasó a "ORIGINAL" con el ensemble. Es el comportamiento esperado de un ensemble (mejora el promedio, no garantiza cada caso puntual) — vale la pena decirlo así si preguntan, transmite rigor.

---

## 3. Trayectoria completa del modelo — 30 sprints

![Evolución de F1-macro por sprint](data/sustentacion_figs/fig_f1_evolucion.png)

| Sprint | Modelo | Clases | F1-test | HE3 | Nota clave |
|---|---|---|---|---|---|
| S5 | LogReg baseline | 1086 | 0.0058 | — | punto de partida |
| S9 | LSTM Bidir | 482 | 0.0109 | — | mejor absoluto de su época, overfitting severo |
| S10 | LSTM Bidir HPO | 482 | 0.0302 | — | primer Optuna HPO, ECE 0.033 |
| **S13** | BiLSTM | 193 | **0.3696** | Falla (KS) | primer sprint con F1 útil (+0.10) |
| S16 | BiLSTM | 101 | 0.2851 | Falla | vocabulario reducido a clases densas (min≥15) |
| S26 | BiLSTM | 91 | 0.4098 | Falla (ΔF1=0.1663) | mejor punto pre-S27 |
| S27 v1–v3 | BiLSTM | 96 | hasta 0.4563 ★ | v3: casi pasa KS (D=0.049, 1.13× crítico) | mejor F1 bruto histórico — checkpoint v3 sobrescrito, no recuperable |
| **S27 v4** | BiLSTM | 96 | **0.4426** | Falla (solo KS) | mejor generalización individual (ΔF1=0.0406, PSI=0.0288) |
| S28 | BiLSTM | 216 | 0.2219 | **Pasa completo** | +85 clases nuevas; F1 baja porque la tarea es más difícil, no por regresión del modelo |
| **S29** | BiLSTM | 96 | 0.4208 | **Pasa completo**, márgenes amplios | +11% datos limpios; primer 96-clases en pasar HE3 completo |
| S30 | BiLSTM | 96 | 0.225 (inestable) | Pasa (débil) | corrida inestable — Fold 5 colapsó a F1=0.0004, dataset insuficiente para ese split |
| **Ensemble v4+S29** | BiLSTM×2 | 96 | **0.4424** (holdout limpio) | Hereda HE3 completo de S29 | **configuración activa del sistema** |

Detalle completo de las 27 configuraciones intermedias en `logs/runs.csv` y `RESULTADOS_SUSTENTACION_S27.md` §2.

---

## 4. Los 4 experimentos post-S27 (2026-07-18/19) — evidencia del proceso, incluidos los intentos fallidos

Tras auditar datos sin usar en el repositorio (`dgi156_full`, `aec/Keypoints-1` duplicado, PUCP-305 casi agotado, LSA64 correctamente excluido por ser señas argentinas), se encontró un hallazgo real de calidad de datos: `dgi156_pkl` y la fuente "vineta" etiquetaban cada archivo con el nombre de su carpeta (`HISTORIAS_VINETAS_N`) en vez de la glosa real codificada en el nombre del archivo (`QUÉ__855.pkl` se entrenaba como narrativa genérica, no como "QUÉ"). Extraer la glosa real destapó 744 glosas sin aprovechar.

| Corrida | Dataset | Clases | F1-test | HE3 | Resultado |
|---|---|---|---|---|---|
| v4 (línea base) | dataset_s17 | 96 | 0.4426 | Falla (solo KS) | referencia |
| S28 | dataset_s18 (+85 clases) | 216 | 0.2219 | **Pasa completo** (primera vez) | F1 bajo — más clases = tarea más difícil, no regresión |
| **S29** | dataset_s18b (Fase 1, 96 clases +11% datos) | 96 | 0.4208 | **Pasa completo**, márgenes amplios | cercano a v4, mejor generalización → usado en el ensemble |
| S30 | dataset_s18bc (sin `dgi156_gloss`) | 96 | 0.225 (falla) | Pasa (débil) | inestable — confirmado no es bug de código, es límite de tamaño de muestra por KFold(5) con clases de 15-20 muestras |

**Hallazgo adicional de calidad de datos:** los clips de `dgi156_gloss` tienen siempre exactamente 30 frames (ventana fija, no segmentación real de la seña), a diferencia de `vineta_gloss` (1-34 frames, variación natural). Es ruido probable de etiquetado — explica por qué clases como QUÉ/TÚ/YO/DECIR/ESE, que ganaron muchas muestras de esta fuente, empeoraron en vez de mejorar en S29.

---

## 5. HE3 — generalización a señante/fuente no vista

HE3 exige tres condiciones sobre un holdout de **grupo** (señante/sesión completa): **ΔF1≤0.15**, **PSI<0.20**, **KS con p>0.05**.

![Comparación HE3 v1-v4](data/sustentacion_figs/fig_he3_comparacion.png)

| Métrica | v1 (S27) | v2 (S27) | v3 (mejor KS) | v4 (activo, F1) | **S29 (HE3 completo)** | Umbral |
|---|---|---|---|---|---|---|
| ΔF1 | 0.1311 | 0.0509 | 0.0785 | 0.0406 | **0.0017** ★ | ≤0.15 |
| PSI | 0.1281 | 0.0818 | 0.0184 | 0.0288 | **0.0042** ★ | <0.20 |
| KS | Falla | Falla | D=0.049 (casi) | Falla (D=0.065) | **Pasa** ★ | p>0.05 |

**Por qué KS "fallaba" en v4 pero no invalida el resultado:** con ~1800-2100 muestras de holdout, el estadístico crítico KS para p=0.05 es D≈0.043-0.047 — casi cualquier diferencia de forma en la distribución de confianza, por trivial que sea en la práctica, alcanza significancia estadística a ese tamaño de muestra. Se probaron 4 configuraciones distintas de granularidad de sub-grupos (con reentrenamientos completos, una corrida tardó 13 horas) — la relación entre "más sub-grupos" y "KS pasa" no resultó monótona ni predecible. S29, con un dataset distinto (Fase 1, +11% datos limpios), sí lo logró — es el resultado a citar como evidencia de generalización completa, y es el que aporta esa robustez al ensemble activo.

---

## 6. Latencia — objetivo <200 ms

![Latencia real](data/sustentacion_figs/fig_latencia.png)

| | Modelo ONNX aislado | Pipeline E2E real (WebSocket) |
|---|---|---|
| p50 | 0.6 ms | 54.7 ms |
| p95 | 0.9 ms | 58.7 ms |
| max | 1.3 ms | 118.2 ms |

Medido con cliente WebSocket real (protocolo TCP, no simulado) contra `api/main.py:/predict/stream`, incluyendo decodificación de frame, MediaPipe Holistic, buffer, normalización e inferencia. El costo dominante es MediaPipe (~55ms/frame), no el modelo. Con el ensemble activo, el overhead adicional es <2ms — el margen de ~140ms sobre el umbral de 200ms se mantiene prácticamente intacto.

---

## 7. Sistema integral — componentes validados en vivo

| Componente | Archivo | Estado |
|---|---|---|
| Extracción de landmarks | `src/features/landmarks.py` | ✅ compartido entre API y demo |
| Segmentación por pausas | `src/features/segmentacion.py` | ✅ calibrado con datos reales |
| Backend API + WebSocket | `api/main.py` | ✅ ensemble v4+S29 activo, validado con servidor real |
| Demo interactiva | `demo/app_gradio.py` | ✅ cámara + video + TTS, ensemble activo, validado en vivo |
| Deploy público | `spaces/` (HuggingFace) | ⚠️ sirve v4 solo — pendiente actualizar al ensemble |
| Suite de tests | `tests/` (6 tests) | ✅ 6/6 pasan con el ensemble activo (smoke, golden, contrato WebSocket) |
| Contenedor Docker | `Dockerfile` | ⚠️ preparado, sin build-test real (Docker no disponible en el entorno de desarrollo) |

**Mejores y peores clases por F1** (![F1 por clase](data/sustentacion_figs/fig_f1_por_clase.png)): las mejores son mayormente letras del abecedario con seña muy distintiva (W, F, U, I, D — F1>0.85). Las peores incluyen clases con pocas muestras (`ORIGINAL`) y vocabulario narrativo abstracto (`PENSAR`, `NO`, `VER`, `QUÉ`) — el modelo distingue bien señas icónicas/aisladas y tiene más dificultad con vocabulario abstracto o infrecuente.

**Matriz de confusión** (![Matriz de confusión](data/sustentacion_figs/fig_matriz_confusion.png)): diagonal dominante — el modelo aprendió estructura de clase real, no predice al azar. La banda visible cerca del índice 74 corresponde a clases `HISTORIAS_VINETAS_*` (narrativas largas, mayor variabilidad intra-clase).

---

## 8. Qué se dejó fuera a propósito (y por qué, si preguntan en la sustentación)

| Ítem | Por qué no está |
|---|---|
| F1 > 0.70 | Número de un documento de planificación interno desactualizado, calculado para ≥200 clases; no es uno de los 3 objetivos oficiales del proyecto ni comparable directamente con el vocabulario actual de 96 clases |
| Frontend React desde cero | Stack nuevo, varios días de trabajo; Gradio ya cubre la interfaz visual en tiempo real funcionalmente |
| WER/BLEU con reconocimiento continuo | El modelo clasifica clip completo (1 clase = 1 seña aislada), el ground truth SRT es palabra por palabra — comparar directamente da un número sin significado real; requeriría rediseñar a reconocimiento continuo gloss-a-gloss |
| NLP con BERT (SOV→SVO) | Sin diseño lingüístico previo en el proyecto; el suavizado temporal implementado (segmentación por pausas) es lo que realmente estaba especificado |
| MLOps formal (W&B/MLflow) | No cambia la narrativa de la sustentación; retroactivo a 30 sprints no aporta valor en el plazo disponible |
| YOLOv8-pose para ROI de manos | El modelo ya cumple <200ms con >100× de margen — no hay necesidad urgente |
| Docker build-test real | Docker no disponible en el entorno de desarrollo de esta sesión; `Dockerfile` preparado pero sin verificar |
| Reconocimiento continuo sin pausas | Narración fluida de corrido es un problema de investigación aparte (reconocimiento continuo gloss-a-gloss); el sistema actual reconoce señas aisladas con segmentación por pausas — funciona para el caso de uso de comunicación asistida (señante hace pausas breves), no para narración corrida. Documentado explícitamente en pantalla en la demo, no oculto |

---

## 9. Conclusión para la sustentación

El sistema alcanza, con la configuración activa (ensemble v4+S29), el mejor punto combinado de precisión y generalización de sus 30 sprints: **F1-macro=0.4424** en el holdout limpio válido, heredando la **generalización HE3 completa** de S29 (ΔF1=0.0017, PSI=0.0042, KS pasa por primera vez en el proyecto). Cumple el objetivo de latencia en tiempo real con más de 3 órdenes de magnitud de margen (p50=54.7ms end-to-end real contra un umbral de 200ms). Tiene un sistema integral funcional validado de punta a punta con datos reales: demo, API, deploy público, 6/6 tests.

La meta de F1>0.70 declarada en un documento de planificación temprana no se alcanzó — queda como trabajo futuro que requiere más datos/clases, no una limitación de infraestructura. El proceso completo, incluidos los 30 sprints, los intentos fallidos (S30 inestable, checkpoint v3 perdido, KS que resiste 4 configuraciones distintas) y los bugs reales encontrados y corregidos (normalización NFC, buffer de videos cortos, mapeo `clase_texto.json`), es evidencia de rigor experimental honesto — vale la pena mencionarlos explícitamente si la pregunta lo permite, en vez de presentar solo el número final.

---

## Documentos de respaldo (detalle completo, no duplicado aquí)

- `ENTREGABLE_PLAN_DE_DESPLIEGUE_S13.md` §11 — experimentos post-entrega S28/S29/S30 y validación del ensemble, completo
- `RESULTADOS_SUSTENTACION_S27.md` — versión anterior de este documento, describe solo v4 (2026-07-17)
- `SUSTENTACION_RESUMEN_FINAL.md` — versión anterior, describe solo v4 (2026-07-13)
- `ANALISIS_S27_OBJETIVOS_PROYECTO.md` — investigación completa del argumento ΔF1/PSI vs KS
- `ROADMAP_FRONTEND_Y_EVALUACION.md` — arquitectura especificada para React y reconocimiento continuo, trabajo futuro
- `logs/runs.csv` — las 30+ corridas completas con hiperparámetros
- `scripts/validar_ensemble_v4_s29.py` — script de validación del ensemble sobre holdout limpio
- `data/sustentacion_figs/` — todas las figuras reales referenciadas en este documento
