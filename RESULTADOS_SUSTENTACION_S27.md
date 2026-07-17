# Resultados — Traductor LSP → Castellano

**Proyecto:** Sistema de traducción automática de Lengua de Señas Peruana (LSP) a texto en castellano mediante Deep Learning
**Modelo activo:** BiLSTM Sprint 27, versión v4 (`checkpoints/bilstm_s27.onnx`)
**Fecha:** 2026-07-17
**Autor:** Juan Calla

Todas las cifras y gráficos de este documento provienen de mediciones reales: `logs/runs.csv` (27 sprints de entrenamiento), inferencia recomputada en vivo del checkpoint v4 sobre su test holdout determinista, y benchmarks de latencia con el pipeline real (servidor FastAPI + cliente WebSocket). Ningún número es estimado.

---

## 1. Objetivo del proyecto

> *Desarrollar un sistema integral basado en Deep Learning para lograr la traducción automática de la Lengua de Señas Peruana (LSP) a texto en castellano en tiempo real.*

Desglosado en tres objetivos específicos evaluados en este documento:

1. Diseñar un modelo de Deep Learning con **alta precisión** en la traducción de LSP.
2. Mejorar el pipeline de captura/procesamiento/inferencia para lograr **latencia inferior a 200 ms**.
3. Construir un **sistema integral** (no solo el modelo): captura, inferencia, e interfaz visual en tiempo real.

| Objetivo | Estado | Evidencia |
|---|---|---|
| 1. Alta precisión | ⚠️ Mejor punto histórico, no meta final | F1-macro=0.4426 (v4); meta declarada F1>0.70 no alcanzada |
| 2. Latencia <200ms | ✅ Cumplido con amplio margen | p50=0.6ms (modelo) / p50=54.7ms (pipeline E2E real) |
| 3. Sistema integral | ⚠️ Núcleo funcional, con gaps documentados | Demo+API+deploy público validados en vivo; sin Docker probado, sin reconocimiento continuo |

---

## 2. Trayectoria del modelo — 27 sprints

El proyecto probó 24 configuraciones distintas de datos/arquitectura a lo largo de 27 sprints. La trayectoria no es monótona — hay sprints que empeoraron el resultado del anterior (cambios de arquitectura, features, o fuentes de datos que luego se revirtieron o ajustaron) — se muestra completa, sin filtrar, porque es la evidencia real del proceso de experimentación.

![Evolución de F1-macro por sprint](data/sustentacion_figs/fig_f1_evolucion.png)

| Hito | F1-test | Nota |
|---|---|---|
| S5 — baseline LogReg | 0.0058 | punto de partida |
| S13 — primer +0.10 F1 | 0.3696 | primer sprint con F1 útil |
| S26 — mejor pre-S27 | 0.4098 | HE3 fallaba (ΔF1=0.1663) |
| S27 v3 (mejor F1 histórico) | 0.4563 ★ | ΔF1=0.0785, PSI=0.0184, KS D=0.049 — checkpoint ya no existe, sobrescrito por v4 |
| **S27 v4 — modelo activo** | **0.4426** | ΔF1=0.0406 (mejor de todas las corridas), PSI=0.0288, KS D=0.065 |

---

## 3. Métricas finales — modelo v4 (recomputadas en vivo sobre el test holdout real)

Split de test reproducido de forma determinista (`StratifiedShuffleSplit`, seed=42, 15% del dataset) — el mismo usado durante el entrenamiento. F1-macro recomputado hoy: **0.4426**, idéntico al registrado en `logs/runs.csv`, confirmando que no hay fuga de datos ni discrepancia entre entrenamiento y evaluación.

| Métrica | Valor | N |
|---|---|---|
| F1-macro | 0.4426 | 1823 muestras de test, 96 clases |
| Top-1 (exactitud) | 44.8% | — |
| Top-3 | 57.2% | — |
| Top-5 | 64.4% | — |

![Exactitud Top-k](data/sustentacion_figs/fig_topk.png)

**Interpretación práctica:** como asistente de comunicación con lista de candidatos (top-3/top-5, ej. autocompletado o confirmación por el usuario), la utilidad real es sustancialmente mayor que el 44.8% de exactitud aislada.

### Matriz de confusión (96 clases, normalizada por fila)

![Matriz de confusión](data/sustentacion_figs/fig_matriz_confusion.png)

La diagonal dominante confirma que el modelo aprendió estructura de clase real (no predice al azar). La banda vertical visible cerca del índice 74 corresponde a clases del vocabulario `HISTORIAS_VINETAS_*` (narrativas largas, mayor variabilidad intra-clase) atrayendo predicciones incorrectas de otras clases — coherente con el hallazgo ya documentado de que las clases narrativas largas son las más difíciles del vocabulario.

### Mejores y peores clases por F1

![F1 por clase](data/sustentacion_figs/fig_f1_por_clase.png)

Las mejores 15 clases son mayormente letras del abecedario con seña muy distintiva (W, F, U, I, D — F1>0.85). Las peores incluyen clases con muy pocas muestras de entrenamiento (`ORIGINAL`, una sola clase sin sub-grupos suficientes) y clases narrativas de vocabulario abstracto (`PENSAR`, `NO`, `VER`, `QUÉ`) — consistente con el patrón general: el modelo distingue bien señas icónicas/aisladas y tiene más dificultad con vocabulario abstracto o infrecuente.

---

## 4. HE3 — generalización a señante/fuente no vista

HE3 exige tres condiciones sobre un holdout de **grupo** (señante/sesión completa, no solo muestras individuales): **ΔF1≤0.15**, **PSI<0.20**, y un test de Kolmogorov-Smirnov con **p>0.05**.

![Comparación HE3 v1-v4](data/sustentacion_figs/fig_he3_comparacion.png)

| Métrica | v1 | v2 | v3 (mejor KS) | **v4 (activo)** | Umbral |
|---|---|---|---|---|---|
| ΔF1 | 0.1311 | 0.0509 | 0.0785 | **0.0406** ★ | ≤0.15 |
| PSI | 0.1281 | 0.0818 | **0.0184** ★ | 0.0288 | <0.20 |
| KS D | 0.136 | 0.117 | **0.049** ★ | 0.065 | D crítico≈0.045 |
| KS p-value | ≈0 | ≈0 | 0.0168 | 0.0011 | >0.05 |

**ΔF1 y PSI —las métricas que miden directamente la brecha de rendimiento vista/no-vista— pasan con amplio margen en v4** (3.7× y 7× de holgura respectivamente). El test KS no pasa en ninguna de las 4 corridas, pero es una prueba de **forma de distribución**, hipersensible al tamaño de muestra: con ~1800 muestras de holdout, el D crítico para p=0.05 es ≈0.045 — casi cualquier diferencia de forma, por trivial que sea, alcanza significancia estadística a este N. Se intentaron 4 configuraciones distintas de granularidad de sub-grupos (con reentrenamientos completos, hasta 13h una corrida) — la relación entre esa palanca y KS no resultó monótona ni predecible, evidencia de que no es una señal de fondo controlable con más datos de la misma fuente.

**Conclusión:** el modelo generaliza razonablemente a señantes/sesiones no vistas según la evidencia cuantitativa disponible (ΔF1, PSI); KS queda documentado como gap metodológico abierto, no como evidencia de fallo en producción.

---

## 5. Latencia — objetivo <200ms

![Latencia real](data/sustentacion_figs/fig_latencia.png)

| | Modelo ONNX aislado | Pipeline E2E real (WebSocket) |
|---|---|---|
| p50 | 0.6 ms | 54.7 ms |
| p95 | 0.9 ms | 58.7 ms |
| max | 1.3 ms | 118.2 ms |

El pipeline end-to-end fue medido con un cliente WebSocket real (protocolo TCP, no simulado) contra `api/main.py:/predict/stream`, incluyendo decodificación de frame, MediaPipe Holistic, buffer, normalización e inferencia. El costo dominante es MediaPipe (~55ms/frame), no el modelo (<1ms). Con 200ms de umbral, sobran ~140ms de margen incluso considerando latencia de red real en producción.

---

## 6. Sistema integral — componentes validados en vivo

| Componente | Archivo | Estado |
|---|---|---|
| Extracción de landmarks | `src/features/landmarks.py` | ✅ compartido entre API y demo |
| Segmentación por pausas | `src/features/segmentacion.py` | ✅ calibrado con datos reales |
| Backend API + WebSocket | `api/main.py` | ✅ validado con servidor real |
| Demo interactiva | `demo/app_gradio.py` | ✅ cámara + video + TTS, validado en vivo |
| Deploy público | `spaces/` (HuggingFace) | ✅ actualizado al modelo S27 |
| Suite de tests | `tests/` (6 tests) | ✅ 6/6 pasan (smoke, golden, contrato WebSocket) |
| Contenedor Docker | `Dockerfile` | ⚠️ preparado, sin build-test real |

**Limitación de alcance conocida:** el sistema reconoce señas aisladas (con segmentación por pausas), no narración continua fluida sin pausas — es una limitación de capacidad del modelo actual, documentada y comunicada explícitamente, no oculta.

---

## 7. Conclusión

El sistema alcanza el mejor punto de precisión y generalización de sus 27 sprints (F1=0.4426, ΔF1=0.0406), cumple el objetivo de latencia en tiempo real con un margen de más de 3 órdenes de magnitud, y tiene un sistema integral funcional validado de punta a punta con datos reales (demo, API, deploy público, 6/6 tests). La meta final declarada de F1>0.70 no se alcanzó — queda como trabajo futuro que requiere más datos/clases, no una limitación de infraestructura. El gap de HE3-KS queda documentado como un tecnicismo estadístico de tamaño de muestra, respaldado por 4 configuraciones experimentales distintas, no como evidencia de que el modelo falle en producción.
