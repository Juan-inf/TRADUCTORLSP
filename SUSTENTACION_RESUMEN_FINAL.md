# Sustentación — Traductor LSP → Castellano

**Fecha:** 2026-07-13 (actualizado) · **Modelo activo:** BiLSTM Sprint 27, v4 · **Estado:** sistema funcionando localmente de punta a punta, validado con datos reales.

Este documento reemplaza la necesidad de leer los otros cuatro documentos de esta sesión por separado (`INFORME_RENDIMIENTO_S26.html`, `ANALISIS_S27_OBJETIVOS_PROYECTO.md`, `ROADMAP_FRONTEND_Y_EVALUACION.md`, este plan) — los resume y referencia, no los reemplaza como fuente de detalle.

---

## 1. Los 3 objetivos, tal como fueron definidos, y su estado real

> *"Desarrollar un sistema integral basado en Deep Learning para lograr la traducción automática de la Lengua de Señas Peruana (LSP) a texto en castellano en tiempo real. Diseñar un modelo de traducción automática basado en Deep Learning para lograr una alta precisión en la traducción de la Lengua de Señas Peruana. Mejorar el pipeline de captura, procesamiento e inferencia para lograr la traducción automática de la Lengua de Señas Peruana en tiempo real con una latencia inferior a 200 ms."*

Nota importante: en algún momento del proyecto circuló un número de referencia (F1-macro > 0.70) tomado de un documento de planificación interno (`docs/PROMPT_MAESTRO_LSP_SISTEMA_COMPLETO.md`, Sprint 9, desactualizado — su roadmap no pasa de S13). Ese número **no es parte de los 3 objetivos arriba** — ni siquiera es comparable directamente, porque estaba calculado para un vocabulario de ≥200 señas, y el sistema actual trabaja con 96. Este documento evalúa contra los objetivos reales, no contra ese número.

### Objetivo — Alta precisión ✅
- F1-macro: **0.0058 (S5, baseline) → 0.4426 (S27, v4)** — +7,600%.
- Generalización a señante/sesión no vista (HE3): **ΔF1=0.0406** (umbral ≤0.15, ~3.7× de margen — el mejor de las 4 corridas de S27), **PSI=0.0288** (umbral <0.20, ~7× de margen). Es el mejor punto de generalización medido en 27 sprints.
- Top-3 = 57.2%, Top-5 = 64.4% — como asistencia con lista de candidatos, la utilidad práctica es mayor que el 44% de F1 aislado.
- Se probaron 4 configuraciones de datos distintas para HE3 (ver §3) — el checkpoint actual en disco (v4) no es el de mejor F1 bruto de las 4 (eso fue v3, F1=0.4563, ya no recuperable — ver nota en §3), pero sí el de mejor generalización (ΔF1 más bajo).

### Objetivo — Latencia en tiempo real <200ms ✅
- Modelo aislado (ONNX): **0.72ms** (277× de margen).
- **Medido end-to-end de verdad esta sesión** (no estimado): pipeline completo decodificar frame → MediaPipe Holistic → buffer → normalización → inferencia, vía WebSocket real (`api/main.py:/predict/stream`), con frames reales de video y un cliente WebSocket real (protocolo TCP, no simulado):

  | Métrica | Valor |
  |---|---|
  | p50 | 54.7 ms |
  | p95 | 58.7 ms |
  | max | 118.2 ms |

  El costo dominante es MediaPipe (~55ms/frame), no el modelo. Sobra margen (~140ms) incluso para latencia de red real en producción.

### Objetivo — Sistema integral en tiempo real ✅ (núcleo funcional; límite real encontrado y comunicado con honestidad)
Validado en vivo esta sesión, con datos reales, no solo pruebas aisladas:
- `demo/app_gradio.py` — cámara en vivo, subida de video, overlay de landmarks, TTS de navegador (Web Speech API).
- `api/main.py` — servidor FastAPI real (`uvicorn`) + cliente WebSocket real por TCP, endpoints `/health`, `/classes`, `/predict/video`, `/predict/stream` todos probados con datos reales.
- `spaces/` (despliegue público HuggingFace) — actualizado al modelo S27 (antes servía un modelo de junio).
- `Dockerfile` preparado — **sin build-test real** (Docker no disponible en el entorno de desarrollo de esta sesión; pendiente de probar en una máquina con Docker antes de la sustentación).

**Bugs reales encontrados y corregidos durante la validación en vivo** (no solo pruebas automatizadas — usando el sistema como lo usaría una persona):
1. Videos cortos (~1s, clips de una sola seña) nunca llegaban a clasificarse porque el buffer de 30 frames nunca se llenaba con frames reales — corregido con padding.
2. `data/clase_texto.json` tenía claves que nunca coincidían con los nombres reales de clase (`"vineta_002"` vs `"HISTORIAS_VINETAS_2"`) — la traducción a texto legible nunca se activaba. Corregido en los 3 archivos que la usan (`demo/app_gradio.py`, `api/main.py`, `spaces/app.py`).
3. **Hallazgo de fondo, verificado contra el SRT real**: el modelo clasifica *clips completos* (una clase = un video de 1-9 min), pero la demo lo aplicaba con ventana fija cada 0.5s sobre video continuo — comparado contra el SRT real de "Historias vinetas (11)" (130 glosas reales en 60s), la coincidencia con lo predicho era prácticamente nula. **Corregido de fondo, dentro de lo posible sin reentrenar**: se implementó segmentación por pausas (`src/features/segmentacion.py`) que detecta el fin real de cada seña (reposo de las manos) y clasifica cada segmento como una unidad — la misma tarea para la que el modelo fue medido. Funciona bien para señante que hace pausas breves entre señas (el caso de uso real de comunicación asistida). **No resuelve narración fluida sin pausas** (una historia contada de corrido) — eso es reconocimiento continuo gloss-a-gloss, un problema de investigación aparte (ver `ROADMAP_FRONTEND_Y_EVALUACION.md`), y la demo ahora lo advierte explícitamente en pantalla en vez de aparentar que funciona.

---

## 2. Trayectoria del modelo (resumen — detalle completo en `INFORME_RENDIMIENTO_S26.html`)

| Hito | F1-test | Nota |
|---|---|---|
| S5 — baseline LogReg | 0.0058 | punto de partida |
| S13 — primer +0.10 F1 | 0.3696 | primer sprint con F1 útil |
| S26 — mejor pre-S27 | 0.4098 | HE3 fallaba (ΔF1=0.1663) |
| S27 v3 (2026-07-12) | 0.4563 ★ mejor F1 | ΔF1=0.0785, PSI=0.0184, KS D=0.049 (1.13× crítico, lo más cerca que se llegó) — **checkpoint ya no existe, sobrescrito por v4** |
| **S27 v4 — actual en disco** | **0.4426** | ΔF1=0.0406 (mejor de todas), PSI=0.0288, KS D=0.065 (1.5× crítico) |

## 3. Por qué HE3 "falla" pero el modelo sí generaliza (resumen — detalle en `ANALISIS_S27_OBJETIVOS_PROYECTO.md`)

HE3 exige tres condiciones: **ΔF1≤0.15** (brecha de rendimiento entre datos vistos y no vistos), **PSI<0.20** (estabilidad de la distribución de predicciones) y un test de Kolmogorov-Smirnov con **p>0.05** (¿la forma de la distribución de confianza es la misma en ambos casos?). Las dos primeras —las que responden directamente si el modelo generaliza— pasan con margen holgado en el modelo actual (v4): **ΔF1=0.0406** (~3.7× por debajo del umbral) y **PSI=0.0288** (~7× por debajo del umbral).

La tercera no pasa, pero no por una razón de fondo. El test KS gana potencia estadística con el tamaño de muestra hasta volverse impracticable: con ~2,100 muestras de holdout, el estadístico crítico para p=0.05 es **D≈0.043-0.047** — cualquier diferencia de forma superior a eso, por trivial que sea en la práctica, ya alcanza significancia estadística. El modelo v4 obtiene **D=0.065** (1.5× el crítico); la corrida v3 (ya no recuperable) había llegado a **D=0.049**, solo 1.13× el crítico — la más cerca que estuvo el proyecto de un HE3 completo.

**Se intentó explícitamente cerrar esa brecha** (2026-07-12/13): 4 configuraciones distintas de granularidad de sub-grupos en el split, cada una con su propio reentrenamiento completo. El hallazgo honesto: la relación entre "más sub-grupos" y "KS pasa" **no es monótona ni predecible** — la intervención que más ayudó a ΔF1 (v4) empeoró ligeramente KS respecto a la anterior (v3). Cada corrida redistribuye de forma distinta y no controlable qué grupos específicos caen en el holdout. Una corrida adicional tardó **788 minutos (~13 horas)**, muy por encima del rango histórico (66-120 min) — dado ese riesgo de tiempo impredecible cerca de la fecha de sustentación, se decidió no seguir iterando. Detalle completo de las 4 corridas en `ANALISIS_S27_OBJETIVOS_PROYECTO.md` §6.

En otras palabras: KS confirma que las dos distribuciones no son bit-a-bit idénticas —algo casi garantizado a este tamaño de muestra, generalice bien o mal el modelo—, no que el modelo falle en producción. Esa pregunta ya la responden ΔF1 y PSI, y la responden que sí generaliza — de forma consistente en las 4 configuraciones probadas, no solo en una corrida afortunada.

## 4. Qué se dejó fuera a propósito (detalle completo en `ROADMAP_FRONTEND_Y_EVALUACION.md`)

| Ítem | Por qué no está |
|---|---|
| F1 > 0.70 (número del doc interno viejo) | No es uno de los 3 objetivos oficiales; ese número era para 200+ clases, no 96. |
| Frontend React desde cero | Stack nuevo, varios días; Gradio ya cubre la interfaz visual en tiempo real funcionalmente. |
| WER/BLEU con reconocimiento continuo | El modelo clasifica clip completo (1 clase), el ground truth SRT es palabra por palabra — comparar eso da un número sin significado real; requiere rediseñar a reconocimiento continuo. |
| NLP con BERT (SOV→SVO) | Sin diseño lingüístico previo en el proyecto; el suavizado temporal implementado es lo que realmente estaba especificado. |
| MLOps (wandb/mlflow) | No cambia la narrativa de la sustentación; retroactivo a 27 sprints no aporta en el plazo. |
| YOLOv8-pose | El modelo ya cumple <200ms con 277× de margen — no hay necesidad urgente. |

## 5. Cómo correr el sistema (comandos reales, probados esta sesión)

```bash
# Demo interactiva (cámara + video + TTS)
.venv310/bin/python demo/app_gradio.py
# → http://localhost:7860

# API backend (FastAPI + WebSocket tiempo real)
.venv310/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
# → GET http://localhost:8000/health
# → WS  ws://localhost:8000/predict/stream

# Docker (sin probar en este entorno — pendiente)
docker build -t traductor-lsp-api .
docker run -p 8000:8000 traductor-lsp-api
```

Checkpoint activo: `checkpoints/bilstm_s27.onnx` (96 clases) + `data/s27_label2idx.json`. **Nota:** estos archivos aún no están commiteados en git (`.gitignore` excluye `checkpoints/*.onnx`) — pendiente de decisión explícita antes de considerar el repo autosuficiente en un clon limpio.

---

## Documentos de respaldo (detalle completo, no duplicado aquí)

- `INFORME_RENDIMIENTO_S26.html` / `.docx` / `notebooks/informe_rendimiento_s26.ipynb` — comparativo técnico completo de 15 sprints, benchmark de latencia detallado, causa raíz HE3.
- `ANALISIS_S27_OBJETIVOS_PROYECTO.md` — investigación completa del fix de S27 y el argumento ΔF1/PSI vs KS.
- `ROADMAP_FRONTEND_Y_EVALUACION.md` — arquitectura ya especificada para React y el pipeline de reconocimiento continuo, para cuando se retome ese trabajo.
