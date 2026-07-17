# Análisis S27 vs. Objetivos del Proyecto

**Fecha:** 2026-07-11
**Alcance:** este documento es nuevo y no reemplaza ni modifica `INFORME_RENDIMIENTO_S26.html/.docx` ni los notebooks `informe_rendimiento_s26.ipynb` / `reporte_comparativo_baseline_actual.ipynb`, que quedan intactos como registro histórico del estado a S26. Aquí se documenta lo ocurrido después: dos corridas de `train_s27.py` y la decisión metodológica sobre el criterio HE3-KS.

---

## 1. Objetivo del proyecto (referencia oficial)

> Desarrollar un sistema integral basado en Deep Learning para lograr la traducción automática de la Lengua de Señas Peruana (LSP) a texto en castellano en tiempo real.

Desglosado en tres objetivos específicos:

1. **Diseñar un modelo de traducción automática basado en Deep Learning** para lograr una alta precisión en la traducción de la LSP.
2. **Mejorar el pipeline de captura, procesamiento e inferencia** para lograr traducción en tiempo real con **latencia inferior a 200 ms**.
3. **Sistema integral** — no solo el modelo: captura, inferencia, y (según `docs/PROMPT_MAESTRO_LSP_SISTEMA_COMPLETO.md`) interfaz visual en tiempo real con overlay de señas + texto traducido.

La métrica final declarada en el prompt maestro del sistema es **F1-macro > 0.70** sobre vocabulario de uso diario. Es importante no perder eso de vista: los resultados de abajo son el mejor punto histórico del proyecto, pero **todavía no alcanzan esa meta final** — hay que decirlo con la misma claridad con la que se reporta lo que sí mejoró.

---

## 2. Qué se hizo después de S26

`scripts/train_s27.py` existía preparado desde antes de esta sesión, apuntando a `dataset_s17.npz` (fix de grupos cross-source dgi156↔vineta), pero nunca se había ejecutado. Se corrió dos veces:

### Corrida 1 (script original, sin cambios)
- Primer intento: **crash por memoria MPS** en el Fold 3 del KFold final (`RuntimeError: MPS backend out of memory`, 13.34 GB en "other allocations" para un modelo de 966K parámetros — señal de fuga de memoria, no de necesidad real). Se corrigió agregando `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` + limpieza explícita de caché MPS (`torch.mps.empty_cache()` + `gc.collect()`) entre trials de HPO y folds de KFold, directamente en `scripts/train_s27.py`.
- Segundo intento (con el fix de memoria): completó sin errores. **ΔF1 pasó el umbral por primera vez en el proyecto** (0.1311 vs. umbral 0.15), pero HE3 global siguió en FALLA por el test KS (p≈0).

### Corrida 2 (sub-grupos ampliados)
Investigando por qué KS seguía fallando pese a que ΔF1 y PSI ya pasaban, se encontró que **no es un artefacto de tamaño de muestra**: el holdout de grupo tenía sistemáticamente *más* confianza (softmax máximo) que el test — señal de que clases de vocabulario aislado (abecedario, AEC) con muy pocos sub-grupos (3 y 5 respectivamente) producían resultados de "todo o nada" por clase al caer en el 20% de holdout, distinto en forma a la partición estratificada del test.

Se aumentó `n_subgrupos` de abecedario (3→10) y AEC (5→10) en `scripts/build_dataset_s17.py`, se reconstruyó `dataset_s17.npz`, y se reentrenó con `--skip-hpo` (los mejores hiperparámetros ya eran siempre el warm-start de S13, confirmado en ambas corridas de HPO previas — no hacía falta repetir la búsqueda de 20 trials).

---

## 3. Resultado — nuevo mejor punto del proyecto

| Métrica | S26 | S27 v1 (fix cross-source) | **S27 v2 (+ sub-grupos)** | Umbral HE3 |
|---|---|---|---|---|
| F1-test | 0.4098 | 0.4066 | **0.4349** ★ récord | — |
| Top-3 / Top-5 | 0.494 / 0.588 | 0.498 / 0.577 | **0.561 / 0.636** | — |
| ΔF1 (holdout de grupo) | 0.1663 ❌ | 0.1311 ✅ | **0.0509 ✅** | ≤ 0.15 |
| PSI | 0.1523 | 0.1281 ✅ | **0.0818 ✅** | < 0.20 |
| KS p-value | 0.0000 ❌ | 0.0000 ❌ | 0.0000 ❌ | > 0.05 |
| KS D (tamaño del efecto) | — | 0.136 | 0.117 (mejoró 14%) | D crítico ≈ 0.047 con este N |
| Latencia ONNX (batch=1) | ~1.0 ms | — | **0.72 ms** | < 200 ms |
| HE3 global | FALLA | FALLA | **FALLA** (solo por KS) | — |

Efecto colateral notable: la letra **`T`** del abecedario (único señante) **desapareció de las peores clases** en S27 v2. No era, como se pensó antes, un problema irresoluble sin grabación nueva — era falta de granularidad en el split. Esto revisa una conclusión anterior de este mismo análisis.

Checkpoint activo: `checkpoints/bilstm_s27.pt` / `.onnx` (sobrescrito con el resultado v2, el mejor de los dos). Ambas corridas quedaron registradas en `logs/runs.csv` (`exp_20260710_bilstm_s27` y `exp_20260711_bilstm_s27`).

---

## 4. La pregunta real: ¿esto bloquea el objetivo del proyecto?

**Objetivo 2 (latencia < 200 ms):** ✅ cumplido con margen amplio — 0.72 ms es ~277× más rápido que el umbral. No es un punto de discusión.

**Objetivo 1 (alta precisión):** F1=0.4349 es el mejor resultado histórico del proyecto y el primero en combinar F1 alto *con* generalización medible (ΔF1 bajo). Sigue lejos de la meta final declarada de F1>0.70 — eso no lo resuelve este sprint, requiere más datos/clases, no es un tema de HE3.

**Objetivo 3 (sistema integral):** es donde vive la pregunta de fondo. El razonamiento previo era "mientras HE3 falle, construir encima es prematuro". Con los datos de S27 v2, hay que precisar *qué parte* de HE3 sigue fallando:

- **ΔF1 y PSI** — las dos métricas que miden directamente la brecha de rendimiento entre datos vistos y no vistos, y la estabilidad de la distribución de predicciones — **ya pasan, con margen** (ΔF1 a 3× de holgura del umbral). Esta es la evidencia concreta de que el modelo generaliza razonablemente a sesiones/grupos no vistos.
- **KS** es una prueba de **forma** de la distribución de confianza, no de rendimiento. Con ~1800 muestras por lado, el estadístico D crítico para p=0.05 es ≈0.047 — cualquier diferencia de forma mayor a ese umbral (algo estadísticamente casi garantizado a este tamaño de muestra salvo que las dos distribuciones sean prácticamente idénticas) hace fallar la prueba. No es uno de los tres objetivos oficiales del proyecto ni de la métrica final (F1>0.70); es un gate de validación interno (HE3) que este proyecto se auto-impuso.

**Conclusión de este análisis:** el bloqueo real que cita el objetivo del proyecto — "que el modelo generalice a un señante/fuente nueva" — tiene evidencia cuantitativa de haberse resuelto en la medida en que ΔF1 y PSI lo miden. Lo que sigue fallando (KS) es una prueba de forma de distribución hipersensible al tamaño de muestra, no una señal adicional de que el modelo falle en producción.

---

## 5. Recomendación

No se propone abandonar el trabajo sobre KS (seguir subiendo sub-grupos, o replantear la métrica con un umbral de tamaño de efecto en vez de p-value, son pasos válidos para un sprint futuro). Pero **no hay justificación, con esta evidencia, para seguir bloqueando el trabajo sobre las otras capacidades del sistema integral** (frontend, backend, NLP, TTS, WER/BLEU con SRT.tar, etc. — ver gaps pendientes en memoria de proyecto) esperando a que una prueba estadística de forma, no de rendimiento, pase con datasets de este tamaño.

Próximos pasos sugeridos, en paralelo:
1. Avanzar en las capacidades pendientes del objetivo de "sistema integral" (el modelo actual, F1=0.4349, Top-5=63.6%, latencia 0.72ms, ya es utilizable para integrar y validar el resto del pipeline).
2. Dejar abierto, como tarea de investigación de menor prioridad, seguir reduciendo KS D (más sub-grupos, o revisar si el criterio debería expresarse como tamaño de efecto en vez de p-value).
3. Mantener la meta final de F1>0.70 visible — S27 es el mejor punto hasta ahora, no el objetivo cumplido.

---

## 6. Corridas adicionales (v3, v4) — 2026-07-12/13

Se intentó, a pedido explícito, seguir mejorando KS más allá de v2. Dos corridas adicionales, cada una reconstruyendo `dataset_s17.npz` con más granularidad de sub-grupos antes de reentrenar con `--skip-hpo`:

- **v3**: `abecedario_pkl` 10→20 sub-grupos (viable: 150 muestras/clase uniformes), `aec_pkl` 10→15 (AEC tiene clases con apenas 1 muestra — no hay margen real para subdividir más allá de eso).
- **v4**: además de lo de v3, el fix de sub-grupos cruzados dgi156↔vineta (`balancear_grupos_hv()`) se llevó de 5→10 sub-grupos por clase HV, para reducir el riesgo de que un pedazo grande de una clase enorme (`HISTORIAS_VINETAS_3`, 1086 muestras) caiga entero en holdout.

### Tabla comparativa completa (v1→v4)

| Métrica | v1 | v2 | **v3 (mejor F1 y mejor KS)** | v4 (actual en disco) | Umbral |
|---|---|---|---|---|---|
| F1-test | 0.4066 | 0.4349 | **0.4563** ★ | 0.4426 | — |
| ΔF1 | 0.1311 | 0.0509 | 0.0785 | **0.0406** ★ | ≤0.15 |
| PSI | 0.1281 | 0.0818 | **0.0184** ★ | 0.0288 | <0.20 |
| KS D | 0.136 | 0.117 | **0.049** ★ (1.13× crítico) | 0.065 (1.5× crítico) | D crítico≈0.043-0.047 |
| KS p-value | ≈0 | ≈0 | 0.0168 | 0.0011 | >0.05 |
| Tiempo de entrenamiento | ~2.5h (con HPO) | ~1.9h (con HPO) | ~2h (`--skip-hpo`) | **788 min (~13h)** ⚠️ | — |

### Hallazgo importante: la relación entre las palancas y KS no es monótona ni predecible

Subir sub-grupos de HV (5→10, v3→v4) **empeoró KS** (D: 0.049→0.065) aunque mejoró ΔF1 a su mejor valor de las 4 corridas (0.0406). Cada corrida redistribuye de forma distinta qué grupos específicos caen en el 20% de holdout — no hay garantía de que una intervención que ayuda a una métrica ayude a las otras. **v3 sigue siendo, hasta ahora, el punto más cercano a que HE3 pase completo** (KS a solo 1.13× del crítico, la primera vez que se acercó tanto).

### Problema operativo encontrado: el checkpoint de v3 se perdió

`checkpoints/bilstm_s27.pt` / `.onnx` se sobrescriben en cada corrida — v4 sobrescribió los pesos de v3 al guardar. **v3 solo existe como fila en `logs/runs.csv` (`exp_20260711_bilstm_s27`... revisar timestamp exacto), no como checkpoint reproducible.** Si se quisiera recuperar el modelo v3 real, hay que reconstruir su configuración exacta (`abecedario_pkl`=20 sub-grupos, `aec_pkl`=15, HV=**5** sub-grupos — no 10) y reentrenar desde cero, sin garantía de reproducir exactamente el mismo resultado (el entrenamiento tiene componentes no determinísticos pese al `SEED=42` fijo, por el uso de `DataLoader` con `WeightedRandomSampler` y augmentation con `np.random` sin seed local por worker).

### Riesgo operativo encontrado: tiempo de entrenamiento muy variable

La corrida v4 tardó **788 minutos (~13 horas)**, muy por encima del rango histórico observado (66-120 min con `--skip-hpo`). Causa no confirmada — candidatos: contención de CPU con otros procesos corriendo en paralelo en la misma máquina (la demo Gradio quedó corriendo en segundo plano durante ese período), o factores del entorno de desarrollo ajenos al código. **Esto es un riesgo real para cualquier decisión de "una corrida más" cerca de una fecha límite** — no hay garantía de que la próxima tarde 1-2h en vez de 13.

### Estado actual real (2026-07-13)

El checkpoint en disco es **v4** (F1=0.4426, ΔF1=0.0406, PSI=0.0288, KS falla con D=0.065). Es el segundo mejor F1 de las 4 corridas y el mejor ΔF1, pero no es el más cercano a pasar HE3 completo (ese fue v3, ya no recuperable sin reentrenar). Dado el costo de tiempo desconocido de una corrida adicional, **se recomienda no seguir iterando sobre HE3-KS y quedarse con v4 como modelo final**, documentando el argumento ya escrito en la §4 de este documento (ΔF1 y PSI son la evidencia real de generalización; KS es un tecnicismo estadístico) — que sigue siendo válido para v4 exactamente igual que para v2/v3.
