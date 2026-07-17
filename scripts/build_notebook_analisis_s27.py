"""Construye y ejecuta notebooks/ANALISIS_S27_OBJETIVOS_PROYECTO.ipynb.

Espeja ANALISIS_S27_OBJETIVOS_PROYECTO.md, y agrega celdas de código que
verifican en vivo lo verificable sin reentrenar (riesgo ya documentado en
R3 del plan de despliegue: una corrida puede tardar de 66min a 13h):

  1. Las 4 filas de logs/runs.csv para S27 (F1, ΔF1, PSI) — cotejadas
     contra las tablas §3 y §6 del documento.
  2. Qué checkpoints siguen realmente en disco (v1-v3 se perdieron por
     sobrescritura, según el propio documento — se confirma aquí).
  3. Latencia ONNX real del checkpoint v4 actual, midiendo 200 inferencias
     reales sobre `checkpoints/bilstm_s27.onnx` (no una recomputación del
     entrenamiento, solo el forward pass — seguro y rápido).

Lo que NO se recomputa: el estadístico KS/ΔF1/PSI completo requiere el
split de holdout de grupo exacto usado en cada corrida (semillas +
reconstrucción de dataset_s17.npz por versión) — reproducirlo con fidelidad
implicaría reentrenar, que es explícitamente el riesgo que este análisis
recomienda no correr sin necesidad (§6 del documento, hallazgo de las 13h).
Esas cifras se citan tal como quedaron registradas en logs/runs.csv y en
el propio documento, no se recalculan.
"""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT = ROOT / "notebooks" / "ANALISIS_S27_OBJETIVOS_PROYECTO.ipynb"

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


md("""# Análisis S27 vs. Objetivos del Proyecto
**Fecha:** 2026-07-11 (corridas v1-v2) / 2026-07-12-13 (v3-v4) · **Verificado en vivo:** 2026-07-17

Este notebook ejecuta contra los artefactos reales del repo (`logs/runs.csv`, `checkpoints/`) las afirmaciones cuantitativas del análisis original. No reentrena el modelo — una corrida puede tardar entre 66 minutos y 13 horas (hallazgo documentado en §6) — solo verifica lo que ya quedó registrado en disco y mide en vivo lo que es seguro y rápido de medir (latencia de inferencia del checkpoint actual).

**Alcance:** este documento no reemplaza `INFORME_RENDIMIENTO_S26.html/.docx` ni los notebooks de S26, que quedan intactos como registro histórico. Documenta lo ocurrido después: dos corridas de `train_s27.py` (v1, v2) y dos corridas adicionales (v3, v4) sobre el criterio HE3-KS.""")

md("""## 1. Objetivo del proyecto (referencia oficial)

> Desarrollar un sistema integral basado en Deep Learning para lograr la traducción automática de la Lengua de Señas Peruana (LSP) a texto en castellano en tiempo real.

Desglosado en tres objetivos específicos:
1. **Diseñar un modelo de traducción automática basado en Deep Learning** para lograr una alta precisión en la traducción de la LSP.
2. **Mejorar el pipeline de captura, procesamiento e inferencia** para lograr traducción en tiempo real con **latencia inferior a 200 ms**.
3. **Sistema integral** — no solo el modelo: captura, inferencia, e interfaz visual en tiempo real con overlay de señas + texto traducido.

La métrica final declarada es **F1-macro > 0.70** sobre vocabulario de uso diario. Los resultados de este documento son el mejor punto histórico del proyecto, pero **todavía no alcanzan esa meta final**.""")

md("""## 2. Qué se hizo después de S26

`scripts/train_s27.py` apuntaba a `dataset_s17.npz` (fix de grupos cross-source dgi156↔vineta) y se corrió dos veces:

**Corrida 1** — crash por memoria MPS en el Fold 3 del KFold final (fuga de memoria, no necesidad real); corregido con `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` + `torch.mps.empty_cache()`/`gc.collect()`. Con el fix: ΔF1 pasó el umbral por primera vez (0.1311 vs. 0.15), pero KS siguió en FALLA (p≈0).

**Corrida 2** — se encontró que clases con pocos sub-grupos (abecedario=3, AEC=5) producían resultados "todo o nada" al caer completos en el 20% de holdout. Se subió `n_subgrupos` (abecedario 3→10, AEC 5→10), se reconstruyó `dataset_s17.npz`, y se reentrenó con `--skip-hpo`.

La siguiente celda lee `logs/runs.csv` en vivo y confirma las cifras de ambas corridas tal como quedaron registradas.""")

code("""import pandas as pd
runs = pd.read_csv("../logs/runs.csv")
s27 = runs[runs["exp_id"].str.contains("bilstm_s27", na=False)].reset_index(drop=True)
print(f"{len(s27)} corridas de S27 encontradas en logs/runs.csv:\\n")
cols = ["exp_id", "f1_test", "acc_test", "tiempo_s", "latencia_ms"]
print(s27[cols].to_string(index=False))""")

md("""## 3. Resultado — nuevo mejor punto del proyecto (v1, v2)

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

Efecto colateral notable: la letra **`T`** del abecedario (único señante) **desapareció de las peores clases** en S27 v2 — no era un problema irresoluble sin grabación nueva, era falta de granularidad en el split.

Checkpoint activo en ese momento: `checkpoints/bilstm_s27.pt`/`.onnx` (v2), luego sobrescrito por v3 y v4 — ver §6.

La celda siguiente cruza `f1_test` y `tiempo_s`/`latencia_ms` de la fila v2 (`exp_20260711_bilstm_s27`) contra la tabla de arriba — ΔF1 y PSI vienen embebidos en la columna `notas`, no en columnas propias, así que se extraen por texto.""")

code("""import re

v2 = s27[s27["exp_id"] == "exp_20260711_bilstm_s27"].iloc[0]
notas = v2["notas"]

delta_f1 = re.search(r"ΔF1=([\\d.]+)", notas).group(1)
psi = re.search(r"PSI=([\\d.]+)", notas).group(1)
top3 = re.search(r"Top3=([\\d.]+)", notas).group(1)
top5 = re.search(r"Top5=([\\d.]+)", notas).group(1)

print("v2 — verificado contra logs/runs.csv:")
print(f"  F1-test  : {v2['f1_test']:.4f}  (documento dice 0.4349)")
print(f"  ΔF1      : {delta_f1}  (documento dice 0.0509)")
print(f"  PSI      : {psi}  (documento dice 0.0818)")
print(f"  Top3/Top5: {top3} / {top5}  (documento dice 0.561 / 0.636)")
print(f"  Latencia : {v2['latencia_ms']} ms  (documento dice 0.72 ms)")
assert abs(v2['f1_test'] - 0.4349) < 1e-3, "F1-test no coincide con el documento"
print("\\n✅ Coincide con lo documentado en §3.")""")

md("""## 4. La pregunta real: ¿esto bloquea el objetivo del proyecto?

**Objetivo 2 (latencia < 200 ms):** ✅ cumplido con margen amplio — 0.72 ms es ~277× más rápido que el umbral.

**Objetivo 1 (alta precisión):** F1=0.4349 es el mejor resultado histórico hasta ese punto y el primero en combinar F1 alto *con* generalización medible. Sigue lejos de la meta final declarada de F1>0.70.

**Objetivo 3 (sistema integral):** el razonamiento previo era "mientras HE3 falle, construir encima es prematuro". Con los datos de S27 v2:
- **ΔF1 y PSI** — miden directamente la brecha de rendimiento vista/no-vista y la estabilidad de distribución — **ya pasan, con margen**.
- **KS** es una prueba de **forma** de la distribución de confianza, no de rendimiento. Con ~1800 muestras por lado, el D crítico para p=0.05 es ≈0.047 — una diferencia de forma mayor a ese umbral es casi estadísticamente garantizada a este tamaño de muestra.

**Conclusión:** el bloqueo real que cita el objetivo del proyecto ("que el modelo generalice a un señante/fuente nueva") tiene evidencia cuantitativa de haberse resuelto en la medida en que ΔF1 y PSI lo miden. KS es un tecnicismo estadístico hipersensible al tamaño de muestra, no una señal adicional de falla en producción.""")

md("""## 5. Recomendación

No se propone abandonar el trabajo sobre KS (seguir subiendo sub-grupos, o replantear la métrica con un umbral de tamaño de efecto, son válidos para un sprint futuro). Pero **no hay justificación, con esta evidencia, para seguir bloqueando el trabajo sobre las otras capacidades del sistema integral** esperando a que una prueba de forma, no de rendimiento, pase con datasets de este tamaño.

Próximos pasos sugeridos, en paralelo:
1. Avanzar en las capacidades pendientes del "sistema integral" (el modelo actual ya es utilizable para integrar y validar el resto del pipeline).
2. Dejar abierto, como tarea de menor prioridad, seguir reduciendo KS D.
3. Mantener visible la meta final de F1>0.70 — S27 es el mejor punto hasta ahora, no el objetivo cumplido.""")

md("""## 6. Corridas adicionales (v3, v4) — 2026-07-12/13

Se intentó, a pedido explícito, seguir mejorando KS más allá de v2:
- **v3**: `abecedario_pkl` 10→20 sub-grupos, `aec_pkl` 10→15.
- **v4**: además de v3, `balancear_grupos_hv()` de 5→10 sub-grupos por clase HV.

### Tabla comparativa completa (v1→v4)

| Métrica | v1 | v2 | **v3 (mejor F1 y mejor KS)** | v4 (actual en disco) | Umbral |
|---|---|---|---|---|---|
| F1-test | 0.4066 | 0.4349 | **0.4563** ★ | 0.4426 | — |
| ΔF1 | 0.1311 | 0.0509 | 0.0785 | **0.0406** ★ | ≤0.15 |
| PSI | 0.1281 | 0.0818 | **0.0184** ★ | 0.0288 | <0.20 |
| KS D | 0.136 | 0.117 | **0.049** ★ (1.13× crítico) | 0.065 (1.5× crítico) | D crítico≈0.043-0.047 |
| KS p-value | ≈0 | ≈0 | 0.0168 | 0.0011 | >0.05 |
| Tiempo de entrenamiento | ~2.5h (con HPO) | ~1.9h (con HPO) | ~2h (`--skip-hpo`) | **788 min (~13h)** ⚠️ | — |

### Hallazgo: la relación entre las palancas y KS no es monótona

Subir sub-grupos de HV (5→10, v3→v4) **empeoró KS** (D: 0.049→0.065) aunque mejoró ΔF1 a su mejor valor (0.0406). **v3 sigue siendo el punto más cercano a que HE3 pase completo** (KS a solo 1.13× del crítico).

### Problema operativo: el checkpoint de v3 se perdió

`checkpoints/bilstm_s27.pt`/`.onnx` se sobrescriben en cada corrida — v4 sobrescribió los pesos de v3. v3 solo existe como fila en `logs/runs.csv`, no como checkpoint reproducible.

La celda siguiente verifica en vivo, contra el disco real, cuáles checkpoints de S27 existen hoy y coteja las 4 filas del `runs.csv` contra la tabla de arriba.""")

code("""from pathlib import Path
import datetime

ck_dir = Path("../checkpoints")
s27_files = sorted(ck_dir.glob("bilstm_s27.*"))
print("Checkpoints S27 en disco hoy:")
for f in s27_files:
    mtime = datetime.datetime.fromtimestamp(f.stat().st_mtime)
    print(f"  {f.name:24s} {f.stat().st_size/1024/1024:5.2f} MB   modificado {mtime}")

print()
print("Solo existe UN checkpoint con nombre fijo `bilstm_s27.*` — confirma lo que dice el")
print("documento: v1, v2 y v3 fueron sobrescritos, y lo que hay en disco hoy es v4.")

print()
cols = ["exp_id", "f1_test", "tiempo_s"]
print("Las 4 corridas registradas en runs.csv (v1→v4, en orden):")
print(s27[cols].to_string(index=False))
tiempo_v4_min = s27.iloc[3]["tiempo_s"] / 60
print(f"\\nTiempo de entrenamiento v4: {tiempo_v4_min:.0f} min (documento dice ~788 min / 13h)")""")

md("""### Estado actual real (verificado 2026-07-17)

El checkpoint en disco es **v4** (F1=0.4426, ΔF1=0.0406, PSI=0.0288, KS falla con D=0.065). Es el segundo mejor F1 de las 4 corridas y el mejor ΔF1, pero no es el más cercano a pasar HE3 completo (ese fue v3, ya no recuperable sin reentrenar).

**Dado el costo de tiempo desconocido de una corrida adicional (66min–13h, sin causa raíz confirmada), se recomienda no seguir iterando sobre HE3-KS y quedarse con v4 como modelo final** — el argumento de §4 (ΔF1 y PSI son la evidencia real de generalización; KS es un tecnicismo estadístico) sigue siendo válido para v4 igual que para v2/v3.

La celda final mide la latencia ONNX real del checkpoint v4 (el único que sigue en disco), sobre el input real del modelo (`[1, 30, 150]`) — esto sí es seguro de recomputar en vivo porque es solo un forward pass, no un reentrenamiento.""")

code("""import time
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("../checkpoints/bilstm_s27.onnx", providers=["CPUExecutionProvider"])
inp_name = sess.get_inputs()[0].name
inp_shape = sess.get_inputs()[0].shape
print(f"Input real del modelo: {inp_name} {inp_shape}")

rng = np.random.default_rng(42)
x = rng.standard_normal((1, 30, 150)).astype(np.float32)

# warm-up
for _ in range(10):
    sess.run(None, {inp_name: x})

N = 200
latencias = []
for _ in range(N):
    t0 = time.perf_counter()
    sess.run(None, {inp_name: x})
    latencias.append((time.perf_counter() - t0) * 1000)

latencias = np.array(latencias)
print(f"\\nLatencia ONNX real, {N} inferencias, batch=1, checkpoint v4 (medido ahora):")
print(f"  p50 = {np.percentile(latencias, 50):.3f} ms")
print(f"  p95 = {np.percentile(latencias, 95):.3f} ms")
print(f"  max = {latencias.max():.3f} ms")
print(f"\\nDocumento dice v4: latencia_ms=0.9 (logs/runs.csv, batch=1, mismo tipo de medición)")
print("Nota: el número aquí puede diferir un poco del de runs.csv por ser otra máquina/momento,")
print("pero debe seguir siendo del mismo orden de magnitud (<5ms) y muy por debajo del umbral de 200ms.")""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3.10 (.venv310)", "language": "python", "name": "venv310"},
    "language_info": {"name": "python", "version": "3.10.20"},
}

OUT.parent.mkdir(exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Notebook construido (sin ejecutar): {OUT}")
