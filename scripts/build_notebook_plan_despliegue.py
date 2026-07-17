"""Construye y ejecuta notebooks/ENTREGABLE_PLAN_DE_DESPLIEGUE_S13.ipynb.

Espeja las 10 secciones del .md, y agrega celdas de código que corren
la suite de tests y el chequeo E2E real (no simulado) como evidencia
ejecutable dentro del propio notebook.
"""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT = ROOT / "notebooks" / "ENTREGABLE_PLAN_DE_DESPLIEGUE_S13.ipynb"

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


md("""# Plan de Despliegue — Traductor LSP → Castellano
**Entregable S13 — antes de la tesis**

| | |
|---|---|
| **Proyecto** | Traductor LSP → Castellano |
| **Fecha** | 2026-07-17 |
| **Responsable** | Juan Calla |
| **Alcance** | Backend FastAPI (`api/main.py`), demo Gradio (`demo/app_gradio.py`), despliegue público (`spaces/`), modelo activo BiLSTM S27 (v4). No introduce componentes nuevos; documenta lo construido y validado esta semana. |

Este notebook no solo documenta el plan — las celdas de código de las secciones 5 y 7 **corren de verdad** contra el sistema real (requiere `.venv310/bin/python -m uvicorn api.main:app` corriendo en `localhost:8000`) como evidencia ejecutable, siguiendo el mismo criterio de esta semana: verificar en vivo, no solo describir.""")

md("""## 1. Resumen ejecutivo

El sistema traduce en tiempo real señas aisladas de un vocabulario de 96 clases LSP a texto en castellano, con latencia end-to-end medida de **p50=54.7ms / p95=58.7ms** (umbral objetivo: 200ms) y **F1-macro=0.4426** con generalización verificada a señantes no vistos (ΔF1=0.0406, 3.7× de margen sobre el umbral de HE3). Existen tres superficies funcionales validadas con datos reales esta semana: demo interactiva (`demo/app_gradio.py`, cámara + video + TTS), backend API con WebSocket (`api/main.py`), y despliegue público en HuggingFace Spaces (`spaces/`).

El sistema **no está listo para producción sin trabajo adicional**: sin build de Docker verificado, checkpoint no versionado en git, sin observabilidad estructurada, CORS abierto (`allow_origins=["*"]`), y el backend API todavía no incorpora las mejoras de calidad que sí tiene la demo (R9, §10). "Validado" en §2 significa que no se cae, no que iguale la calidad de la demo.

**Límite de alcance conocido:** el sistema reconoce señas aisladas, no narración continua ni gramática conectada — limitación de capacidad del modelo, no de infraestructura.""")

md("""## 2. Arquitectura candidata

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

Rutas alternativas (mismo modelo y módulo de landmarks, sin pasar por `api/main.py`): `demo/app_gradio.py` (Gradio standalone, incluye además segmentación por pausas) y `spaces/app.py` (copia autónoma para HuggingFace Spaces).

> ⚠️ **Divergencia real, encontrada al escribir este plan:** los 3 ajustes de esta semana (segmentación por pausas, filtro de clases narrativas, umbral de confianza calibrado a 0.20) están solo en `demo/` y `spaces/`. `api/main.py` sigue con ventana fija de 30 frames sin filtrar — ver **R9** en §10.

| Componente | Archivo | Estado |
|---|---|---|
| Extracción de landmarks | `src/features/landmarks.py` | ✅ compartido entre `api/` y `demo/` |
| Segmentación por pausas | `src/features/segmentacion.py` | ✅ calibrado con datos reales |
| Modelo | `checkpoints/bilstm_s27.onnx` (3.9MB) | ✅ entrenado — ⚠️ no está en git |
| Backend API | `api/main.py` | ✅ validado con servidor + WS real |
| Demo | `demo/app_gradio.py` | ✅ validado en vivo, `:7860` |
| Deploy público | `spaces/` | ✅ actualizado al modelo S27 |
| Contenedor | `Dockerfile` | ⚠️ preparado, sin build-test real |""")

md("""## 3. Contratos I/O

**`GET /health`**
```json
{"status": "ok", "model_ready": true, "device": "cpu", "n_classes": 96}
```

**`GET /classes`**
```json
{"classes": ["A", "AHORA", "APRENDER", "..."], "total": 96}
```

**`POST /predict/video`** (multipart/form-data, campo `file`)
```json
// Response 200 — capturado en vivo con `make test-video`
{
  "clase": "DIEZ",
  "texto_castellano": "DIEZ",
  "confidence": 0.0984,
  "latency_ms": 4662.9,
  "top3": [
    {"clase": "DIEZ", "confidence": 0.0984},
    {"clase": "ORIGINAL", "confidence": 0.0908},
    {"clase": "IGUAL", "confidence": 0.0593}
  ]
}
```
Nota: `api/main.py` no filtra por confianza (R9) — se devuelve tal cual aunque la confianza sea baja.

**`WS /predict/stream`**
```
Cliente  → {"frame": "<base64 JPEG>", "include_landmarks": false}
Servidor → {"status": "buffering", "frames_collected": 12, "frames_needed": 30}
Servidor → {"clase": "DOS", "confidence": 0.149, "latency_ms": 46.4, "top3": [...]}
```
Latencia real medida (cliente WebSocket real): p50=54.7ms, p95=58.7ms, max=118.2ms.""")

md("""## 4. Reproducibilidad

| Ítem | Estado real |
|---|---|
| Entorno | `.venv310/` (Python 3.10.20 — torch, backend) y `.venv311/` (Python 3.11.15 — MediaPipe, Gradio) |
| Seeds | `SEED=42` fijo en todos los splits. **No determinístico al 100%** — `WeightedRandomSampler` + augmentation sin seed por worker; confirmado que v3/v4 no reproducen bit-a-bit |
| Lockfile | ✅ `requirements.lock.txt` — 198 paquetes con versión exacta |
| Makefile | ✅ comandos validados: `install`, `run-demo`, `run-api`, `health`, `test`, `test-video`, `docker-build` |""")

code("""# Verificación real del lockfile (no simulada)
from pathlib import Path
lock = Path("../requirements.lock.txt")
lines = [l for l in lock.read_text().splitlines() if l.strip() and not l.startswith("#")]
print(f"requirements.lock.txt: {len(lines)} paquetes pineados")
print("Ejemplos:", lines[:5])""")

md("""## 5. E2E en limpio

```bash
# 1. Entorno
python3.10 -m venv .venv310 && .venv310/bin/pip install -r requirements-api.txt

# 2. Modelo (no está en git — copiar manualmente, ver R1)

# 3. Levantar backend
.venv310/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# 4. make health
# 5. make test-video
```

**Criterio de éxito:** `/health` responde `model_ready:true`, y `/predict/video` devuelve una clase válida en menos de 10s.

La siguiente celda ejecuta el criterio de éxito contra el servidor real si está corriendo en `localhost:8000` (`make run-api` en otra terminal). Si el servidor no está arriba, lo indica explícitamente en vez de simular una respuesta.""")

code("""import requests

try:
    r = requests.get("http://localhost:8000/health", timeout=3)
    print("GET /health →", r.status_code, r.json())
except requests.exceptions.ConnectionError:
    print("Servidor no está corriendo en localhost:8000.")
    print("Para verificar en vivo: .venv310/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000")""")

md("""## 6. Observabilidad

- `logs/runs.csv` — historial de entrenamientos, no de inferencia en producción.
- `print()` a stdout — sin niveles, sin structured logging, sin destino persistente.
- `latency_ms` se devuelve por request pero no se agrega ni persiste.

**Gap real:** sin métricas agregadas, sin alertas, sin dashboard. Recomendación mínima: `logging` estándar + contador de latencias antes de producción.""")

md("""## 7. Validación & tests

**Implementado y verificado (`tests/`, 6/6 pasan):**
- `tests/test_smoke.py` — health, classes, formato de respuesta.
- `tests/test_golden.py` — 3 clips reales (AHORA, TÚ, PROTEÍNA) en top-3. Elegidos con evidencia: de 7 videos de viñeta completa solo 1/7 acertó — los clips cortos sí son representativos de la tarea real (F1=0.4426).
- `tests/test_websocket_contract.py` — protocolo buffering→predicción, manejo de error.

La siguiente celda corre la suite real de pytest (misma que corre `make test`).""")

code("""import subprocess
result = subprocess.run(
    [".venv310/bin/python", "-m", "pytest", "tests/", "-v"],
    cwd="..", capture_output=True, text=True, timeout=120,
)
print(result.stdout[-2500:])
if result.returncode != 0:
    print(result.stderr[-1500:])
print("Exit code:", result.returncode)""")

md("""## 8. Seguridad & config

| Ítem | Estado | Riesgo |
|---|---|---|
| CORS | `allow_origins=["*"]` | Alto — abierto a cualquier origen |
| Secretos/.env | No existe, todo hardcodeado | Bajo hoy, no escala |
| Límite de upload | No valida tamaño | Medio — memoria/tiempo |
| Rate limiting | No existe | Medio — sin protección DoS |
| Autenticación | Endpoints públicos | Alto si se expone fuera de red controlada |""")

md("""## 9. Hoja de ruta a Docker/API

| Tarea | Responsable | Fecha | Estado |
|---|---|---|---|
| Tests (smoke+golden+WS) | Juan Calla | — | ✅ Hecho |
| Lockfile congelado | Juan Calla | — | ✅ Hecho |
| Portar mejoras a `api/main.py` (R9) | Juan Calla | Antes de integrar | ⚠️ Pendiente |
| Build de Docker real | Juan Calla | Próxima sesión c/Docker | ⚠️ Bloqueado |
| Checkpoint a git | Juan Calla | Antes del próximo deploy | ⚠️ Pendiente |
| `requirements-api.txt` exacto | Juan Calla | Antes de producción | Pendiente |
| Logging estructurado | Juan Calla | 1 semana | Pendiente |
| Restringir CORS/límites | Juan Calla | Antes de exponer | Pendiente |
| CI (pytest en cada push) | Juan Calla | 1 semana | Pendiente |""")

md("""## 10. Riesgos & mitigaciones

Todos reales, encontrados esta semana durante el desarrollo — no hipotéticos.

| Riesgo | Encontrado | Mitigación / Rollback |
|---|---|---|
| R1 — Checkpoint fuera de git | `.gitignore` excluye `checkpoints/*.onnx` | Commitear con excepción, o release aparte |
| R2 — Checkpoint v3 perdido | v4 sobrescribió el mejor KS (D=0.049) | Versionar por nombre de corrida |
| R3 — Entrenamiento impredecible | 788min vs. 66-120min históricos | +10× margen; máquina dedicada |
| R4 — Umbral mal calibrado | 0.30 ocultaba aciertos reales | Corregido a 0.20 con datos reales |
| R5 — Docker sin probar | Sin acceso a Docker esta semana | Probar antes de deploy en contenedor |
| R6 — Regresión silenciosa | `model_complexity=0` rompió detección | Validar contra videos de referencia, no solo velocidad |
| R7 — CORS abierto | `allow_origins=["*"]` | Restringir antes de exponer |
| R8 — Sin reconocimiento continuo | Verificado contra SRT real | Fuera de alcance, investigación aparte |
| R9 — API sin las mejoras de la semana | Segmentación/filtro/umbral solo en demo | Portar antes de integrar backend real |
| R10 — Videos largos rinden peor | 1/7 en viñetas completas vs 3/4 en clips cortos | No usar videos largos como demo del API |

**Rollback general:** todos los cambios de esta semana están en el working tree sin commitear — `git diff`/`git checkout` revierte cualquier cambio individualmente. El modelo v4 es el único artefacto no versionado (R1).""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3.10 (.venv310)", "language": "python", "name": "venv310"},
    "language_info": {"name": "python", "version": "3.10.20"},
}

OUT.parent.mkdir(exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Notebook construido (sin ejecutar): {OUT}")
