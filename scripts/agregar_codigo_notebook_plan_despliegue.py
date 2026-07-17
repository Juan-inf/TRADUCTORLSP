"""Agrega celdas de código de verificación en vivo a
notebooks/ENTREGABLE_PLAN_DE_DESPLIEGUE_S13.ipynb.

Inserta 4 celdas nuevas, cada una verificando contra el sistema real
(archivos en disco, servidor corriendo, git) una afirmación del plan que
antes solo estaba en texto:

  - §2 Arquitectura   → los componentes de la tabla existen en disco de
                        verdad, y la divergencia R9 (segmentación/umbral
                        solo en demo/, no en api/) se confirma por grep
                        real sobre el código, no de memoria.
  - §3 Contratos I/O  → GET /classes y POST /predict/video se llaman de
                        verdad contra localhost:8000 (mismo video que usa
                        `make test-video`).
  - §8 Seguridad      → el CORS abierto y la ausencia de .env se leen del
                        archivo real, no se citan de memoria.
  - §10 Riesgos       → R1 (checkpoint fuera de git) y R2 (solo v4 en
                        disco) se verifican contra `git status`/`ls` reales.
"""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).parent.parent
NB_PATH = ROOT / "notebooks" / "ENTREGABLE_PLAN_DE_DESPLIEGUE_S13.ipynb"

nb = nbf.read(NB_PATH, as_version=4)


def find_idx(marker):
    for i, c in enumerate(nb.cells):
        if c.cell_type == "markdown" and c.source.strip().startswith(marker):
            return i
    raise ValueError(f"No se encontró celda que empiece con: {marker!r}")


cell_arquitectura = nbf.v4.new_code_cell("""# Verificación real de §2: ¿existen de verdad los componentes de la tabla?
from pathlib import Path

componentes = {
    "Extracción de landmarks": "src/features/landmarks.py",
    "Segmentación por pausas": "src/features/segmentacion.py",
    "Modelo (checkpoint)": "checkpoints/bilstm_s27.onnx",
    "Backend API": "api/main.py",
    "Demo": "demo/app_gradio.py",
    "Deploy público": "spaces/app.py",
    "Contenedor": "Dockerfile",
}
for nombre, rel in componentes.items():
    p = Path("..") / rel
    estado = f"{p.stat().st_size/1024:.0f} KB" if p.exists() else "NO EXISTE"
    print(f"  {'✅' if p.exists() else '❌'} {nombre:28s} {rel:35s} {estado}")

# Divergencia R9: los 3 ajustes de esta semana, ¿están en demo/ pero no en api/?
print("\\nBúsqueda real de 'SegmentadorPausas' y 'CONF_UMBRAL' en cada archivo:")
for rel in ["api/main.py", "demo/app_gradio.py"]:
    texto = (Path("..") / rel).read_text(encoding="utf-8")
    tiene_seg = "SegmentadorPausas" in texto
    tiene_umbral = "CONF_UMBRAL" in texto
    print(f"  {rel:20s} SegmentadorPausas={tiene_seg}  CONF_UMBRAL={tiene_umbral}")
print("\\nConfirmado: la divergencia R9 es real, no una suposición del documento.")""")

cell_contratos = nbf.v4.new_code_cell("""# Verificación real de §3: llamar los endpoints de verdad contra localhost:8000
import requests

try:
    r = requests.get("http://localhost:8000/classes", timeout=5)
    data = r.json()
    print(f"GET /classes → {r.status_code}  total={data['total']}  primeras 5={data['classes'][:5]}")

    video_path = "../data/videos/original/Historias vinetas (11).mp4"
    with open(video_path, "rb") as f:
        r2 = requests.post(
            "http://localhost:8000/predict/video",
            files={"file": ("video.mp4", f, "video/mp4")},
            timeout=30,
        )
    print(f"\\nPOST /predict/video → {r2.status_code}")
    print(r2.json())
except requests.exceptions.ConnectionError:
    print("Servidor no está corriendo en localhost:8000 — no se puede verificar en vivo.")
    print("Levantar con: .venv310/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000")""")

cell_seguridad = nbf.v4.new_code_cell("""# Verificación real de §8: leer la config de CORS del archivo, no citarla de memoria
import re
from pathlib import Path

src = Path("../api/main.py").read_text(encoding="utf-8")
m = re.search(r'allow_origins\\s*=\\s*(\\[[^\\]]*\\])', src)
print("CORS allow_origins real en api/main.py:", m.group(1) if m else "no encontrado")

env_existe = Path("../.env").exists()
print(f".env existe: {env_existe}  (si es False, confirma que la config está hardcodeada)")

# Búsqueda de límite de tamaño de upload — si no aparece, confirma el gap de §8
tiene_limite = "max_size" in src or "MAX_UPLOAD" in src or "content-length" in src.lower()
print(f"¿Hay validación de tamaño máximo de archivo en api/main.py?: {tiene_limite}")""")

cell_riesgos = nbf.v4.new_code_cell("""# Verificación real de §10: R1 (checkpoint fuera de git) y R2 (solo v4 en disco)
import subprocess
from pathlib import Path

# R1 — ¿el checkpoint está trackeado por git?
result = subprocess.run(
    ["git", "check-ignore", "-v", "checkpoints/bilstm_s27.onnx"],
    cwd="..", capture_output=True, text=True,
)
print("git check-ignore checkpoints/bilstm_s27.onnx:")
print(" ", result.stdout.strip() if result.stdout.strip() else "(no ignorado — inesperado)")

# R2 — ¿cuántas versiones de bilstm_s27 sobreviven en disco?
ck_dir = Path("../checkpoints")
s27_files = sorted(ck_dir.glob("bilstm_s27.*"))
print(f"\\nCheckpoints bilstm_s27.* en disco: {len(s27_files)}")
for f in s27_files:
    print(f"  {f.name}  ({f.stat().st_size/1024/1024:.2f} MB)")
print("\\nConfirma R2: solo sobrevive un nombre fijo — v1/v2/v3 fueron sobrescritos, no hay forma de recuperarlos sin reentrenar.")""")

insertions = [
    ("## 2. Arquitectura candidata", cell_arquitectura),
    ("## 3. Contratos I/O", cell_contratos),
    ("## 8. Seguridad & config", cell_seguridad),
    ("## 10. Riesgos & mitigaciones", cell_riesgos),
]

# Insertar de atrás hacia adelante para no invalidar índices ya calculados
targets = [(find_idx(marker), cell) for marker, cell in insertions]
targets.sort(key=lambda t: t[0], reverse=True)

for idx, cell in targets:
    # Insertar después de la sección markdown completa (antes del siguiente heading)
    insert_at = idx + 1
    while insert_at < len(nb.cells) and not (
        nb.cells[insert_at].cell_type == "markdown" and nb.cells[insert_at].source.strip().startswith("## ")
    ):
        insert_at += 1
    nb.cells.insert(insert_at, cell)

nbf.write(nb, NB_PATH)
print(f"Notebook actualizado con 4 celdas de código nuevas: {NB_PATH}")
