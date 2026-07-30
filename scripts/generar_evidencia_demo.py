"""generar_evidencia_demo.py — Corre casos reales a través del código real de
la demo (demo/app_gradio.py, sin simulación) y guarda el frame anotado con
landmarks + el resultado de predicción real de cada caso, para usar como
evidencia gráfica en la tesis (Anexo C.13).

Casos:
  1. Imagen real de la letra "N" del abecedario (data/Abecedario/n/)
  2. Video real de la seña "DOS" (en vocabulario) — resultado real: sin detección
  3. Video real de la seña "CHAU" (fuera de vocabulario) — resultado real: clasificación errónea
  4. Video narrativo real "Historias vinetas (2)" — resultado real: traducción parcial

Salida: data/evidencia_demo/*.png (frame anotado) + data/evidencia_demo/resultados.json
(texto de la predicción real de cada caso) + fig_evidencia_demo.png (grilla compuesta
para insertar en la tesis).
"""
import json
import sys
import pathlib

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
EVID = ROOT / "data" / "evidencia_demo"
EVID.mkdir(exist_ok=True)
FIGS = ROOT / "data" / "sustentacion_figs"

import demo.app_gradio as demo_mod

resultados = {}

# ── Caso 1: imagen real de letra "N" ──────────────────────────────────────────
print("Caso 1: imagen letra N...")
img_bgr = cv2.imread(str(ROOT / "data/Abecedario/n/n (118).jpg"))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
vis, result_md, top3_md, estado_md, _ = demo_mod.process_image(img_rgb)
Image.fromarray(vis).save(EVID / "caso1_imagen_n.png")
resultados["caso1_imagen_n"] = {
    "entrada": "data/Abecedario/n/n (118).jpg (imagen real, letra N)",
    "resultado_md": result_md,
    "top3_md": top3_md,
}
print("  ", result_md.replace("\n", " "))

# ── Caso 2: video real "DOS" ──────────────────────────────────────────────────
print("Caso 2: video DOS...")
demo_mod.historial.clear()
demo_mod.last_result = None
demo_mod.segmentador_cam = demo_mod.SegmentadorPausas()
gen = demo_mod.process_video_streaming(str(ROOT / "data/Glosas/DOS/DOS_1.mp4"))
last = None
for out in gen:
    last = out
vis2 = last[0]
if vis2 is not None:
    Image.fromarray(vis2).save(EVID / "caso2_video_dos.png")
resultados["caso2_video_dos"] = {
    "entrada": "data/Glosas/DOS/DOS_1.mp4 (video real, seña DOS, en vocabulario)",
    "resultado_md": last[1],
    "resumen": last[4],
}
print("  ", last[4])

# ── Caso 3: video real "CHAU" ──────────────────────────────────────────────────
print("Caso 3: video CHAU...")
gen = demo_mod.process_video_streaming(str(ROOT / "data/Glosas/CHAU/CHAU_1.mp4"))
last = None
for out in gen:
    last = out
vis3 = last[0]
if vis3 is not None:
    Image.fromarray(vis3).save(EVID / "caso3_video_chau.png")
resultados["caso3_video_chau"] = {
    "entrada": "data/Glosas/CHAU/CHAU_1.mp4 (video real, seña CHAU, fuera de vocabulario)",
    "resultado_md": last[1],
    "resumen": last[4],
}
print("  ", last[1].replace("\n", " "))

# ── Caso 4: video narrativo real ──────────────────────────────────────────────
print("Caso 4: video narrativo Historias vinetas (2)...")
gen = demo_mod.process_video_streaming(str(ROOT / "data/videos/original/Historias vinetas (2).mp4"))
last = None
for out in gen:
    last = out
vis4 = last[0]
if vis4 is not None:
    Image.fromarray(vis4).save(EVID / "caso4_video_narrativo.png")
resultados["caso4_video_narrativo"] = {
    "entrada": "data/videos/original/Historias vinetas (2).mp4 (video narrativo real)",
    "resultado_md": last[1],
    "resumen": last[4],
}
print("  ", last[4])

with open(EVID / "resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)
print("\n✅ data/evidencia_demo/resultados.json")

# ── Figura compuesta: grilla 2x2 con frame + resultado real ──────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
casos = [
    ("caso1_imagen_n.png", "Imagen real — letra N", "N — 41.1% (correcto)", "#1f8f5f"),
    ("caso2_video_dos.png", "Video real — seña DOS (en vocabulario)", "0 señas detectadas (fallo)", "#c2384a"),
    ("caso3_video_chau.png", "Video real — seña CHAU (fuera de vocabulario)", "\"Igual\" 58% (incorrecto)", "#c2384a"),
    ("caso4_video_narrativo.png", "Video narrativo real — Historias Viñetas (2)", "7 señas / 3045 frames (mayormente \"Igual\")", "#b3791f"),
]
for ax, (fname, titulo, resultado, color) in zip(axes.flat, casos):
    img_path = EVID / fname
    if img_path.exists():
        img = Image.open(img_path)
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, "(sin frame)", ha="center", va="center")
    ax.set_title(titulo, fontsize=10.5, pad=6)
    ax.text(0.5, -0.06, resultado, transform=ax.transAxes, ha="center", va="top",
            fontsize=10.5, fontweight="bold", color=color)
    ax.axis("off")

fig.suptitle("Evidencia real: casos procesados a través del código de producción de la demo",
             fontsize=13, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(FIGS / "fig_evidencia_demo.png", dpi=170)
plt.close(fig)
print("✅ fig_evidencia_demo.png")
