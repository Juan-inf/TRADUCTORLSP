"""Genera diagramas reales (no texto monoespaciado) para:
  - Figura 1: Arquitectura del pipeline de traducción LSP a texto castellano
  - Figura 3: Contratos I/O de los endpoints del sistema

Usa la misma paleta que data/sustentacion_figs/ (teal/ámbar) para
mantener consistencia visual con el resto de figuras del documento.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).parent.parent
OUT = ROOT / "data" / "sustentacion_figs"
OUT.mkdir(exist_ok=True)

TEAL = "#0e7c78"
TEAL_DARK = "#075c59"
TEAL_SOFT = "#e2f2f0"
AMBER = "#b3791f"
AMBER_SOFT = "#f7ecd9"
GOOD = "#1f8f5f"
GOOD_SOFT = "#e3f4ea"
INK = "#182225"
INK_SOFT = "#425055"
MUTED = "#6b7a80"
BORDER = "#c8d2d4"
MONO = "Courier New"

plt.rcParams.update({
    "font.family": "sans-serif",
    "text.color": INK,
})


def rounded_box(ax, xy, w, h, fc, ec, text_lines, text_color=INK, fontsize=10.5,
                 title=None, title_color=None, mono=False, align="center"):
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                          linewidth=1.3, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(box)
    x0, y0 = xy
    cx = x0 + w / 2
    n_lines = len(text_lines) + (1 if title else 0)
    line_h = h / (n_lines + 1)
    ty = y0 + h - line_h * 0.65
    if title:
        ax.text(cx, ty, title, ha="center", va="center", fontsize=fontsize + 1.5,
                 fontweight="bold", color=title_color or text_color, zorder=3)
        ty -= line_h * 1.15
    fam = MONO if mono else "sans-serif"
    for line in text_lines:
        ax.text(cx, ty, line, ha="center", va="center", fontsize=fontsize,
                 color=text_color, family=fam, zorder=3)
        ty -= line_h


def arrow(ax, x, y0, y1, label=None, label_dx=0.12):
    a = FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=16,
                         linewidth=1.6, color=INK_SOFT, zorder=1)
    ax.add_patch(a)
    if label:
        ax.text(x + label_dx, (y0 + y1) / 2, label, ha="left", va="center",
                 fontsize=9, color=MUTED, style="italic")


# ═══════════════════════════════════════════════════════════════════════
# FIGURA 1 — Arquitectura del pipeline
# ═══════════════════════════════════════════════════════════════════════
print("Generando fig_arquitectura_pipeline.png...")
fig, ax = plt.subplots(figsize=(7.2, 8.6))
ax.set_xlim(0, 6)
ax.set_ylim(2.6, 12.65)
ax.axis("off")

bw, bx = 4.6, 0.7
gap = 0.55

y = 11.7
rounded_box(ax, (bx, y), bw, 0.85, "#ffffff", INK_SOFT,
            ["navegador o cámara"], title="Cliente", fontsize=10.5)
arrow(ax, bx + bw / 2, y, y - gap, "captura frame (JPEG base64)")

y -= gap + 1.05
rounded_box(ax, (bx, y), bw, 1.05, TEAL_SOFT, TEAL,
            ["/health · /classes · /predict/video", "WS /predict/stream"],
            title="api/main.py — FastAPI", title_color=TEAL_DARK, fontsize=10)
arrow(ax, bx + bw / 2, y, y - gap)

y -= gap + 0.95
rounded_box(ax, (bx, y), bw, 0.95, "#ffffff", INK_SOFT,
            ["543 puntos → 75 usados", "landmarks [75, 3] por fotograma"],
            title="MediaPipe Holistic", fontsize=10)
arrow(ax, bx + bw / 2, y, y - gap)

y -= gap + 1.05
rounded_box(ax, (bx, y), bw, 1.05, "#ffffff", INK_SOFT,
            ["kp_seq_to_features()  →  [30, 150]", "normalize_sample()    →  z-score"],
            title="src/features/landmarks.py", fontsize=10)
arrow(ax, bx + bw / 2, y, y - gap)

y -= gap + 1.05
rounded_box(ax, (bx, y), bw, 1.05, AMBER_SOFT, AMBER,
            ["checkpoints/bilstm_s27.onnx", "96 clases · latencia < 1 ms"],
            title="ONNXPredictor", title_color=AMBER, fontsize=10)
arrow(ax, bx + bw / 2, y, y - gap - 0.3,
      "{clase, texto_castellano,\nconfidence, latency_ms, top3}", label_dx=0.12)

y -= gap + 0.3 + 0.85
rounded_box(ax, (bx, y), bw, 0.85, GOOD_SOFT, GOOD,
            ["respuesta JSON / WebSocket"], title="Cliente", title_color=GOOD, fontsize=10.5)

ax.text(3, y - 0.55,
        "Costo dominante: MediaPipe Holistic (~55 ms/frame). Inferencia ONNX < 1 ms.\n"
        "Segmentación por pausas activa en demo/ y spaces/; pendiente en api/main.py (R9).",
        ha="center", va="top", fontsize=8.7, color=MUTED, style="italic")

plt.tight_layout()
plt.savefig(OUT / "fig_arquitectura_pipeline.png", dpi=180, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════
# FIGURA 3 — Contratos I/O
# ═══════════════════════════════════════════════════════════════════════
print("Generando fig_contratos_io.png...")
fig, ax = plt.subplots(figsize=(9.5, 11.5))
ax.set_xlim(0, 9.5)
ax.set_ylim(0, 11.5)
ax.axis("off")


def method_badge(ax, x, y, method, color):
    w = 0.62 if method == "GET" else (0.72 if method == "POST" else 0.55)
    box = FancyBboxPatch((x, y), w, 0.32, boxstyle="round,pad=0.02,rounding_size=0.06",
                          linewidth=0, facecolor=color, zorder=4)
    ax.add_patch(box)
    ax.text(x + w / 2, y + 0.16, method, ha="center", va="center",
             fontsize=9.5, fontweight="bold", color="white", zorder=5)
    return x + w


def endpoint_panel(ax, x, y, w, h, method, method_color, path, desc, json_lines,
                    status_note=None):
    panel = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                            linewidth=1.1, edgecolor=BORDER, facecolor="#fbfcfc", zorder=1)
    ax.add_patch(panel)
    mx = method_badge(ax, x + 0.22, y + h - 0.45, method, method_color)
    ax.text(mx + 0.15, y + h - 0.29, path, ha="left", va="center",
             fontsize=11.5, fontweight="bold", color=INK, family=MONO, zorder=5)
    ax.text(x + 0.22, y + h - 0.7, desc, ha="left", va="top",
             fontsize=9, color=MUTED, style="italic", zorder=5)
    # JSON box
    json_top = y + h - 1.05
    json_h = json_top - (y + 0.25)
    jbox = FancyBboxPatch((x + 0.22, y + 0.25), w - 0.44, json_h,
                           boxstyle="round,pad=0.015,rounding_size=0.05",
                           linewidth=0.8, edgecolor=BORDER, facecolor="#0f1720", zorder=2)
    ax.add_patch(jbox)
    ty = json_top - 0.14
    for line in json_lines:
        ax.text(x + 0.36, ty, line, ha="left", va="top", fontsize=8.3,
                 family=MONO, color="#d7e4e2", zorder=5)
        ty -= 0.155
    if status_note:
        ax.text(x + w - 0.22, y + 0.08, status_note, ha="right", va="bottom",
                 fontsize=7.8, color=MUTED, style="italic", zorder=5)


# Panel 1: GET /health
endpoint_panel(ax, 0.3, 8.85, 8.9, 2.35, "GET", GOOD, "/health",
               "Verifica el estado del servicio y del modelo de inferencia.",
               ['{',
                '  "status": "ok",',
                '  "model_ready": true,',
                '  "device": "cpu",',
                '  "n_classes": 96',
                '}'],
               "Respuesta capturada en vivo")

# Panel 2: GET /classes
endpoint_panel(ax, 0.3, 6.35, 8.9, 2.1, "GET", GOOD, "/classes",
               "Devuelve el vocabulario completo de clases soportadas.",
               ['{',
                '  "classes": ["A", "AHORA", "APRENDER", "..."],',
                '  "total": 96',
                '}'],
               "Respuesta capturada en vivo")

# Panel 3: POST /predict/video
endpoint_panel(ax, 0.3, 3.05, 8.9, 3.0, "POST", TEAL_DARK, "/predict/video",
               "Recibe un video (MP4/AVI/MOV) y devuelve la predicción para la seña detectada.",
               ['{',
                '  "clase": "DIEZ", "texto_castellano": "DIEZ",',
                '  "confidence": 0.0984, "latency_ms": 4662.9,',
                '  "top3": [',
                '    {"clase": "DIEZ", "confidence": 0.0984},',
                '    {"clase": "ORIGINAL", "confidence": 0.0908},',
                '    {"clase": "IGUAL", "confidence": 0.0593}',
                '  ]',
                '}'],
               "200 OK · 422 si no procesa · 503 si el modelo no cargó")

# Panel 4: WS /predict/stream
endpoint_panel(ax, 0.3, 0.3, 8.9, 2.55, "WS", "#6a4fb3", "/predict/stream",
               "Recibe frames en tiempo real; acumula hasta completar el buffer y predice.",
               ['Cliente  -> {"frame": "<base64 JPEG>"}',
                'Servidor -> {"status": "buffering", "frames_collected": 12,',
                '             "frames_needed": 30}',
                'Servidor -> {"clase": "DOS", "confidence": 0.149,',
                '             "latency_ms": 46.4, "top3": [...]}'],
               "p50=54.7ms · p95=58.7ms · max=118.2ms (cliente real)")

plt.tight_layout()
plt.savefig(OUT / "fig_contratos_io.png", dpi=180, bbox_inches="tight")
plt.close()

print(f"\nGuardado en {OUT}/")
for name in ["fig_arquitectura_pipeline.png", "fig_contratos_io.png"]:
    fp = OUT / name
    print(f"  {name}  ({fp.stat().st_size/1024:.0f} KB)")
