"""generar_figuras_entrega_final.py — 2 figuras nuevas para
TESIS_FINAL_S15.md / ENTREGA_FINAL_SEMANA15.md, con datos reales ya
medidos esta semana (data/s33_wer_resultados.json + los conteos de los 4
slices del Anexo). Mismo estilo y paleta que scripts/generar_figuras_sustentacion.py
para mantener consistencia visual con el resto de figuras del proyecto.

  - fig_wer_progresion.png : WER real S31→S32→S33 vs producción (5 videos)
  - fig_slices_ic.png      : proporciones de acierto de los slices 1 y 3,
                             con intervalo de confianza Wilson 95%
"""
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
OUT = ROOT / "data" / "sustentacion_figs"
OUT.mkdir(exist_ok=True)

TEAL = "#0e7c78"
TEAL_DARK = "#075c59"
AMBER = "#b3791f"
BAD = "#c2384a"
GOOD = "#1f8f5f"
GREY = "#8398a0"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.edgecolor": "#425055",
    "axes.labelcolor": "#182225",
    "text.color": "#182225",
    "xtick.color": "#425055",
    "ytick.color": "#425055",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def wilson(k, n, z=1.96):
    p = k / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    adj = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return p, max(0, (centre - adj) / denom), min(1, (centre + adj) / denom)


# ── 1. WER progresión ──────────────────────────────────────────────────────
print("Generando fig_wer_progresion.png...")
with open(ROOT / "data" / "s33_wer_resultados.json", encoding="utf-8") as f:
    wer = json.load(f)

labels = ["S31\n(aislado)", "S32\n(combinado,\nsplit único)", "S33\n(combinado,\nKFold completo)", "Producción\n(v4+S29)"]
keys = ["s31", "s32", "s33", "produccion"]
medias = [wer["wer_medio"][k] for k in keys]
colores = [GREY, "#5fb3af", TEAL_DARK, AMBER]

fig, ax = plt.subplots(figsize=(7.2, 4.6))
bars = ax.bar(labels, medias, color=colores, width=0.6, zorder=3)
ax.axhline(1.0, color=BAD, linestyle="--", linewidth=1.2, zorder=2)
ax.text(3.45, 1.03, "WER=1.0 (tantos errores\ncomo palabras reales)", color=BAD,
        fontsize=8.5, va="bottom", ha="right")

for b, v in zip(bars, medias):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.3f}",
            ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#182225")

ax.set_ylabel("WER medio (5 videos narrativos nunca vistos)")
ax.set_title("WER real — narración continua: 3 corridas consecutivas vs. producción", fontsize=12, pad=14)
ax.set_ylim(0, max(medias) * 1.22)
ax.grid(axis="y", color="#e2e8ea", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "fig_wer_progresion.png", dpi=160)
plt.close(fig)


# ── 2. Slices con IC Wilson ─────────────────────────────────────────────────
print("Generando fig_slices_ic.png...")
slices = [
    ("Abecedario — video/cámara\n(cuerpo completo)", 0, 6),
    ("Abecedario — imagen\n(post-fix R14)", 19, 24),
    ("Videos largos\nsin segmentar (top-3)", 1, 7),
    ("Clips ya aislados\n(cortos, top-3)", 3, 4),
]

fig, ax = plt.subplots(figsize=(8.6, 4.4))
ys = np.arange(len(slices))[::-1]
for y, (label, k, n) in zip(ys, slices):
    p, lo, hi = wilson(k, n)
    color = BAD if p < 0.5 else (AMBER if p < 0.7 else GOOD)
    ax.plot([lo * 100, hi * 100], [y, y], color=color, linewidth=3, solid_capstyle="round", zorder=3)
    ax.scatter([p * 100], [y], color=color, s=70, zorder=4, edgecolor="white", linewidth=1.2)
    ax.text(hi * 100 + 3, y, f"{p*100:.1f}%  (n={n})", va="center", fontsize=9.5, color="#182225")

ax.set_yticks(ys)
ax.set_yticklabels([s[0] for s in slices], fontsize=9.5)
ax.set_xlim(0, 118)
ax.set_xlabel("Tasa de acierto (%) — punto = medición real, línea = IC95 Wilson")
ax.set_title("Slices problemáticos — proporción de acierto con intervalo de confianza", fontsize=11.5, pad=14)
ax.axvline(50, color="#c8d2d4", linestyle=":", linewidth=1)
ax.grid(axis="x", color="#e2e8ea", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "fig_slices_ic.png", dpi=160)
plt.close(fig)

print("✅ fig_wer_progresion.png y fig_slices_ic.png en data/sustentacion_figs/")
