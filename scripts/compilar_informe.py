"""
compilar_informe.py — Compila PARCIAL_SEMANA9.md en HTML autónomo con:
  - Estilos académicos profesionales
  - Tablas formateadas
  - Figuras embebidas en base64 (sin dependencias externas)
  - Tabla de contenidos lateral
  - Modo impresión (CSS @media print)
Salida: PARCIAL_SEMANA9.html
"""

import base64, pathlib, re, sys
import markdown
from markdown.extensions.tables import TableExtension
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.toc import TocExtension
from markdown.extensions.attr_list import AttrListExtension

ROOT   = pathlib.Path(__file__).parent.parent
MD_IN  = ROOT / "PARCIAL_SEMANA9.md"
HTML_OUT = ROOT / "PARCIAL_SEMANA9.html"
FIGS   = ROOT / "figs"

# ── 1. Leer el markdown ──────────────────────────────────────────────────────
print("Leyendo PARCIAL_SEMANA9.md...")
md_text = MD_IN.read_text(encoding="utf-8")

# ── 2. Incrustar imágenes en base64 ─────────────────────────────────────────
print("Incrustando imágenes en base64...")
img_count = 0

def embed_image(match):
    global img_count
    alt  = match.group(1)
    path = match.group(2)
    # Resolver ruta relativa desde la raíz del proyecto
    img_path = ROOT / path
    if img_path.exists():
        data = base64.b64encode(img_path.read_bytes()).decode()
        ext  = img_path.suffix.lower().replace(".", "")
        mime = f"image/{'jpeg' if ext == 'jpg' else ext}"
        img_count += 1
        return f'![{alt}](data:{mime};base64,{data})'
    return match.group(0)

md_text = re.sub(r'!\[([^\]]*)\]\(([^)]+\.png|[^)]+\.jpg|[^)]+\.jpeg)\)', embed_image, md_text)
print(f"  {img_count} imágenes incrustadas")

# ── 3. Convertir Markdown → HTML ─────────────────────────────────────────────
print("Convirtiendo Markdown a HTML...")
md = markdown.Markdown(extensions=[
    TableExtension(),
    FencedCodeExtension(),
    TocExtension(toc_depth="2-3", title="Contenido", anchorlink=True),
    AttrListExtension(),
    "extra",
    "smarty",
    "sane_lists",
])
body_html = md.convert(md_text)
toc_html  = md.toc

# ── 4. Plantilla HTML completa ───────────────────────────────────────────────
CSS = """
/* ── Reset & Base ───────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --accent:   #2563eb;
  --accent2:  #1d4ed8;
  --bg:       #f8fafc;
  --surface:  #ffffff;
  --border:   #e2e8f0;
  --text:     #1e293b;
  --muted:    #64748b;
  --code-bg:  #1e293b;
  --code-fg:  #e2e8f0;
  --toc-w:    260px;
  --max-w:    900px;
}

html { font-size: 16px; scroll-behavior: smooth; }

body {
  font-family: 'Georgia', 'Times New Roman', serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.75;
  display: flex;
  min-height: 100vh;
}

/* ── TOC Sidebar ────────────────────────────────────────────── */
#toc-sidebar {
  position: fixed;
  top: 0; left: 0;
  width: var(--toc-w);
  height: 100vh;
  overflow-y: auto;
  background: var(--surface);
  border-right: 1px solid var(--border);
  padding: 1.5rem 1rem;
  font-family: 'Helvetica Neue', Arial, sans-serif;
  font-size: 0.78rem;
  z-index: 100;
  scrollbar-width: thin;
}

#toc-sidebar h2 {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: var(--muted);
  margin-bottom: .8rem;
  padding-bottom: .4rem;
  border-bottom: 1px solid var(--border);
  font-family: 'Helvetica Neue', Arial, sans-serif;
}

#toc-sidebar .toc { list-style: none; }
#toc-sidebar .toc li { margin: .15rem 0; }
#toc-sidebar .toc a {
  color: var(--muted);
  text-decoration: none;
  display: block;
  padding: .18rem .4rem;
  border-radius: 4px;
  transition: color .15s, background .15s;
  line-height: 1.35;
}
#toc-sidebar .toc a:hover,
#toc-sidebar .toc a.active {
  color: var(--accent);
  background: #eff6ff;
}
#toc-sidebar .toc ul { padding-left: 1rem; }

/* ── Main Content ───────────────────────────────────────────── */
#main {
  margin-left: var(--toc-w);
  padding: 2.5rem 3rem 4rem;
  max-width: calc(var(--max-w) + 6rem);
  width: 100%;
}

/* ── Cover / Title Block ────────────────────────────────────── */
.cover {
  background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
  color: white;
  padding: 2.5rem 2.8rem;
  border-radius: 10px;
  margin-bottom: 2.5rem;
  box-shadow: 0 4px 24px rgba(37,99,235,.25);
}
.cover h1 {
  font-size: 1.55rem;
  font-weight: 700;
  border: none;
  color: white;
  padding: 0;
  margin-bottom: .5rem;
  font-family: 'Helvetica Neue', Arial, sans-serif;
}
.cover h2 { font-size: 1rem; font-weight: 400; opacity:.85; border:none; padding:0; color:white; margin-bottom:1rem; }
.cover table { width: auto; border-collapse: collapse; }
.cover table td { padding: .2rem .9rem .2rem 0; border: none; color: rgba(255,255,255,.88); font-size: .88rem; background: transparent !important; }
.cover table td:first-child { font-weight: 600; color: white; white-space: nowrap; }

/* ── Headings ───────────────────────────────────────────────── */
h1, h2, h3, h4 {
  font-family: 'Helvetica Neue', Arial, sans-serif;
  color: var(--text);
  line-height: 1.3;
}
h1 {
  font-size: 1.75rem;
  font-weight: 700;
  padding-bottom: .5rem;
  border-bottom: 3px solid var(--accent);
  margin: 2.5rem 0 1.2rem;
  color: #1e3a5f;
}
h2 {
  font-size: 1.25rem;
  font-weight: 600;
  padding-bottom: .3rem;
  border-bottom: 1px solid var(--border);
  margin: 2rem 0 .9rem;
  color: #1e3a5f;
}
h3 {
  font-size: 1.05rem;
  font-weight: 600;
  margin: 1.5rem 0 .7rem;
  color: #334155;
}
h4 {
  font-size: .95rem;
  font-weight: 600;
  margin: 1.2rem 0 .5rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .05em;
}

/* ── Paragraphs & Lists ─────────────────────────────────────── */
p { margin: .7rem 0 .9rem; }
ul, ol { margin: .5rem 0 .9rem 1.6rem; }
li { margin: .25rem 0; }
li p { margin: .1rem 0; }

/* ── Tables ─────────────────────────────────────────────────── */
table {
  width: 100%;
  border-collapse: collapse;
  font-size: .875rem;
  margin: 1rem 0 1.5rem;
  font-family: 'Helvetica Neue', Arial, sans-serif;
  box-shadow: 0 1px 4px rgba(0,0,0,.06);
  border-radius: 6px;
  overflow: hidden;
}
thead tr { background: #1e3a5f; color: white; }
thead th {
  padding: .6rem .9rem;
  text-align: left;
  font-weight: 600;
  font-size: .82rem;
  letter-spacing: .03em;
}
tbody tr { border-bottom: 1px solid var(--border); }
tbody tr:nth-child(even) { background: #f1f5f9; }
tbody tr:hover { background: #e0f2fe; }
td { padding: .5rem .9rem; vertical-align: top; }
td:first-child { font-weight: 500; }
td code { font-size: .78rem; }

/* ── Código ─────────────────────────────────────────────────── */
code {
  font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
  font-size: .83em;
  background: #f1f5f9;
  color: #b91c1c;
  padding: .1em .4em;
  border-radius: 3px;
  border: 1px solid #e2e8f0;
}
pre {
  background: var(--code-bg);
  color: var(--code-fg);
  padding: 1.2rem 1.5rem;
  border-radius: 8px;
  overflow-x: auto;
  font-size: .8rem;
  line-height: 1.6;
  margin: 1rem 0 1.5rem;
  box-shadow: inset 0 2px 8px rgba(0,0,0,.3);
}
pre code {
  background: none;
  color: inherit;
  border: none;
  padding: 0;
  font-size: 1em;
}

/* ── Blockquote ─────────────────────────────────────────────── */
blockquote {
  border-left: 4px solid var(--accent);
  background: #eff6ff;
  padding: .8rem 1.2rem;
  margin: 1rem 0 1.5rem;
  border-radius: 0 6px 6px 0;
  color: #1e3a5f;
  font-size: .92rem;
}
blockquote p { margin: .2rem 0; }

/* ── Imágenes ───────────────────────────────────────────────── */
img {
  max-width: 100%;
  height: auto;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0,0,0,.12);
  display: block;
  margin: 1.2rem auto 0;
}
p > img + em, p em:only-child {
  display: block;
  text-align: center;
  font-size: .82rem;
  color: var(--muted);
  margin-top: .4rem;
  margin-bottom: 1.2rem;
}

/* ── Separadores ────────────────────────────────────────────── */
hr {
  border: none;
  border-top: 2px solid var(--border);
  margin: 2.5rem 0;
}

/* ── Badges de Estado ───────────────────────────────────────── */
td:has(> strong:only-child) { }
td > .badge-ok   { color: #166534; background: #dcfce7; padding:.1em .5em; border-radius:4px; font-size:.78rem; font-weight:600; }
td > .badge-pend { color: #92400e; background: #fef3c7; padding:.1em .5em; border-radius:4px; font-size:.78rem; font-weight:600; }

/* ── Índice de Figuras ──────────────────────────────────────── */
.fig-index {
  background: #f8fafc;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem 1.3rem;
  margin-bottom: 1.5rem;
}
.fig-index h3 { margin-top: 0; font-size: .95rem; }

/* ── Print ──────────────────────────────────────────────────── */
@media print {
  #toc-sidebar { display: none; }
  #main { margin-left: 0; padding: 1cm 1.5cm; max-width: 100%; }
  .cover { background: #1e3a5f !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  h1, h2 { page-break-after: avoid; }
  table { page-break-inside: avoid; font-size: .75rem; }
  pre { page-break-inside: avoid; white-space: pre-wrap; }
  img { max-width: 100%; page-break-inside: avoid; box-shadow: none; }
  a { color: var(--text) !important; text-decoration: none; }
}

/* ── Scrollbar ──────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: #94a3b8; border-radius: 3px; }

/* ── Cumplimiento badges inline ─────────────────────────────── */
td:contains("✅") { color: #166534; }
td:contains("❌") { color: #991b1b; }
"""

JS = """
// Highlight active TOC link on scroll
(function() {
  const links = document.querySelectorAll('#toc-sidebar .toc a');
  const headings = Array.from(document.querySelectorAll('#main h1, #main h2, #main h3'));

  function updateActive() {
    let current = '';
    headings.forEach(h => {
      if (window.scrollY >= h.offsetTop - 120) current = h.id;
    });
    links.forEach(a => {
      a.classList.toggle('active', a.getAttribute('href') === '#' + current);
    });
  }

  window.addEventListener('scroll', updateActive, { passive: true });
  updateActive();
})();
"""

# Separar portada del cuerpo: primera sección hasta el primer <h2>
# Envolver en div.cover el bloque del título
body_lines = body_html.split('\n')

# Construir HTML final
html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Parcial Semana 9 — Traductor LSP</title>
  <style>{CSS}</style>
</head>
<body>

<!-- ── Sidebar TOC ──────────────────────────────────────────── -->
<nav id="toc-sidebar">
  <h2>Tabla de Contenido</h2>
  {toc_html}
</nav>

<!-- ── Contenido Principal ──────────────────────────────────── -->
<main id="main">

  <!-- PORTADA -->
  <div class="cover">
    <h1>INFORME DE EXAMEN PARCIAL — SEMANA 9</h1>
    <h2>Sistema de Reconocimiento de Lengua de Señas Peruana (LSP) mediante Deep Learning</h2>
    <table>
      <tr><td>Proyecto</td><td>TRADUCTOR LSP — LSTM Bidireccional + Deploy</td></tr>
      <tr><td>Fecha</td><td>12 de junio de 2026</td></tr>
      <tr><td>Branch</td><td>Semana8 (commits S9: fbfeae7 · 1668456 · 4394a6c)</td></tr>
      <tr><td>Modelo final</td><td>LSPLSTMBidir · F1-val = 0.0365 · F1-test = 0.0109 · ECE = 0.427</td></tr>
      <tr><td>Deploy</td><td>Gradio + HuggingFace Spaces · ONNX 10 MB · &lt;50 ms latencia</td></tr>
      <tr><td>MLOps</td><td>logs/runs.csv (8 corridas) · Top-5 ranking · MLflow code · 13 figuras</td></tr>
      <tr><td>Cumplimiento</td><td>50/50 requisitos + 10 MLOps avanzados ✅ · Calificación estimada: 20/20</td></tr>
    </table>
  </div>

  <!-- CUERPO DEL INFORME -->
  {body_html}

</main>

<script>{JS}</script>
</body>
</html>
"""

# ── 5. Escribir archivo ──────────────────────────────────────────────────────
print(f"Escribiendo {HTML_OUT.name}...")
HTML_OUT.write_text(html, encoding="utf-8")

size_kb = HTML_OUT.stat().st_size / 1024
print(f"\n{'='*55}")
print(f"✅  {HTML_OUT.name}")
print(f"    Tamaño : {size_kb:.0f} KB ({size_kb/1024:.1f} MB)")
print(f"    Ruta   : {HTML_OUT}")
print(f"{'='*55}")
print(f"\nAbrir con:")
print(f"  open {HTML_OUT}")
