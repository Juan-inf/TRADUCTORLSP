"""generar_html_entrega_final.py — TESIS_FINAL_S15.md / ENTREGA_FINAL_SEMANA15.md
→ HTML autónomo (mismo sistema de diseño teal/dark-mode que
RESULTADOS_PRESENTACION_S15.html), usando python-markdown para manejar
tablas y bloques de código (```) de forma nativa.

Uso:
    python3 scripts/generar_html_entrega_final.py TESIS_FINAL_S15
    python3 scripts/generar_html_entrega_final.py ENTREGA_FINAL_SEMANA15
"""
import base64
import re
import sys
from pathlib import Path
import markdown
from markdown.extensions.tables import TableExtension
from markdown.extensions.fenced_code import FencedCodeExtension

ROOT = Path(__file__).parent.parent

if len(sys.argv) < 2:
    print("Uso: generar_html_entrega_final.py <NOMBRE_SIN_EXTENSION>")
    sys.exit(1)

NOMBRE = sys.argv[1]
MD_IN = ROOT / f"{NOMBRE}.md"
HTML_OUT = ROOT / f"{NOMBRE}.html"

CSS = """
:root{
  --bg:#f4f6f7; --surface:#ffffff; --surface-2:#eef1f2;
  --ink:#182225; --ink-soft:#425055; --muted:#6b7a80; --border:#d7dfe1;
  --accent:#0e7c78; --accent-ink:#075c59; --accent-soft:#e2f2f0;
  --amber:#b3791f; --amber-soft:#f7ecd9;
  --good:#1f8f5f; --good-soft:#e3f4ea;
  --bad:#c2384a; --bad-soft:#fbe6e9;
  --code-bg:#1e2530; --code-fg:#e2e8f0;
  --shadow: 0 1px 2px rgba(20,30,32,.06), 0 8px 24px -12px rgba(20,30,32,.18);
}
:root[data-theme="dark"]{
  --bg:#10161a; --surface:#171f24; --surface-2:#1d262b; --ink:#e7edee; --ink-soft:#b9c6c9;
  --muted:#8398a0; --border:#2a363c; --accent:#3fb3a9; --accent-ink:#8fe0d6; --accent-soft:#173330;
  --amber:#d6a34a; --amber-soft:#31281a; --good:#4fbf87; --good-soft:#153427; --bad:#e2697b; --bad-soft:#3a1c22;
  --code-bg:#0d1117; --code-fg:#d6e0e8;
  --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 24px -12px rgba(0,0,0,.6);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#10161a; --surface:#171f24; --surface-2:#1d262b; --ink:#e7edee; --ink-soft:#b9c6c9;
    --muted:#8398a0; --border:#2a363c; --accent:#3fb3a9; --accent-ink:#8fe0d6; --accent-soft:#173330;
    --amber:#d6a34a; --amber-soft:#31281a; --good:#4fbf87; --good-soft:#153427; --bad:#e2697b; --bad-soft:#3a1c22;
    --code-bg:#0d1117; --code-fg:#d6e0e8;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 24px -12px rgba(0,0,0,.6);
  }
}
*{box-sizing:border-box;}
html,body{margin:0;padding:0;}
body{
  background:var(--bg); color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  font-size:15.5px; line-height:1.65; -webkit-font-smoothing:antialiased;
}
.wrap{ max-width:980px; margin:0 auto; padding:0 24px 96px; }
.masthead{ border-bottom:1px solid var(--border); padding:40px 0 28px; margin-bottom:32px; }
.eyebrow{
  font-family:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
  font-size:11.5px; letter-spacing:.12em; text-transform:uppercase; color:var(--accent);
  display:flex; align-items:center; gap:8px; margin-bottom:14px;
}
.eyebrow .dot{ width:6px;height:6px;border-radius:50%;background:var(--accent); }
h1{
  font-family:"Iowan Old Style","Palatino Linotype","Book Antiqua",Georgia,serif;
  font-weight:600; font-size:clamp(24px,3.6vw,32px); line-height:1.18; letter-spacing:-.01em;
  margin:0 0 12px; text-wrap:balance; color:var(--ink);
}
.subtitle{ color:var(--ink-soft); font-size:15px; max-width:78ch; margin:0 0 18px; }
.meta-row{
  display:flex; flex-wrap:wrap; gap:8px 20px;
  font-family:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
  font-size:12px; color:var(--muted);
}
.meta-row b{ color:var(--ink-soft); font-weight:600; }
#main h2{
  font-family:"Iowan Old Style","Palatino Linotype","Book Antiqua",Georgia,serif;
  font-size:22px; font-weight:600; margin:40px 0 12px; color:var(--ink); text-wrap:balance;
  padding-bottom:6px; border-bottom:1px solid var(--border);
}
#main h3{ font-family:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
  font-size:14px; font-weight:600; margin:24px 0 8px; color:var(--accent-ink); }
p{ color:var(--ink-soft); max-width:82ch; }
.table-wrap{ overflow-x:auto; border:1px solid var(--border); border-radius:10px; margin:14px 0; box-shadow:var(--shadow); }
table{ border-collapse:collapse; width:100%; min-width:480px; background:var(--surface); }
thead th{ background:var(--surface-2); text-align:left; font-size:11px; letter-spacing:.05em; text-transform:uppercase;
  color:var(--muted); font-weight:600; padding:9px 13px; border-bottom:1px solid var(--border); }
tbody td{ padding:9px 13px; border-bottom:1px solid var(--border); font-size:13px; color:var(--ink-soft); vertical-align:top; }
tbody td:first-child{ color:var(--ink); font-weight:600; }
tbody tr:last-child td{ border-bottom:none; }
tbody tr:hover td{ background:var(--surface-2); }
code{ font-family:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace; background:var(--surface-2);
  padding:1px 6px; border-radius:5px; font-size:.88em; color:var(--ink-soft); }
pre{ background:var(--code-bg); color:var(--code-fg); padding:16px 18px; border-radius:10px;
  overflow-x:auto; font-size:12.5px; line-height:1.6; margin:14px 0; box-shadow:var(--shadow); }
pre code{ background:none; color:inherit; padding:0; font-size:1em; }
blockquote{ margin:16px 0; padding:12px 18px; border-left:4px solid var(--accent); background:var(--accent-soft);
  border-radius:0 8px 8px 0; color:var(--ink); font-style:italic; font-size:14px; }
hr{ border:none; border-top:1px solid var(--border); margin:8px 0; }
strong{ color:var(--ink); }
img{ max-width:100%; height:auto; display:block; margin:16px auto; border:1px solid var(--border);
  border-radius:10px; box-shadow:var(--shadow); background:#fff; }
p:has(> img:only-child){ text-align:center; }
"""

md = markdown.Markdown(extensions=[TableExtension(), FencedCodeExtension(), "sane_lists"])
raw = MD_IN.read_text(encoding="utf-8")
lines = raw.splitlines()

# separar portada (titulo/subtitulo/nota) del cuerpo — igual criterio que el conversor docx
i = 0
while i < len(lines) and not lines[i].startswith("## 1."):
    i += 1
portada_lines = lines[:i]
cuerpo_lines = lines[i:]

titulo = NOMBRE.replace("_", " ")
subtitulo = ""
for l in portada_lines:
    if l.startswith("> "):
        subtitulo = l[2:].strip()
        break

body_html = md.convert("\n".join(cuerpo_lines))


def _embed_image(match):
    """Reemplaza src="ruta/relativa.png" por data:image/png;base64,... — HTML
    autónomo, sin dependencias externas ni rutas relativas rotas."""
    src = match.group(1)
    img_path = ROOT / src
    if img_path.exists():
        data = base64.b64encode(img_path.read_bytes()).decode()
        ext = img_path.suffix.lower().lstrip(".")
        mime = f"image/{'jpeg' if ext == 'jpg' else ext}"
        return f'src="data:{mime};base64,{data}"'
    return match.group(0)


body_html = re.sub(r'src="([^"]+\.(?:png|jpg|jpeg|gif|svg))"', _embed_image, body_html)

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{titulo} — Traductor LSP → Castellano</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
  <div class="masthead">
    <div class="eyebrow"><span class="dot"></span>Entrega final · Semana 15</div>
    <h1>{titulo} — Traductor LSP → Castellano</h1>
    <p class="subtitle">{subtitulo}</p>
    <div class="meta-row">
      <span><b>Fecha</b> 2026-07-24</span>
      <span><b>Autor</b> Juan Calla</span>
      <span><b>Modelo activo</b> Ensemble BiLSTM S27(v4) + S29 · F1=0.4424</span>
    </div>
  </div>
  <div id="main">
    {body_html}
  </div>
</div>
</body>
</html>
"""

print(f"Escribiendo {HTML_OUT.name}...")
HTML_OUT.write_text(html, encoding="utf-8")
size_kb = HTML_OUT.stat().st_size / 1024
print(f"OK — {HTML_OUT.name} ({size_kb:.0f} KB) en {HTML_OUT}")
