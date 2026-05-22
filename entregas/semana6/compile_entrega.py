"""
Compila ENTREGA_SPRINT_SEMANA6.md → HTML y TXT limpios.
Uso: .venv310/bin/python entregas/semana6/compile_entrega.py
"""
import sys
from pathlib import Path

try:
    import markdown
except ImportError:
    sys.exit("Instalar: pip install markdown")

ROOT     = Path(__file__).parent
MD_PATH  = ROOT / "ENTREGA_SPRINT_SEMANA6.md"
HTML_OUT = ROOT / "ENTREGA_SPRINT_SEMANA6.html"
TXT_OUT  = ROOT / "ENTREGA_SPRINT_SEMANA6_resumen.txt"

md_text = MD_PATH.read_text()

CSS = """
<style>
  body { font-family: -apple-system, 'Segoe UI', Arial, sans-serif;
         max-width: 900px; margin: 40px auto; padding: 0 24px;
         color: #1a1a2e; line-height: 1.65; }
  h1   { color: #0f3460; border-bottom: 3px solid #0f3460; padding-bottom: 8px; }
  h2   { color: #16213e; border-bottom: 1px solid #dde; margin-top: 2em; }
  h3   { color: #1a1a2e; }
  table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.92em; }
  th   { background: #0f3460; color: #fff; padding: 8px 12px; text-align: left; }
  td   { padding: 6px 12px; border-bottom: 1px solid #dde; }
  tr:nth-child(even) td { background: #f4f6f9; }
  code { background: #f0f2f5; padding: 2px 6px; border-radius: 3px;
         font-family: 'Fira Code', 'Courier New', monospace; font-size: 0.88em; }
  pre  { background: #1a1a2e; color: #e2e8f0; padding: 16px 20px;
         border-radius: 6px; overflow-x: auto; line-height: 1.5; }
  pre code { background: none; color: inherit; padding: 0; }
  blockquote { border-left: 4px solid #0f3460; margin: 0; padding: 4px 16px;
               background: #f0f4ff; color: #444; border-radius: 0 4px 4px 0; }
  .header-meta { background: #f0f4ff; border: 1px solid #c5d5f5;
                 border-radius: 6px; padding: 12px 16px; margin-bottom: 1.5em;
                 font-size: 0.9em; color: #444; }
  strong { color: #0f3460; }
  hr { border: none; border-top: 1px solid #dde; margin: 2em 0; }
</style>
"""

# Extraer meta-líneas del encabezado para el banner
lines      = md_text.splitlines()
title_line = lines[0].lstrip("# ").strip() if lines else "Entrega"
meta_lines = [l for l in lines[1:5] if l.strip().startswith("**")]
meta_html  = "  ".join(f"<b>{m.strip('*').split(':')[0]}:</b> {':'.join(m.strip('*').split(':')[1:])}"
                       for m in meta_lines if ":" in m)

body_html = markdown.markdown(
    md_text,
    extensions=["tables", "fenced_code", "nl2br", "toc"],
)

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title_line}</title>
  {CSS}
</head>
<body>
  <div class="header-meta">{meta_html}</div>
  {body_html}
</body>
</html>"""

HTML_OUT.write_text(html, encoding="utf-8")
print(f"HTML compilado: {HTML_OUT}")

# Resumen TXT plano (secciones + tabla de resultados)
import re
sections, curr = [], []
for line in lines:
    if line.startswith("## "):
        if curr:
            sections.append("\n".join(curr))
        curr = [line]
    else:
        curr.append(line)
if curr:
    sections.append("\n".join(curr))

# Extraer la tabla de resultados
tabla_block = []
in_tabla = False
for line in lines:
    if "Fusión concat" in line or "F1-macro" in line and "|" in line:
        in_tabla = True
    if in_tabla:
        tabla_block.append(line)
    if in_tabla and line.strip() == "":
        break

sep = "=" * 70
resumen = [
    sep,
    f"  {title_line}",
    sep,
    "",
]
for m in meta_lines:
    resumen.append("  " + m.replace("**", "").replace("|", "·"))
resumen += ["", sep, "  TABLA DE RESULTADOS (test set)", sep]
resumen += tabla_block[:8]
resumen += ["", sep]

TXT_OUT.write_text("\n".join(resumen), encoding="utf-8")
print(f"Resumen TXT:    {TXT_OUT}")
print(f"\nHTML tamaño:    {HTML_OUT.stat().st_size // 1024} KB")
