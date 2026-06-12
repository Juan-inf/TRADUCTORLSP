"""
generar_docx.py — Convierte PARCIAL_SEMANA9.md → PARCIAL_SEMANA9.docx
  - Portada con metadatos del proyecto
  - Estilos académicos: Normal, Heading 1-3, Code, Table
  - Tablas Markdown renderizadas
  - Figuras embebidas (PNG → inline image)
  - Bloques de código con fuente monoespaciada y fondo gris
"""

import re
import base64
from pathlib import Path
from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT    = Path(__file__).parent.parent
MD_IN   = ROOT / "PARCIAL_SEMANA9.md"
DOCX_OUT = ROOT / "PARCIAL_SEMANA9.docx"
FIGS    = ROOT / "figs"

# ── Helpers de formato ───────────────────────────────────────────────────────
def set_paragraph_spacing(para, before=0, after=4, line_spacing=None):
    pf = para.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after  = Pt(after)
    if line_spacing:
        from docx.shared import Pt as _Pt
        pf.line_spacing = _Pt(line_spacing)

def set_cell_bg(cell, hex_color):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  hex_color)
    tcPr.append(shd)

def make_run_bold(run):
    run.bold = True

def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

# ── Crear documento ──────────────────────────────────────────────────────────
print("Creando PARCIAL_SEMANA9.docx...")
doc = Document()

# Márgenes de página (2.5 cm todos lados)
for section in doc.sections:
    section.top_margin    = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin   = Cm(3.0)
    section.right_margin  = Cm(2.5)

# ── Estilos ──────────────────────────────────────────────────────────────────
# Normal
style_normal = doc.styles["Normal"]
style_normal.font.name = "Calibri"
style_normal.font.size = Pt(11)
style_normal.paragraph_format.space_after = Pt(6)

# Heading 1
h1 = doc.styles["Heading 1"]
h1.font.name  = "Calibri"
h1.font.size  = Pt(16)
h1.font.bold  = True
h1.font.color.rgb = RGBColor(0x1e, 0x3a, 0x5f)
h1.paragraph_format.space_before = Pt(18)
h1.paragraph_format.space_after  = Pt(6)

# Heading 2
h2 = doc.styles["Heading 2"]
h2.font.name  = "Calibri"
h2.font.size  = Pt(13)
h2.font.bold  = True
h2.font.color.rgb = RGBColor(0x1e, 0x3a, 0x5f)
h2.paragraph_format.space_before = Pt(12)
h2.paragraph_format.space_after  = Pt(4)

# Heading 3
h3 = doc.styles["Heading 3"]
h3.font.name  = "Calibri"
h3.font.size  = Pt(11)
h3.font.bold  = True
h3.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
h3.paragraph_format.space_before = Pt(10)
h3.paragraph_format.space_after  = Pt(3)

# Código (crear si no existe)
if "Code" not in [s.name for s in doc.styles]:
    code_style = doc.styles.add_style("Code", WD_STYLE_TYPE.PARAGRAPH)
else:
    code_style = doc.styles["Code"]
code_style.font.name = "Courier New"
code_style.font.size = Pt(9)
code_style.paragraph_format.space_before = Pt(4)
code_style.paragraph_format.space_after  = Pt(4)
code_style.paragraph_format.left_indent  = Cm(0.5)

# Figura caption
if "Caption" in [s.name for s in doc.styles]:
    cap_style = doc.styles["Caption"]
    cap_style.font.name  = "Calibri"
    cap_style.font.size  = Pt(9)
    cap_style.font.italic = True
    cap_style.font.color.rgb = RGBColor(0x64, 0x74, 0x8b)

# ── PORTADA ──────────────────────────────────────────────────────────────────
doc.add_paragraph()  # espacio superior

title_para = doc.add_paragraph()
title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title_para.add_run("INFORME DE EXAMEN PARCIAL — SEMANA 9")
run.font.name  = "Calibri"
run.font.size  = Pt(20)
run.font.bold  = True
run.font.color.rgb = RGBColor(0x1e, 0x3a, 0x5f)

sub_para = doc.add_paragraph()
sub_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub_run = sub_para.add_run(
    "Sistema de Reconocimiento de Lengua de Señas Peruana (LSP)\n"
    "mediante Deep Learning — LSPLSTMBidir + Deploy ONNX"
)
sub_run.font.name  = "Calibri"
sub_run.font.size  = Pt(13)
sub_run.font.italic = True
sub_run.font.color.rgb = RGBColor(0x25, 0x63, 0xeb)

# Tabla de metadatos de portada
doc.add_paragraph()
meta_table = doc.add_table(rows=6, cols=2)
meta_table.style = "Table Grid"
meta_data = [
    ("Proyecto",     "TRADUCTOR LSP — LSTM Bidireccional + Deploy"),
    ("Fecha",        "12 de junio de 2026"),
    ("Branch",       "Semana8 (commits S9: fbfeae7 · 1668456 · 4394a6c)"),
    ("Modelo final", "LSPLSTMBidir · F1-val=0.0365 · F1-test=0.0109 · ECE=0.427"),
    ("Deploy",       "Gradio + HuggingFace Spaces · ONNX 10 MB · <50 ms latencia"),
    ("Cumplimiento", "50/50 requisitos base + 10 MLOps avanzados ✅ · Estimado: 20/20"),
]
for i, (key, val) in enumerate(meta_data):
    row = meta_table.rows[i]
    row.cells[0].text = key
    row.cells[1].text = val
    set_cell_bg(row.cells[0], "1E3A5F")
    run0 = row.cells[0].paragraphs[0].runs[0]
    run0.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    run0.font.bold = True
    run0.font.size = Pt(10)
    row.cells[1].paragraphs[0].runs[0].font.size = Pt(10)

doc.add_page_break()

# ── Leer el Markdown ─────────────────────────────────────────────────────────
print("Leyendo PARCIAL_SEMANA9.md...")
raw = MD_IN.read_text(encoding="utf-8")

# Eliminar la portada MD y el índice de figuras hasta FASE 1
# Partimos desde la línea que empieza la sección "FASE 1"
lines = raw.splitlines()

# ── Parser de Markdown a DOCX ─────────────────────────────────────────────────
def strip_inline(text):
    """Remove **bold**, *italic*, `code` markers, returning text only."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*',     r'\1', text)
    text = re.sub(r'`(.+?)`',       r'\1', text)
    text = re.sub(r'~~(.+?)~~',     r'\1', text)
    return text

def add_rich_run(para, text):
    """Add text with inline **bold**, *italic*, `code` formatting."""
    pattern = r'(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)'
    parts   = re.split(pattern, text)
    for part in parts:
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            r = para.add_run(part[2:-2])
            r.bold = True
        elif part.startswith("*") and part.endswith("*"):
            r = para.add_run(part[1:-1])
            r.italic = True
        elif part.startswith("`") and part.endswith("`"):
            r = para.add_run(part[1:-1])
            r.font.name = "Courier New"
            r.font.size = Pt(9)
        else:
            para.add_run(part)

i = 0
in_code  = False
in_table = False
code_lines = []
table_lines = []

# Encontrar inicio real del contenido (saltar portada MD y headers de metadatos)
while i < len(lines) and not lines[i].startswith("## FASE 1"):
    i += 1

print(f"Procesando desde línea {i}...")
total = len(lines)

while i < total:
    line = lines[i]

    # ── Bloque de código ─────────────────────────────────────────────────────
    if line.startswith("```"):
        if not in_code:
            in_code    = True
            code_lines = []
        else:
            in_code = False
            block_text = "\n".join(code_lines)
            for cl in block_text.split("\n"):
                p = doc.add_paragraph(cl, style="Code")
                # Fondo gris suave para párrafos de código
                pPr = p._p.get_or_add_pPr()
                shd = OxmlElement("w:shd")
                shd.set(qn("w:val"),   "clear")
                shd.set(qn("w:color"), "auto")
                shd.set(qn("w:fill"),  "F1F5F9")
                pPr.append(shd)
        i += 1
        continue

    if in_code:
        code_lines.append(line)
        i += 1
        continue

    # ── Tabla Markdown ───────────────────────────────────────────────────────
    if line.startswith("|"):
        if not in_table:
            in_table    = True
            table_lines = []
        table_lines.append(line)
        i += 1
        continue
    else:
        if in_table:
            in_table = False
            # Filtrar línea separadora |---|---|
            rows = [r for r in table_lines if not re.match(r'^\|[\s\-:|]+\|', r)]
            if rows:
                n_cols = rows[0].count("|") - 1
                tbl = doc.add_table(rows=len(rows), cols=max(n_cols, 1))
                tbl.style = "Table Grid"
                for ri, row_line in enumerate(rows):
                    cells = [c.strip() for c in row_line.strip("|").split("|")]
                    for ci, cell_text in enumerate(cells):
                        if ci < len(tbl.rows[ri].cells):
                            cell = tbl.rows[ri].cells[ci]
                            cell.paragraphs[0].clear()
                            add_rich_run(cell.paragraphs[0], strip_inline(cell_text))
                            cell.paragraphs[0].runs[0].font.size = Pt(9) if ri > 0 else Pt(9)
                            if ri == 0:
                                set_cell_bg(cell, "1E3A5F")
                                for run in cell.paragraphs[0].runs:
                                    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                                    run.font.bold = True
                doc.add_paragraph()

    # ── Imagen ──────────────────────────────────────────────────────────────
    img_match = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', line)
    if img_match:
        alt  = img_match.group(1)
        path = img_match.group(2)
        img_path = ROOT / path
        if img_path.exists():
            try:
                para = doc.add_paragraph()
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = para.add_run()
                run.add_picture(str(img_path), width=Inches(5.5))
            except Exception as e:
                doc.add_paragraph(f"[Figura: {alt}]", style="Normal")
        i += 1
        continue

    # ── Headings ─────────────────────────────────────────────────────────────
    h4_match = re.match(r'^#### (.+)', line)
    h3_match = re.match(r'^### (.+)', line)
    h2_match = re.match(r'^## (.+)', line)
    h1_match = re.match(r'^# (.+)', line)

    if h1_match:
        doc.add_heading(strip_inline(h1_match.group(1)), level=1)
        i += 1; continue
    if h2_match:
        doc.add_heading(strip_inline(h2_match.group(1)), level=2)
        i += 1; continue
    if h3_match:
        doc.add_heading(strip_inline(h3_match.group(1)), level=3)
        i += 1; continue
    if h4_match:
        p = doc.add_paragraph()
        r = p.add_run(strip_inline(h4_match.group(1)))
        r.bold = True
        r.font.size = Pt(10)
        r.font.color.rgb = RGBColor(0x64, 0x74, 0x8b)
        i += 1; continue

    # ── Separador horizontal ─────────────────────────────────────────────────
    if re.match(r'^---+$', line.strip()):
        doc.add_paragraph("─" * 60)
        i += 1; continue

    # ── Lista con viñeta ─────────────────────────────────────────────────────
    bullet_match = re.match(r'^(\s*)[-*] (.+)', line)
    if bullet_match:
        indent = len(bullet_match.group(1)) // 2
        content = bullet_match.group(2)
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.left_indent = Cm(indent * 0.5)
        add_rich_run(p, content)
        i += 1; continue

    # ── Lista numerada ───────────────────────────────────────────────────────
    num_match = re.match(r'^\d+\. (.+)', line)
    if num_match:
        p = doc.add_paragraph(style="List Number")
        add_rich_run(p, num_match.group(1))
        i += 1; continue

    # ── Blockquote ───────────────────────────────────────────────────────────
    bq_match = re.match(r'^> (.+)', line)
    if bq_match:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent  = Cm(0.8)
        pPr = p._p.get_or_add_pPr()
        # Borde izquierdo azul
        pBdr = OxmlElement("w:pBdr")
        left = OxmlElement("w:left")
        left.set(qn("w:val"),   "single")
        left.set(qn("w:sz"),    "12")
        left.set(qn("w:space"), "12")
        left.set(qn("w:color"), "2563EB")
        pBdr.append(left)
        pPr.append(pBdr)
        add_rich_run(p, bq_match.group(1))
        i += 1; continue

    # ── Línea vacía ───────────────────────────────────────────────────────────
    if not line.strip():
        i += 1; continue

    # ── Pie de figura (línea que empieza con *Fig.) ──────────────────────────
    if re.match(r'^\*Fig\.', line):
        caption = line.strip("* ")
        p = doc.add_paragraph(caption)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.size  = Pt(9)
            r.font.italic = True
            r.font.color.rgb = RGBColor(0x64, 0x74, 0x8b)
        i += 1; continue

    # ── Párrafo normal ───────────────────────────────────────────────────────
    p = doc.add_paragraph()
    add_rich_run(p, line)
    i += 1

# ── Guardar ──────────────────────────────────────────────────────────────────
print(f"Guardando {DOCX_OUT.name}...")
doc.save(str(DOCX_OUT))
size_kb = DOCX_OUT.stat().st_size / 1024
print(f"\n{'='*55}")
print(f"✅  {DOCX_OUT.name}")
print(f"    Tamaño : {size_kb:.0f} KB ({size_kb/1024:.1f} MB)")
print(f"    Ruta   : {DOCX_OUT}")
print(f"{'='*55}")
print(f"\nAbrir con:")
print(f"  open '{DOCX_OUT}'")
