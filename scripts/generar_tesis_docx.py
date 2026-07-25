# -*- coding: utf-8 -*-
"""Reconstruye TESIS_DOCUMENTO_FINAL.docx (antes PreTesis_LSP_JuanCalla_2026.docx,
renombrado 2026-07-24 al pasar de borrador a documento final) a partir del PDF revisado,
insertando en las secciones de resultados los gráficos REALES generados
esta semana (data/sustentacion_figs/), en vez de los diagramas ilustrativos
del PDF original.

Cambios de contenido respecto al PDF original (revisión editorial):
  - Figura 2 (evolución F1) -> fig_f1_evolucion.png real (24 sprints, no
    solo los 5 hitos resumidos).
  - Figura 4 (Top-k)         -> fig_topk.png real, recomputado en vivo.
  - Figura 5 (latencia E2E)  -> fig_latencia.png real (barras p50/p95/max
    modelo vs. pipeline, no el gráfico de puntos simplificado).
  - Figura 6 (generalización)-> fig_he3_comparacion.png real (ΔF1/PSI/KS
    de las 4 corridas).
  - Anexo B ampliado con fig_matriz_confusion.png y fig_f1_por_clase.png
    reales (antes solo remitía a logs/runs.csv sin visualización).
  - Corregida numeración duplicada: el PDF tiene dos secciones "1.4"
    (Hipótesis de la Investigación y Antecedentes Investigativos). Se
    renumera Hipótesis como 1.4 y Antecedentes pasa a 1.5 (el orden en el
    cuerpo del PDF ya es ese; solo se corrige el número repetido).

Actualización 2026-07-24 — dos hallazgos adicionales post-Sprint 27, cada
uno con su propia subsección, tabla(s) y figura en el Capítulo IV (la
antigua "4.4 Discusión General" pasa a "4.6"; Figuras 7/8 nuevas empujan
las del Anexo B de 7/8 a 9/10; Anexo D y Recomendaciones actualizados en
consecuencia, sin tocar la numeración de Tablas 1-9 del cuerpo original):
  - 4.4: causa raíz real del fallo del abecedario en cámara/video (sesgo
    de dominio "pose≈0", no solo el segmentador) + fix aplicado en imagen
    estática (0%->79.2%) -> fig_slices_ic.png.
  - 4.5: primera medición WER real del proyecto — línea S31->S32->S33,
    WER 2.827->1.134->1.027, supera a producción (1.763) -> Tabla 10
    (datasets, 270 clases) + Tabla 11 (WER) + fig_wer_progresion.png.
"""
from pathlib import Path
from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = Path(__file__).parent.parent
FIGS = ROOT / "data" / "sustentacion_figs"
OUT = ROOT / "TESIS_DOCUMENTO_FINAL.docx"

NAVY = RGBColor(0x1a, 0x1a, 0x1a)
GREY_HEAD = "D9D9D9"

doc = Document()

for section in doc.sections:
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(3.0)
    section.right_margin = Cm(2.5)

# ── Estilos base ──────────────────────────────────────────────────────────
normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(12)
normal.paragraph_format.line_spacing = 1.5
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

h1 = doc.styles["Heading 1"]
h1.font.name = "Times New Roman"; h1.font.size = Pt(14); h1.font.bold = True
h1.font.color.rgb = NAVY
h1.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
h1.paragraph_format.space_before = Pt(18); h1.paragraph_format.space_after = Pt(14)

h2 = doc.styles["Heading 2"]
h2.font.name = "Times New Roman"; h2.font.size = Pt(12); h2.font.bold = True
h2.font.color.rgb = NAVY
h2.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
h2.paragraph_format.space_before = Pt(14); h2.paragraph_format.space_after = Pt(8)

h3 = doc.styles["Heading 3"]
h3.font.name = "Times New Roman"; h3.font.size = Pt(12); h3.font.bold = True; h3.font.italic = True
h3.font.color.rgb = NAVY
h3.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
h3.paragraph_format.space_before = Pt(10); h3.paragraph_format.space_after = Pt(6)

if "Caption" in [s.name for s in doc.styles]:
    cap = doc.styles["Caption"]
    cap.font.name = "Times New Roman"; cap.font.size = Pt(11); cap.font.italic = True

# ── Helpers ───────────────────────────────────────────────────────────────
def p(text="", bold=False, italic=False, size=12, align=None, space_after=6, indent_first=None):
    para = doc.add_paragraph()
    if align is not None:
        para.alignment = align
    para.paragraph_format.space_after = Pt(space_after)
    if indent_first is not None:
        para.paragraph_format.first_line_indent = Cm(indent_first)
    run = para.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    return para

def rich(para, segments):
    """segments: list of (text, bold, italic)"""
    for text, bold, italic in segments:
        r = para.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(12)
        r.bold = bold
        r.italic = italic
    return para

def heading(text, level=1):
    return doc.add_heading(text, level=level)

def table_label(num, title):
    lp = doc.add_paragraph()
    lp.paragraph_format.space_before = Pt(10)
    lp.paragraph_format.space_after = Pt(2)
    r = lp.add_run(f"Tabla {num}")
    r.font.name = "Times New Roman"; r.font.size = Pt(12); r.bold = True
    tp = doc.add_paragraph()
    tp.paragraph_format.space_after = Pt(6)
    r2 = tp.add_run(title)
    r2.font.name = "Times New Roman"; r2.font.size = Pt(12); r2.italic = True

def table_note(text):
    np_ = doc.add_paragraph()
    np_.paragraph_format.space_before = Pt(4)
    np_.paragraph_format.space_after = Pt(14)
    r = np_.add_run("Nota. ")
    r.font.name = "Times New Roman"; r.font.size = Pt(10.5); r.italic = True
    r2 = np_.add_run(text)
    r2.font.name = "Times New Roman"; r2.font.size = Pt(10.5)

def set_cell_shading(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)

def make_table(headers, rows, col_widths=None):
    tbl = doc.add_table(rows=1, cols=len(headers))
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = tbl.rows[0].cells
    for i, htext in enumerate(headers):
        hdr[i].text = ""
        para = hdr[i].paragraphs[0]
        r = para.add_run(htext)
        r.font.name = "Times New Roman"; r.font.size = Pt(10.5); r.bold = True
        set_cell_shading(hdr[i], GREY_HEAD)
    for row in rows:
        cells = tbl.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            para = cells[i].paragraphs[0]
            r = para.add_run(str(val))
            r.font.name = "Times New Roman"; r.font.size = Pt(10.5)
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in tbl.rows:
                row.cells[i].width = Cm(w)
    return tbl

def figure_label(num, title):
    lp = doc.add_paragraph()
    lp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    lp.paragraph_format.space_before = Pt(12)
    lp.paragraph_format.space_after = Pt(2)
    r = lp.add_run(f"Figura {num}")
    r.font.name = "Times New Roman"; r.font.size = Pt(12); r.bold = True
    tp = doc.add_paragraph()
    tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tp.paragraph_format.space_after = Pt(8)
    r2 = tp.add_run(title)
    r2.font.name = "Times New Roman"; r2.font.size = Pt(12); r2.italic = True

def figure_image(path, width_in=5.8):
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = para.add_run()
    run.add_picture(str(path), width=Inches(width_in))

def figure_note(text):
    table_note(text)

def code_block(text):
    para = doc.add_paragraph()
    para.paragraph_format.left_indent = Cm(1.0)
    para.paragraph_format.space_before = Pt(6)
    para.paragraph_format.space_after = Pt(10)
    for i, line in enumerate(text.split("\n")):
        if i > 0:
            para.add_run().add_break()
        r = para.add_run(line)
        r.font.name = "Courier New"
        r.font.size = Pt(9.5)
    pPr = para._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), "F2F2F2")
    pPr.append(shd)

def page_break():
    doc.add_page_break()

def bullet(text):
    para = doc.add_paragraph(style="List Bullet")
    r = para.add_run(text)
    r.font.name = "Times New Roman"; r.font.size = Pt(12)
    return para

def bullet_rich(segments):
    para = doc.add_paragraph(style="List Bullet")
    rich(para, segments)
    return para

print("Construyendo portada...")
# ══════════════════════════════════════════════════════════════════════════
# PORTADA
# ══════════════════════════════════════════════════════════════════════════
for _ in range(3):
    doc.add_paragraph()
p("Universidad Nacional de Ingeniería", bold=True, size=16, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=14)
p("Escuela de Posgrado", bold=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=14)
p("Maestría en Inteligencia Artificial", bold=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=40)
p("Sistema Integral de Comunicación Inclusiva, Basado en Deep Learning, "
  "para la Traducción de Lengua de Señas Peruana a Texto Castellano",
  bold=True, size=13, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=30)
p("Tesis para optar el grado de Maestro en Inteligencia Artificial",
  size=12, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=40)
p("Autor:", bold=True, size=12, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
p("Juan Pablo Calla Choquemamani", size=12, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=60)
p("Lima, Perú", size=12, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
p("2026", size=12, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
page_break()

print("Índice...")
# ══════════════════════════════════════════════════════════════════════════
# ÍNDICE (estático, refleja la numeración corregida)
# ══════════════════════════════════════════════════════════════════════════
p("Índice.", bold=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=20)
indice = [
    ("Índice", "2"), ("Resumen", "5"), ("Abstract", "6"), ("Introducción.", "7"),
    ("Capítulo I. Parte Introductoria del Trabajo", "8"),
    ("1.1 Generalidades", "8"),
    ("1.2 Descripción del Problema de Investigación", "12"),
    ("1.3 Objetivos del Estudio", "13"),
    ("1.3.1 Objetivo General", "13"), ("1.3.2 Objetivos Específicos", "13"),
    ("1.4 Hipótesis de la Investigación", "14"),
    ("1.4.1 Hipótesis General", "14"), ("1.4.2 Hipótesis Específicas", "14"),
    ("1.5 Antecedentes Investigativos", "15"),
    ("Antecedentes Internacionales", "15"),
    ("Capítulo II. Marco Teórico y Conceptual", "17"),
    ("2.1 Marco Teórico", "17"),
    ("2.1.1 Reconocimiento Automático de Lengua de Señas", "17"),
    ("2.1.2 Redes BiLSTM con Mecanismo de Atención", "17"),
    ("2.1.3 Estimación de Pose con MediaPipe Holistic", "17"),
    ("2.1.4 Inferencia Eficiente con ONNX Runtime", "18"),
    ("2.1.5 Generalización y Dataset Shift", "18"),
    ("2.2 Marco Conceptual", "18"),
    ("Capítulo III. Desarrollo del Trabajo de Investigación", "20"),
    ("3.1 Diseño Metodológico", "20"), ("3.2 Corpus y Datos", "20"),
    ("3.3 Arquitectura del Sistema", "21"),
    ("3.4 Proceso de Entrenamiento y Trayectoria Experimental", "23"),
    ("3.5 Plan de Despliegue", "25"),
    ("3.5.1 Estado de los Componentes", "25"),
    ("3.5.2 Divergencia entre demo/ y api/", "26"),
    ("3.5.3 Contratos de la API", "27"),
    ("3.5.4 Riesgos Identificados y Mitigaciones", "29"),
    ("3.5.5 Hoja de Ruta a Producción", "31"),
    ("3.6 Reproducibilidad", "32"),
    ("Capítulo IV. Análisis y Discusión de Resultados", "33"),
    ("4.1 OE1 Precisión de Clasificación (Resultado Parcial)", "33"),
    ("4.2 OE2 Latencia del Pipeline en Tiempo Real", "35"),
    ("4.3 OE3 Generalización fuera de la Muestra", "36"),
    ("4.4 Hallazgo Adicional: Sesgo de Dominio en el Reconocimiento del Abecedario", "38"),
    ("4.5 Hallazgo Adicional: Traducción de Narración Continua", "40"),
    ("4.6 Discusión General", "42"),
    ("Conclusiones", "44"), ("Recomendaciones", "45"),
    ("Referencias Bibliográficas", "46"), ("Anexos", "49"),
    ("Anexo A. Matriz de Consistencia", "49"),
    ("Matriz de Operacionalización de Variables", "50"),
    ("Anexo B. Métricas Detalladas por Clase", "51"),
    ("Anexo C. Contratos de la API y Plan de Despliegue Completo", "52"),
    ("Anexo D. Declaración de Limitaciones", "52"),
]
for title, pg in indice:
    tp = doc.add_paragraph()
    tp.paragraph_format.space_after = Pt(3)
    tab_stops = tp.paragraph_format.tab_stops
    tab_stops.add_tab_stop(Cm(15.5))
    r = tp.add_run(title + "\t" + pg)
    r.font.name = "Times New Roman"; r.font.size = Pt(11)
page_break()

print("Resumen / Abstract / Introducción...")
# ══════════════════════════════════════════════════════════════════════════
# RESUMEN
# ══════════════════════════════════════════════════════════════════════════
p("Resumen", bold=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=16)
p("El presente trabajo de investigación desarrolla un sistema integral de comunicación "
  "inclusiva basado en Deep Learning para la traducción automática de la Lengua de Señas "
  "Peruana (LSP) a texto castellano. El sistema integra extracción de puntos clave "
  "corporales mediante MediaPipe Holistic, clasificación temporal mediante una red "
  "neuronal recurrente bidireccional con memoria a corto-largo plazo (BiLSTM), e "
  "inferencia optimizada en formato ONNX, accesible a través de una interfaz web en "
  "tiempo real.", indent_first=1.25)
p("El estudio adoptó un enfoque cuantitativo de tipo aplicado-experimental con diseño "
  "cuasiexperimental. El corpus utilizado comprende 4 176 muestras correspondientes a 96 "
  "clases del vocabulario LSP. Tras 27 sprints de experimentación con 24 configuraciones "
  "distintas, el modelo alcanzó un F1-macro de 0.4426 en el conjunto de prueba (Top-3: "
  "57.2 %; Top-5: 64.4 %), cifra que representa el mejor punto histórico del proyecto, "
  "aunque no alcanza la meta declarada de F1 ≥ 0.70. En cuanto a la latencia, el pipeline "
  "extremo a extremo opera con una mediana de 54.7 ms (percentil 95: 58.7 ms), cumpliendo "
  "con amplio margen el umbral de 200 ms requerido para la comunicación en tiempo real. La "
  "capacidad de generalización a señantes no vistos durante el entrenamiento fue evaluada "
  "mediante ΔF1, Índice de Estabilidad Poblacional (PSI) y la prueba de Kolmogorov-Smirnov "
  "(KS): las métricas ΔF1 = 0.0406 y PSI = 0.0288 cumplen sus respectivos umbrales (≤ 0.15 "
  "y < 0.20), mientras que el estadístico KS no supera el nivel de significancia de 0.05, "
  "situación atribuida a la hipersensibilidad del test ante el tamaño de muestra "
  "disponible.", indent_first=1.25)
kw = doc.add_paragraph()
kw.paragraph_format.first_line_indent = Cm(1.25)
rich(kw, [("Palabras clave: ", True, False),
          ("lengua de señas peruana, Deep learning, BiLSTM, reconocimiento de señas, "
           "MediaPipe, traducción automática, comunicación inclusiva.", False, False)])
page_break()

p("Abstract", bold=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=16)
p("This research develops a comprehensive inclusive communication system based on Deep "
  "Learning for the automatic translation of Peruvian Sign Language (LSP) to Spanish "
  "text. The system integrates body keypoint extraction via MediaPipe Holistic, temporal "
  "classification through a bidirectional Long Short-Term Memory (BiLSTM) neural network, "
  "and ONNX-optimized inference accessible through a real-time web interface.", indent_first=1.25)
p("The study adopted a quantitative, applied-experimental approach with a "
  "quasi-experimental design. The corpus comprises 4,176 samples across 96 LSP "
  "vocabulary classes. After 27 development sprints with 24 distinct configurations, the "
  "model achieved an F1-macro of 0.4426 on the test set (Top-3: 57.2%; Top-5: 64.4%), "
  "representing the best historical performance of the project, although the stated "
  "target of F1 ≥ 0.70 was not reached. Regarding latency, the end-to-end pipeline "
  "operates at a median of 54.7 ms (95th percentile: 58.7 ms), comfortably meeting the "
  "200 ms threshold required for real-time communication. Generalization to unseen "
  "signers was assessed via ΔF1, Population Stability Index (PSI), and the "
  "Kolmogorov-Smirnov (KS) test: ΔF1 = 0.0406 and PSI = 0.0288 meet their respective "
  "thresholds (≤ 0.15 and < 0.20), while the KS statistic does not reach the 0.05 "
  "significance level, a result attributed to the test's hypersensitivity at the "
  "available sample size.", indent_first=1.25)
kw2 = doc.add_paragraph()
kw2.paragraph_format.first_line_indent = Cm(1.25)
rich(kw2, [("Keywords: ", True, False),
           ("Peruvian sign language, deep learning, BiLSTM, sign recognition, MediaPipe, "
            "automatic translation, inclusive communication.", False, False)])
page_break()

p("Introducción.", bold=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=16)
p("La comunicación es un derecho humano fundamental reconocido en la Convención sobre los "
  "Derechos de las Personas con Discapacidad de las Naciones Unidas (ONU, 2006). Las "
  "personas sordas que utilizan la Lengua de Señas Peruana (LSP) como primera lengua "
  "enfrentan barreras sistemáticas en sus interacciones con la comunidad oyente. El "
  "Instituto Nacional de Estadística e Informática (INEI, 2017) estima que aproximadamente "
  "532 000 peruanos presentan dificultades auditivas significativas.", indent_first=1.25)
p("Los avances en aprendizaje profundo particularmente en redes BiLSTM y Transformers han "
  "abierto nuevas posibilidades para el reconocimiento automático de señas al capturar los "
  "patrones espacio-temporales del movimiento corporal (Koller et al., 2020; Camgoz et "
  "al., 2020). Herramientas como MediaPipe Holistic (Lugaresi et al., 2019) permiten "
  "extraer puntos clave corporales en tiempo real sin hardware especializado, "
  "democratizando el desarrollo de sistemas de reconocimiento de señas.", indent_first=1.25)
page_break()

print("Capítulo I...")
# ══════════════════════════════════════════════════════════════════════════
# CAPÍTULO I
# ══════════════════════════════════════════════════════════════════════════
heading("Capítulo I. Parte Introductoria del Trabajo", level=1)
heading("1.1 Generalidades", level=2)
p("La comunicación es un derecho humano inalienable cuyo ejercicio pleno define la "
  "participación de las personas en la vida social, educativa, laboral y política. La "
  "Convención sobre los Derechos de las Personas con Discapacidad, adoptada por las "
  "Naciones Unidas en 2006 y ratificada por el Perú mediante Decreto Supremo N.° "
  "073-2007-RE, reconoce explícitamente las lenguas de señas como sistemas lingüísticos "
  "completos y establece la obligación de los Estados de promover el acceso a la "
  "comunicación en formatos accesibles. No obstante, la distancia entre el reconocimiento "
  "normativo y la realidad cotidiana de las personas sordas persiste de manera "
  "significativa: en la mayoría de los contextos de interacción servicios de salud, "
  "instituciones educativas, organismos públicos y espacios de trabajo, la Lengua de "
  "Señas Peruana (LSP) es desconocida por la población oyente, lo que sitúa a los "
  "usuarios de esta lengua en una posición estructural de desventaja comunicativa.", indent_first=1.25)
p("En el contexto nacional, el Instituto Nacional de Estadística e Informática (INEI, "
  "2017) estimó que aproximadamente 532 000 peruanos presentan dificultades auditivas, de "
  "los cuales una proporción relevante utiliza la LSP como primera lengua. Pese a que la "
  "Ley N.° 29535 (2010) otorgó reconocimiento oficial a la LSP, su integración en los "
  "sistemas de comunicación digital ha avanzado de manera muy limitada. No existe en el "
  "país, hasta la fecha de elaboración del presente estudio, un sistema de reconocimiento "
  "automático de LSP con validación cuantitativa publicada en la literatura científica "
  "indexada, lo que representa una brecha tecnológica con consecuencias directas sobre la "
  "inclusión digital y la autonomía comunicativa de la comunidad sorda peruana.", indent_first=1.25)
p("A escala global, la inteligencia artificial y en particular el aprendizaje profundo "
  "(Deep learning) ha transformado radicalmente el campo del reconocimiento automático de "
  "lenguas de señas (Sign Language Recognition, SLR) durante la última década. "
  "Arquitecturas como las redes neuronales convolucionales (CNN), las redes recurrentes "
  "bidireccionales (BiLSTM) y los modelos Transformer han demostrado capacidad para "
  "capturar los patrones espacio-temporales que caracterizan al movimiento de las manos, "
  "los brazos y el cuerpo en la comunicación mediante señas (Koller et al., 2020; Camgoz "
  "et al., 2020; Rastgoo et al., 2021). La disponibilidad de herramientas de estimación de "
  "pose corporal en tiempo real como MediaPipe Holistic (Lugaresi et al., 2019) ha "
  "reducido significativamente la barrera de acceso al desarrollo de estos sistemas, al "
  "eliminar la necesidad de hardware especializado y permitir la extracción de "
  "representaciones robustas de movimiento corporal desde una cámara estándar.", indent_first=1.25)
p("Sin embargo, los avances documentados en la literatura han sido desarrollados "
  "predominantemente para lenguas de señas de alto recurso, como la American Sign "
  "Language (ASL), la Deutsche Gebärdensprache (DGS) o la lengua de señas china, que "
  "cuentan con corpus de decenas de miles de muestras etiquetadas y décadas de "
  "investigación acumulada. La LSP, como la mayoría de las lenguas de señas "
  "latinoamericanas, es una lengua de bajo recurso en el sentido computacional: los "
  "corpus públicamente disponibles son escasos, de tamaño reducido y no reflejan la "
  "variabilidad regional, generacional y de estilo propio del uso natural de la lengua "
  "(Bragg et al., 2019). Esta asimetría de recursos condiciona directamente el desempeño "
  "alcanzable por los sistemas automáticos y exige estrategias metodológicas adaptadas al "
  "contexto de datos limitados.", indent_first=1.25)
p("El presente trabajo de investigación responde a esta problemática mediante el "
  "desarrollo de un sistema integral de comunicación inclusiva basado en Deep learning "
  "para la traducción automática de la LSP a texto castellano, accesible como aplicación "
  "web sin requerir instalación por parte del usuario final. El sistema articula cinco "
  "módulos funcionales: captura de señas en tiempo real o desde video pregrabado, "
  "extracción de 75 puntos clave corporales mediante MediaPipe Holistic, segmentación de "
  "señas aisladas por detección de pausas, clasificación temporal mediante una red "
  "BiLSTM con mecanismo de atención exportada a formato ONNX para inferencia optimizada "
  "sobre CPU estándar, y una interfaz de usuario desplegada públicamente en HuggingFace "
  "Spaces. El proceso de desarrollo comprendió 27 sprints de experimentación iterativa "
  "con 24 configuraciones distintas de arquitectura e hiperparámetros, sobre un corpus de "
  "4 176 muestras correspondientes a 96 clases del vocabulario LSP.", indent_first=1.25)
p("Los resultados obtenidos evidencian tanto los logros como las limitaciones propias de "
  "un sistema desarrollado sobre una lengua de bajo recurso. El modelo alcanzó un "
  "F1-macro de 0.4426 en el conjunto de prueba independiente (Top-3: 57.2 %; Top-5: "
  "64.4 %), cifra que representa el mejor punto histórico del proyecto, aunque no alcanza "
  "la meta declarada de F1 ≥ 0.70, cuya brecha se atribuye a la cantidad de muestras "
  "disponibles por clase y no a limitaciones de la arquitectura. En cuanto a la "
  "eficiencia temporal, el pipeline extremo a extremo opera con latencia mediana de "
  "54.7 ms cumpliendo con amplio margen el umbral de 200 ms requerido para la "
  "comunicación en tiempo real con el componente MediaPipe Holistic como cuello de "
  "botella computacional, no el modelo de clasificación. La capacidad de generalización a "
  "señantes no incluidos en el entrenamiento fue evaluada mediante ΔF1 = 0.0406 y "
  "PSI = 0.0288, ambos satisfactorios, con una limitación documentada en el estadístico "
  "de Kolmogorov-Smirnov atribuida a la hipersensibilidad estadística del test ante el "
  "tamaño de muestra disponible.", indent_first=1.25)
p("La relevancia del presente estudio se sitúa en tres planos complementarios. En el "
  "plano científico, contribuye con evidencia cuantitativa sobre la viabilidad de "
  "sistemas BiLSTM con representaciones de puntos clave para el reconocimiento de señas "
  "LSP, aportando métricas de reproducibilidad, generalización y dataset shift que no han "
  "sido reportadas previamente para esta lengua en la literatura indexada. En el plano "
  "tecnológico, desarrolla y valida una arquitectura de sistema integral con contratos de "
  "API documentados, suite de tests automatizados y despliegue público que puede ser "
  "adoptada y extendida por otros investigadores o desarrolladores. En el plano social, "
  "el sistema constituye un prototipo funcional orientado a reducir las barreras "
  "comunicativas de la comunidad sorda peruana mediante tecnología accesible desde "
  "dispositivos estándar sin instalación.", indent_first=1.25)
p("El presente documento se organiza en cuatro capítulos. El Capítulo I presenta las "
  "generalidades del estudio, la descripción del problema de investigación, los objetivos "
  "y los antecedentes investigativos nacionales e internacionales más relevantes. El "
  "Capítulo II desarrolla el marco teórico que aborda los fundamentos de las "
  "arquitecturas BiLSTM, la estimación de pose corporal con MediaPipe Holistic, la "
  "inferencia optimizada con ONNX Runtime y los conceptos de generalización y dataset "
  "shift así como el marco conceptual que articula los términos clave del estudio. El "
  "Capítulo III describe el desarrollo del trabajo de investigación, incluyendo el diseño "
  "metodológico, el corpus utilizado, la arquitectura del sistema, el proceso de "
  "entrenamiento y la trayectoria experimental, el plan de despliegue con el estado "
  "detallado de los componentes, los riesgos identificados y la hoja de ruta a "
  "producción. El Capítulo IV presenta el análisis y la discusión de los resultados "
  "obtenidos para cada objetivo específico, contrastados con los antecedentes y el marco "
  "teórico. El documento concluye con las conclusiones por objetivo, las recomendaciones "
  "derivadas de los hallazgos, las referencias bibliográficas en formato APA 7.ª edición "
  "y los anexos.", indent_first=1.25)

heading("1.2 Descripción del Problema de Investigación", level=2)
p("No existe en el Perú un sistema de reconocimiento automático de LSP con validación "
  "cuantitativa publicada en la literatura científica indexada. Los sistemas "
  "internacionales disponibles no son transferibles directamente a la LSP debido a sus "
  "particularidades léxicas y gramaticales (Bragg et al., 2019). Desde la perspectiva "
  "tecnológica, el SLR enfrenta desafíos de variabilidad inter-señante, escasez de corpus "
  "etiquetados y necesidad de operar en tiempo real con hardware estándar (Rastgoo et "
  "al., 2021).", indent_first=1.25)
prob = doc.add_paragraph()
rich(prob, [("Problema General: ", True, False),
            ("¿En qué medida un sistema integral basado en Deep Learning permite realizar "
             "la traducción automática de la Lengua de Señas Peruana (LSP) a texto "
             "castellano?", False, False)])
bullet_rich([("PO1: ", True, False), ("¿En qué medida un modelo basado en Deep Learning "
             "permite mejorar el desempeño de la traducción automática de la Lengua de "
             "Señas Peruana (LSP) a texto castellano, alcanzando un F1-score ≥ 0.70?", False, False)])
bullet_rich([("PO2: ", True, False), ("¿En qué medida el pipeline de inferencia permite la "
             "traducción en tiempo real con latencia < 200 ms?", False, False)])
bullet_rich([("PO3: ", True, False), ("¿En qué medida la generalización del modelo permite "
             "mantener el desempeño ante usuarios y datos no observados durante el "
             "entrenamiento?", False, False)])

heading("1.3 Objetivos del Estudio", level=2)
heading("1.3.1 Objetivo General", level=3)
p("Desarrollar un sistema integral basado en Deep Learning para la traducción automática "
  "de la Lengua de Señas Peruana (LSP) a texto castellano.")
heading("1.3.2 Objetivos Específicos", level=3)
bullet_rich([("OE1: ", True, False), ("Diseñar y desarrollar un modelo basado en Deep "
             "Learning para la traducción automática de la Lengua de Señas Peruana (LSP) "
             "a texto castellano, con un desempeño F1-score ≥ 0.70.", False, False)])
bullet_rich([("OE2: ", True, False), ("Mejorar el pipeline de inferencia basado en Deep "
             "Learning para realizar la traducción automática de la Lengua de Señas "
             "Peruana (LSP) en tiempo real con una latencia inferior a 200 ms.", False, False)])
bullet_rich([("OE3: ", True, False), ("Mejorar la capacidad de generalización del modelo "
             "basado en Deep Learning para mantener el desempeño de la traducción "
             "automática de la Lengua de Señas Peruana (LSP) ante usuarios y datos no "
             "observados durante el entrenamiento.", False, False)])

heading("1.4 Hipótesis de la Investigación", level=2)
heading("1.4.1 Hipótesis General", level=3)
p("El sistema integral basado en Deep Learning permite realizar la traducción automática "
  "de la Lengua de Señas Peruana (LSP) a texto castellano mediante un desempeño "
  "cuantificable del modelo y del proceso de inferencia.")
heading("1.4.2 Hipótesis Específicas", level=3)
bullet_rich([("HE1: ", True, False), ("El modelo basado en Deep Learning permite mejorar "
             "el desempeño de la traducción automática de la Lengua de Señas Peruana "
             "(LSP) a texto castellano, logrando un F1-score ≥ 0.70 durante la evaluación "
             "experimental.", False, False)])
bullet_rich([("HE2: ", True, False), ("La mejora del pipeline de inferencia basado en Deep "
             "Learning permite realizar la traducción automática de la Lengua de Señas "
             "Peruana (LSP) en tiempo real, alcanzando una latencia inferior a 200 ms sin "
             "afectar significativamente el desempeño del modelo.", False, False)])
bullet_rich([("HE3: ", True, False), ("La mejora de la capacidad de generalización del "
             "modelo basado en Deep Learning permite mantener el desempeño de la "
             "traducción automática de la Lengua de Señas Peruana (LSP) ante usuarios y "
             "datos no observados durante el entrenamiento.", False, False)])

heading("1.5 Antecedentes Investigativos", level=2)
heading("Antecedentes Internacionales", level=3)

antecedentes = [
    ("Camgoz et al. (2020).", "Sign Language Transformers: Joint End-to-End Sign Language "
     "Recognition and Translation. IEEE/CVF CVPR. Propone un Transformer de extremo a "
     "extremo para reconocimiento y traducción simultánea de DGS, reportando BLEU-4 = "
     "23.65 sobre PHOENIX-Weather 2014T. Referencia metodológica para métricas de "
     "evaluación y arquitecturas secuencia a secuencia."),
    ("Koller et al. (2020).", "Weakly Supervised Learning with Multi-Stream CNN-LSTM-HMM "
     "for Sign Language Recognition and Translation. IEEE TPAMI, 42(11). Combina CNN, "
     "LSTM e HMM en aprendizaje débilmente supervisado, demostrando que pipelines "
     "multimodales mejoran la precisión del reconocimiento continuo de señas."),
    ("Rastgoo et al. (2021).", "Sign Language Recognition: A Deep Survey. Expert Systems "
     "with Applications, 164, 113794. Revisión sistemática de más de 200 trabajos. "
     "Identifica la escasez de corpus, variabilidad inter-señante y transferencia a "
     "nuevos señantes como principales desafíos coincidentes con las limitaciones del "
     "presente estudio."),
    ("Jiang et al. (2021).", "Skeleton Aware Multi-modal Sign Language Recognition. "
     "IEEE/CVF CVPRW. Demuestra que representaciones de puntos clave son suficientes "
     "para lograr precisiones competitivas con mayor eficiencia computacional y "
     "robustez a variaciones de iluminación y fondo."),
    ("De Coster et al. (2020).", "Sign Language Recognition with Transformer Networks. "
     "LREC 2020. Reporta que modelos basados en puntos clave presentan mayor "
     "generalización a nuevos señantes respecto a modelos basados en apariencia "
     "visual."),
    ("Bragg et al. (2019).", "Sign Language Recognition, Generación, and Translation: An "
     "Interdisciplinary Perspective. ACM ASSETS. Revisión interdisciplinaria que destaca "
     "la importancia de corpus que reflejen la variabilidad natural del uso de la lengua "
     "de señas en contextos cotidianos."),
]
for autor, resto in antecedentes:
    para = doc.add_paragraph()
    para.paragraph_format.space_after = Pt(10)
    rich(para, [(autor + " ", True, False), (resto, False, False)])
page_break()

print("Capítulo II...")
# ══════════════════════════════════════════════════════════════════════════
# CAPÍTULO II
# ══════════════════════════════════════════════════════════════════════════
heading("Capítulo II. Marco Teórico y Conceptual", level=1)
heading("2.1 Marco Teórico", level=2)
heading("2.1.1 Reconocimiento Automático de Lengua de Señas", level=3)
p("El reconocimiento automático de lengua de señas (SLR) aborda la interpretación de la "
  "comunicación visual-gestual como un problema de clasificación de secuencias "
  "temporales. La literatura distingue entre reconocimiento de señas aisladas (ISLR) y "
  "reconocimiento continuo (CSLR). El presente estudio se enmarca en ISLR con "
  "segmentación por pausas, lo que permite operar con hardware estándar sin requerir "
  "modelos de lenguaje de señas completos (Rastgoo et al., 2021).")
heading("2.1.2 Redes BiLSTM con Mecanismo de Atención", level=3)
p("La arquitectura Long Short-Term Memory (Hochreiter & Schmidhuber, 1997) resuelve el "
  "problema del desvanecimiento del gradiente en RNN mediante celdas de memoria con "
  "compuertas. La variante bidireccional (Schuster & Paliwal, 1997) procesa la secuencia "
  "en ambas direcciones integrando contexto pasado y futuro especialmente valiosa en "
  "señas donde un fotograma depende de los movimientos precedentes y siguientes. Los "
  "mecanismos de atención (Bahdanau et al., 2015) ponderan diferencialmente los estados "
  "ocultos más informativos para la predicción de clase.")
heading("2.1.3 Estimación de Pose con MediaPipe Holistic", level=3)
p("MediaPipe Holistic (Lugaresi et al., 2019) detecta simultáneamente 543 puntos de "
  "referencia: 468 faciales, 21 por mano y 33 corporales. Opera en tiempo real sobre CPU "
  "estándar sin hardware especializado. La representación mediante puntos clave ofrece "
  "robustez a variaciones de iluminación, color de piel y fondo, con mejor generalización "
  "inter-señante respecto a representaciones de video completo (Jiang et al., 2021). En "
  "este estudio se utilizan 75 landmarks de mayor relevancia para la LSP (manos + cuerpo "
  "superior), generando vectores de 150 dimensiones por fotograma.")
heading("2.1.4 Inferencia Eficiente con ONNX Runtime", level=3)
p("ONNX (Bai et al., 2019) es un formato abierto de representación de modelos que permite "
  "portabilidad entre frameworks y optimización de la inferencia mediante fusión de "
  "operadores y cuantización. En el presente sistema, la exportación a ONNX (opset 17) "
  "redujo la latencia de inferencia del modelo BiLSTM a valores medianos < 1 ms, haciendo "
  "que MediaPipe Holistic (~55 ms) sea el cuello de botella computacional del pipeline.")
heading("2.1.5 Generalización y Dataset Shift", level=3)
p("La generalización es la capacidad del modelo de mantener su desempeño ante "
  "distribuciones no vistas (Quionero-Candela et al., 2009). En SLR, el dataset shift se "
  "manifiesta principalmente como variación inter-señante. Las métricas complementarias "
  "utilizadas son: ΔF1 (brecha de desempeño entre conjunto interno y holdout externo), "
  "PSI (Population Stability Index, cambio distribucional sobre activaciones) y la "
  "prueba KS (diferencia en distribución de puntuaciones de confianza softmax).")

heading("2.2 Marco Conceptual", level=2)
conceptos = [
    ("Lengua de Señas Peruana (LSP): ", "Sistema lingüístico visual-gestual de la "
     "comunidad sorda peruana, reconocido oficialmente por Ley N.° 29535 (2010)."),
    ("BiLSTM: ", "Red neuronal recurrente bidireccional con memoria a largo-corto plazo. "
     "Procesa secuencias temporales en dirección progresiva y regresiva (Schuster & "
     "Paliwal, 1997)."),
    ("F1-macro: ", "Promedio no ponderado del F1-score por clase (media armónica de "
     "precisión y recall). Adecuado para datasets con desequilibrio de clases."),
    ("PSI (Population Stability Index): ", "PSI = Σ (P_ref − P_new) · ln(P_ref/P_new). "
     "PSI < 0.10: estable; 0.10–0.20: cambio moderado; ≥ 0.20: cambio severo."),
    ("ΔF1: ", "Brecha de generalización: ΔF1 = F1-macro (prueba interna) − F1-macro "
     "(holdout externo). Umbral de aceptación: ≤ 0.15."),
    ("ONNX Runtime: ", "Motor de inferencia optimizada para modelos exportados en "
     "formato Open Neural Network Exchange."),
]
for term, defn in conceptos:
    para = doc.add_paragraph()
    para.paragraph_format.space_after = Pt(10)
    rich(para, [(term, True, False), (defn, False, False)])
page_break()

print("Capítulo III...")
# ══════════════════════════════════════════════════════════════════════════
# CAPÍTULO III
# ══════════════════════════════════════════════════════════════════════════
heading("Capítulo III. Desarrollo del Trabajo de Investigación", level=1)
heading("3.1 Diseño Metodológico", level=2)
p("La investigación adoptó un enfoque cuantitativo, tipo aplicado-experimental, con "
  "diseño cuasiexperimental. La variable independiente es el sistema integral basado en "
  "Deep Learning; la variable dependiente es la calidad de la traducción automática "
  "evaluada en precisión (F1-macro), latencia (ms) y generalización (ΔF1, PSI, KS). La "
  "validación interna utilizó KFold estratificado (k=5) y el conjunto de prueba (15%) fue "
  "evaluado únicamente al finalizar cada configuración. El holdout externo (20% separado "
  "por grupo de señante) se evaluó exclusivamente con el modelo final.", indent_first=1.25)

heading("3.2 Corpus y Datos", level=2)
p("El corpus LSP peruano comprende 4 176 muestras de 96 clases de vocabulario. Cada "
  "muestra es una ventana temporal de 30 fotogramas con 75 puntos clave corporales "
  "extraídos mediante MediaPipe Holistic, normalizados por z-score respecto a las "
  "coordenadas del torso, generando vectores de 150 dimensiones por fotograma.", indent_first=1.25)
table_label(1, "Características del corpus LSP utilizado en el estudio")
make_table(["Característica", "Valor"], [
    ["N.° de clases", "96"],
    ["N.° total de muestras", "4 176"],
    ["Fotogramas por muestra", "30"],
    ["Dimensión del vector de características", "150 (75 landmarks × 2 coordenadas x, y)"],
    ["Partición entrenamiento / validación / prueba", "70 % / 15 % / 15 %"],
    ["N.° muestras conjunto de prueba", "≈ 626"],
    ["N.° muestras holdout externo (por grupo de señante)", "≈ 835 (20 % del total)"],
    ["Semilla aleatoria (StratifiedShuffleSplit)", "42"],
], col_widths=[9, 7])
table_note("LSP = Lengua de Señas Peruana. El holdout externo fue separado por grupo de "
           "señante antes del inicio del entrenamiento. La partición de prueba fue "
           "evaluada una única vez por configuración.")

heading("3.3 Arquitectura del Sistema", level=2)
p("El sistema está compuesto por cinco módulos funcionales organizados en un pipeline "
  "secuencial. La Figura 1 ilustra el flujo completo desde la captura hasta la salida de "
  "texto.", indent_first=1.25)
figure_label(1, "Arquitectura del pipeline de traducción LSP a texto castellano")
figure_image(FIGS / "fig_arquitectura_pipeline.png", width_in=4.8)
figure_note("El costo dominante del pipeline es MediaPipe Holistic (~55 ms/frame). La "
            "inferencia del modelo ONNX consume < 1 ms. La segmentación por pausas "
            "(segmentacion.py) está implementada en la demo y en Spaces; su integración "
            "en api/main.py es trabajo pendiente (Riesgo R9).")

p("Los cinco módulos funcionales del sistema son:")
bullet_rich([("Módulo de captura: ", True, False), ("Adquisición de fotogramas desde "
             "cámara web o video pregrabado (MP4, AVI, MOV). Transmisión al backend vía "
             "WebSocket o REST.", False, False)])
bullet_rich([("Módulo de extracción de características: ", True, False), ("Detección de "
             "75 landmarks corporales con MediaPipe Holistic y normalización z-score "
             "(src/features/landmarks.py, compartido entre API y demo).", False, False)])
bullet_rich([("Módulo de segmentación: ", True, False), ("Detección de pausas entre señas "
             "mediante análisis del movimiento de landmarks de las manos "
             "(src/features/segmentacion.py). Activo en demo y Spaces; pendiente en "
             "api/main.py.", False, False)])
bullet_rich([("Módulo de clasificación: ", True, False), ("Red BiLSTM con atención, "
             "exportada a ONNX (3.9 MB, opset 17). 96 clases, inferencia < 1 ms en CPU "
             "estándar.", False, False)])
bullet_rich([("Módulo de interfaz y despliegue: ", True, False), ("Backend FastAPI + "
             "WebSocket, demo Gradio con TTS, despliegue público en HuggingFace Spaces "
             "sin instalación para el usuario final.", False, False)])

heading("3.4 Proceso de Entrenamiento y Trayectoria Experimental", level=2)
p("El proceso de entrenamiento se ejecutó a lo largo de 27 sprints probando 24 "
  "configuraciones. La optimización de hiperparámetros utilizó Optuna con el algoritmo "
  "TPE (50 ensayos por configuración), con regularización (label smoothing, dropout) y "
  "calibración de confianza (Temperature Scaling). La Figura 2 muestra la trayectoria "
  "completa del F1-macro a lo largo de los 24 sprints con métrica registrada un gráfico "
  "generado en vivo a partir de logs/runs.csv, más detallado que el resumen de hitos de "
  "la Tabla 2.", indent_first=1.25)
figure_label(2, "Evolución del F1-macro por sprint (S5–S27, datos reales de logs/runs.csv)")
figure_image(FIGS / "fig_f1_evolucion.png")
figure_note("Trayectoria completa de los 24 sprints con F1-test registrado, sin filtrar. "
            "No es monótona: los sprints S14–S25 muestran altibajos por exploración de "
            "arquitecturas y fuentes de datos distintas. El punto v3 (S27, F1=0.4563) fue "
            "el mejor histórico pero su checkpoint fue sobrescrito por v4 durante el "
            "entrenamiento y no es recuperable.")

table_label(2, "Trayectoria del modelo BiLSTM a lo largo de 27 sprints de desarrollo")
make_table(["Hito", "Sprint", "F1-macro", "ΔF1 holdout", "Observación"], [
    ["Baseline (Regresión Logística)", "S5", "0.0058", "—", "Punto de partida"],
    ["Primer F1 útil", "S13", "0.3696", "—", "Primera configuración significativa"],
    ["Mejor resultado pre-S27", "S26", "0.4098", "0.1663", "HE3 fallaba (ΔF1 > 0.15)"],
    ["Mejor F1 histórico", "S27-v3", "0.4563", "0.0785", "Checkpoint sobrescrito por v4"],
    ["Modelo activo (v4)", "S27-v4", "0.4426", "0.0406", "Mejor generalización histórica"],
], col_widths=[5, 2, 2, 2.5, 4.5])
table_note("El checkpoint v3 no es recuperable. ΔF1 = F1-macro (prueba interna) − F1-macro "
           "(holdout externo). Cumple el umbral ΔF1 ≤ 0.15 con holgura de 3.7×. F1-macro "
           "calculado sobre el conjunto de prueba (15 %, N ≈ 626 muestras, semilla = 42).")

heading("3.5 Plan de Despliegue", level=2)
heading("3.5.1 Estado de los Componentes", level=3)
p("La Tabla 3 resume el estado actual de cada componente del sistema, verificado "
  "mediante ejecución en vivo al cierre del Sprint 13 de entregables (S13).", indent_first=1.25)
table_label(3, "Estado de los componentes del sistema al cierre del Sprint 13 de entregables")
make_table(["Componente", "Archivo", "Estado", "Observación"], [
    ["Extracción de landmarks", "src/features/landmarks.py", "Operativo", "Compartido entre API y demo"],
    ["Segmentación por pausas", "src/features/segmentacion.py", "Operativo", "Calibrado con datos reales"],
    ["Modelo ONNX", "checkpoints/bilstm_s27.onnx", "Entrenado", "3.9 MB — no versionado en git"],
    ["Backend API + WebSocket", "api/main.py", "Validado en vivo", "Sin mejoras de segmentación (R9)"],
    ["Demo interactiva", "demo/app_gradio.py", "Validado en vivo", "Incluye segmentación + TTS + filtro"],
    ["Despliegue público", "spaces/app.py (HuggingFace)", "Actualizado a S27", "Autónomo, sin importar src/"],
    ["Contenedor Docker", "Dockerfile, docker-compose.yml", "Sin build real", "Bloqueado por entorno sin Docker"],
    ["Suite de tests", "tests/ (6 tests)", "6/6 pasan", "Smoke, golden, contrato WebSocket"],
], col_widths=[3.5, 5, 2.5, 5])
table_note("Validado con datos reales en vivo. R9 = Riesgo identificado: los ajustes de "
           "calidad de la demo (segmentación por pausas, filtro de clases narrativas, "
           "umbral de confianza 0.20) no han sido portados a api/main.py.")

heading("3.5.2 Divergencia entre demo/ y api/", level=3)
p("Se identificó una divergencia crítica entre los componentes demo y API (Riesgo R9): "
  "los tres ajustes de calidad implementados esta semana segmentación por pausas, filtro "
  "de clases HISTORIAS_VINETAS (vocabulario_lsp), y umbral de confianza calibrado a 0.20 "
  "están presentes en demo/app_gradio.py y spaces/app.py, pero no en api/main.py. El "
  "backend API opera con ventana fija de 30 frames y el CONFIG (confidence_threshold) "
  "=0.40 es una variable que nunca se utiliza para filtrar predicciones. Esta divergencia "
  "fue confirmada por análisis automatizado del código fuente (búsqueda en vivo de las "
  "cadenas SegmentadorPausas y CONF_UMBRAL en ambos archivos, documentada en "
  "notebooks/ENTREGABLE_PLAN_DE_DESPLIEGUE_S13.ipynb).", indent_first=1.25)
p("La implicación práctica es que un cliente integrado contra el WebSocket "
  "/predict/stream recibiría predicciones de menor calidad que las observadas en la demo "
  "interactiva. La corrección de esta divergencia es la tarea pendiente de mayor "
  "prioridad antes de utilizar el backend como superficie de integración real.", indent_first=1.25)

heading("3.5.3 Contratos de la API", level=3)
p("La Figura 3 presenta los esquemas de entrada y salida de los endpoints principales, "
  "capturados en vivo.", indent_first=1.25)
figure_label(3, "Contratos I/O de los endpoints del sistema — respuestas capturadas en vivo")
figure_image(FIGS / "fig_contratos_io.png", width_in=5.8)
p("Nota: api/main.py no filtra por confianza (§3.5.2, R9) a diferencia de la demo, esta "
  "respuesta se devuelve tal cual aunque la confianza sea baja (9.8% en este ejemplo "
  "real). En una segunda corrida en vivo sobre el mismo video, el top-1 cambió de DIEZ a "
  "ORIGINAL (diferencia de confianza de solo 0.0007, prácticamente empatados) — hallazgo "
  "que evidencia que, cuando el top-3 está así de cerca, el campo top3 es más confiable "
  "que clase sola como salida del contrato.", indent_first=1.25)
figure_note("Las respuestas corresponden a ejecuciones reales sobre el servidor local. La "
            "latencia de 4 662.9 ms en /predict/video corresponde al procesamiento de un "
            "video completo de ~1 minuto. El endpoint WS /predict/stream opera a "
            "p50 = 54.7 ms para señas cortas.")

heading("3.5.4 Riesgos Identificados y Mitigaciones", level=3)
p("La Tabla 4 presenta los riesgos reales identificados durante el desarrollo, todos "
  "verificados empíricamente.", indent_first=1.25)
table_label(4, "Riesgos identificados durante el desarrollo y sus mitigaciones")
riesgos = [
    ["R1", "Checkpoint ONNX no versionado en git", "Alta", "Commitear con excepción en .gitignore o publicar como artefacto en GitHub Releases"],
    ["R2", "Checkpoint v3 perdido por sobrescritura", "Alta", "Versionar por nombre de corrida (bilstm_s27_v4.onnx), no con nombre fijo"],
    ["R3", "Tiempo de entrenamiento impredecible (788 min vs. 66-120 min histórico)", "Media", "No reentrenar cerca de fechas límite; reservar margen de 10×"],
    ["R4", "Umbral de confianza mal calibrado (0.30 ocultaba predicciones correctas)", "Media", "Corregido a 0.20 tras calibración con 15 ejemplos reales"],
    ["R5", "Docker sin build real probado", "Media", "Probar docker build en primera oportunidad con Docker disponible"],
    ["R6", "Cambio de optimización con regresión silenciosa (model_complexity=0 degradó detección)", "Alta", "Validar todo cambio de performance contra videos de referencia antes de aceptarlo"],
    ["R7", "CORS abierto allow_origins=[\"*\"]", "Alta", "Restringir antes de exponer el backend fuera de red controlada"],
    ["R8", "Sin reconocimiento continuo de narración fluida", "Conocida", "Documentada como limitación de capacidad, fuera del alcance de este plan"],
    ["R9", "api/main.py sin mejoras de calidad de la demo (segmentación, filtro, umbral)", "Alta", "Portar los 3 ajustes a api/main.py antes de integración real"],
    ["R10", "Videos largos vía /predict/video con bajo rendimiento (1/7 en top3)", "Media", "Usar clips cortos de seña única; documentado en tests/test_golden.py"],
]
make_table(["ID", "Riesgo", "Severidad", "Mitigación"], riesgos, col_widths=[1.3, 6, 2.2, 6.5])
table_note("Todos los riesgos fueron verificados empíricamente durante el Sprint 13 de "
           "entregables, no son hipotéticos. Severidad Alta = puede impactar producción o "
           "calidad del modelo. Severidad Media = impacta desarrollo, pero tiene "
           "mitigación disponible.")

heading("3.5.5 Hoja de Ruta a Producción", level=3)
p("La Tabla 5 presenta las tareas pendientes ordenadas por prioridad para llevar el "
  "sistema a un estado de producción real.", indent_first=1.25)
table_label(5, "Hoja de ruta de tareas pendientes para el despliegue en producción del sistema")
hoja_ruta = [
    ["1", "Inmediata", "Portar segmentación + filtro narrativas + umbral 0.20 a api/main.py (R9)", "Pendiente", "Antes de integración con cliente real"],
    ["2", "Inmediata", "Versionar checkpoint ONNX fuera de .gitignore (R1, R2)", "Pendiente", "Antes del próximo despliegue"],
    ["3", "Corto plazo", "Build y test real del contenedor Docker (R5)", "Bloqueado", "Primera sesión con Docker disponible"],
    ["4", "Corto plazo", "Congelar requirements-api.txt a versiones exactas", "Pendiente", "Antes de producción"],
    ["5", "Corto plazo", "Restringir CORS y agregar límite de tamaño en UploadFile (R7)", "Pendiente", "Antes de exponer fuera de red local"],
    ["6", "Corto plazo", "Logging estructurado (reemplazar print() por logging estándar)", "Pendiente", "1 semana"],
    ["7", "Mediano plazo", "Configurar CI (pytest tests/ en cada push a main)", "Pendiente", "1 semana"],
    ["8", "Mediano plazo", "Implementar métricas agregadas de latencia en producción", "Pendiente", "2 semanas"],
]
make_table(["Prioridad", "", "Tarea", "Estado", "Fecha objetivo"], hoja_ruta, col_widths=[1.5, 2.5, 7, 2, 3.5])
table_note("Las tareas de prioridad 1 y 2 son bloqueantes para el uso del backend en "
           "integración real. Las tareas 3–8 son necesarias para un despliegue en "
           "producción pública seguro y mantenible.")

heading("3.6 Reproducibilidad", level=2)
table_label(6, "Estado de reproducibilidad del sistema al cierre del Sprint 13 de entregables")
make_table(["Ítem", "Estado"], [
    ["Entorno Python", ".venv310 (Python 3.10.20, PyTorch, Optuna) para entrenamiento e "
     "inferencia; .venv311 (Python 3.11.15, MediaPipe, Gradio) para demo"],
    ["Semilla aleatoria", "SEED = 42 en scripts/train_s27.py y todos los splits. No "
     "determinístico al 100 %: DataLoader usa WeightedRandomSampler y augmentation sin "
     "seed por worker"],
    ["Lockfile", "requirements.lock.txt con 198 paquetes a versión exacta (pip freeze "
     "sobre .venv310 real)"],
    ["Checkpoint", "bilstm_s27.onnx (3.9 MB) no versionado en git (excluido por "
     ".gitignore:23). Copia manual requerida para reproducir"],
    ["Test", "6/6 pasan con .venv310/bin/python -m pytest tests/ -v"],
], col_widths=[4, 12])
table_note("El no determinismo parcial del entrenamiento implica que dos corridas con la "
           "misma configuración pueden producir resultados ligeramente distintos "
           "(confirmado: v3 y v4 con misma config produjeron F1 = 0.4563 vs. 0.4426).")
page_break()

print("Capítulo IV...")
# ══════════════════════════════════════════════════════════════════════════
# CAPÍTULO IV
# ══════════════════════════════════════════════════════════════════════════
heading("Capítulo IV. Análisis y Discusión de Resultados", level=1)
heading("4.1 OE1 Precisión de Clasificación (Resultado Parcial)", level=2)
p("El modelo BiLSTM v4 (Sprint 27) alcanzó un F1-macro de 0.4426 en el conjunto de "
  "prueba interno, con exactitud Top-1 del 44.8 %, Top-3 del 57.2 % y Top-5 del 64.4 %. "
  "El F1-macro fue recomputado en vivo sobre el mismo split determinista "
  "(StratifiedShuffleSplit, semilla = 42, 15%), confirmando la ausencia de fuga de "
  "datos.", indent_first=1.25)
table_label(7, "Métricas de precisión del modelo BiLSTM v4 sobre el conjunto de prueba interno")
make_table(["Métrica", "Valor", "N de evaluación"], [
    ["F1-macro", "0.4426", "1 823 muestras, 96 clases"],
    ["Exactitud Top-1", "44.8 %", "1 823 muestras"],
    ["Exactitud Top-3", "57.2 %", "1 823 muestras"],
    ["Exactitud Top-5", "64.4 %", "1 823 muestras"],
    ["Meta declarada (HE1)", "F1-macro ≥ 0.70", "(no alcanzada)"],
], col_widths=[5, 5, 6])
table_note("F1-macro recomputado en vivo con el mismo split de entrenamiento. La meta "
           "declarada de F1 ≥ 0.70 no fue alcanzada tras 27 sprints. Top-k = porcentaje "
           "de muestras en que la clase correcta aparece entre los k candidatos de mayor "
           "confianza.")

figure_label(4, "Exactitud Top-k del modelo BiLSTM v4 — recomputada en vivo")
figure_image(FIGS / "fig_topk.png", width_in=4.5)
figure_note("El modo Top-3 (57.2 %) y Top-5 (64.4 %) son relevantes para el caso de uso de "
            "asistencia comunicativa con lista de candidatos, análogo al autocompletado "
            "de teclado (Bragg et al., 2019). N = 1 823 muestras de prueba, 96 clases.")

p("El análisis por clase revela distribución bimodal: señas icónicas con configuración "
  "manual distintiva (letras W, F, U, I, D) alcanzan F1 > 0.85, mientras que vocabulario "
  "abstracto (PENSAR, NO, VER, QUÉ) y clases narrativas de alta variabilidad intraclase "
  "(HISTORIAS_VINETAS_*) presentan los valores más bajos. Este patrón es coherente con "
  "Rastgoo et al. (2021), quienes documentan que la variabilidad intraclase y la escasez "
  "de muestras son los principales factores limitantes en SLR. La Figura 4-bis, incluida "
  "en el Anexo B, presenta la matriz de confusión completa de las 96 clases y el detalle "
  "de las mejores/peores clases por F1, calculadas en vivo sobre el mismo checkpoint.", indent_first=1.25)
p("Resultado parcial: La meta de F1 ≥ 0.70 no fue alcanzada. La brecha se atribuye a la "
  "cantidad de muestras disponibles por clase (~44 muestras promedio por clase), no a "
  "limitaciones de la arquitectura. La ampliación del corpus es la vía prioritaria de "
  "trabajo futuro.", indent_first=1.25)
p("Actualización (2026-07-19): se auditaron exhaustivamente los datos sin usar en el "
  "repositorio y se corrieron cuatro configuraciones adicionales de entrenamiento. Una de "
  "ellas (checkpoint S29, mismo vocabulario de 96 clases, +11% de muestras reales) alcanzó "
  "F1-test=0.4208 y fue el primer checkpoint del proyecto en pasar HE3 completo — incluido "
  "el estadístico KS — con márgenes amplios (ΔF1=0.0017, PSI=0.0042, KS p=0.8939). "
  "Comparado con v4 sobre un holdout reconstruido para estar limpio de ambos entrenamientos "
  "(342 muestras, 72 clases, ver ENTREGABLE_PLAN_DE_DESPLIEGUE_S13.md §11), un ensemble por "
  "promedio de probabilidades de v4 y S29 alcanzó F1=0.4424 — superando a ambos modelos "
  "individuales en esa comparación y heredando la generalización de S29. El sistema activo "
  "pasó de un modelo único a este ensemble de dos modelos.", indent_first=1.25)

heading("4.2 OE2 Latencia del Pipeline en Tiempo Real", level=2)
p("La Tabla 8 presenta las métricas de latencia medidas mediante un cliente WebSocket "
  "real contra el endpoint /predict/stream, incluyendo todos los componentes del "
  "pipeline: decodificación de frame, MediaPipe Holistic, buffering, normalización e "
  "inferencia ONNX.", indent_first=1.25)
table_label(8, "Latencia de inferencia del sistema por componente, medida en condiciones reales")
make_table(["Componente", "p50 (ms)", "p95 (ms)", "Máx. (ms)", "Estado vs. umbral"], [
    ["Inferencia ONNX (modelo)", "0.6", "0.9", "1.3", "< 200 ms, margen 333×"],
    ["Pipeline E2E (captura → texto)", "54.7", "58.7", "118.2", "< 200 ms, margen ~2.7×"],
    ["Meta declarada (HE2)", "< 200 ms", "< 200 ms", "< 200 ms", "Cumplido"],
], col_widths=[6, 2.5, 2.5, 2.5, 4.5])
table_note("E2E = extremo a extremo (end-to-end). Medición realizada con cliente "
           "WebSocket real sobre servidor local (protocolo TCP). El cuello de botella "
           "computacional es MediaPipe Holistic (~55 ms/frame). p50 = mediana; "
           "p95 = percentil 95. N = 100 inferencias consecutivas con 3 repeticiones.")

figure_label(5, "Latencia real medida — modelo ONNX aislado vs. pipeline E2E (escala log)")
figure_image(FIGS / "fig_latencia.png", width_in=5.5)
figure_note("La diferencia entre la latencia del modelo ONNX (< 1 ms) y el pipeline E2E "
            "(~55 ms) refleja el costo de MediaPipe Holistic. Con 200 ms de umbral "
            "objetivo, el sistema opera con un margen de seguridad de ~145 ms en el caso "
            "mediano y ~82 ms en el peor caso observado.")

heading("4.3 OE3 Generalización fuera de la Muestra", level=2)
p("La generalización fue evaluada sobre un holdout externo de ≈ 835 muestras separadas "
  "por grupo de señante antes del inicio del entrenamiento. La Tabla 9 presenta la "
  "evolución de las métricas de generalización a través de las cuatro corridas del "
  "Sprint 27.", indent_first=1.25)
table_label(9, "Evolución de métricas de generalización en las cuatro corridas del Sprint 27")
make_table(["Métrica", "v1", "v2", "v3 (mejor KS)", "v4 (activo)", "Umbral"], [
    ["ΔF1", "0.1311", "0.0509", "0.0785", "0.0406", "≤ 0.15 (margen 3.7×)"],
    ["PSI", "0.1281", "0.0818", "0.0184", "0.0288", "< 0.20 (margen 7×)"],
    ["KS D", "0.136", "0.117", "0.049", "0.065", "D crítico ≈ 0.045"],
    ["KS p-valor", "≈ 0", "≈ 0", "0.0168", "0.0011", "> 0.05"],
], col_widths=[3, 2, 2, 3, 3, 3.5])
table_note("El checkpoint v3 fue sobrescrito por v4 y no es recuperable. ΔF1 = F1-macro "
           "(prueba interna) − F1-macro (holdout externo), modelo sin reentrenamiento. "
           "PSI calculada sobre histogramas de activaciones de capa oculta final (10 "
           "bins). D crítico de KS calculado para N ≈ 1 800 muestras y α = 0.05.")

figure_label(6, "Comparación de métricas de generalización v1–v4 respecto a umbrales de aceptación")
figure_image(FIGS / "fig_he3_comparacion.png", width_in=6.0)
figure_note("ΔF1 y PSI cumplen sus umbrales en v4 con amplio margen. El estadístico KS no "
            "supera el umbral en ninguna corrida, resultado atribuido a la "
            "hipersensibilidad estadística del test ante tamaños de muestra grandes "
            "(Razali & Wah, 2011). Cuatro configuraciones distintas de granularidad de "
            "subgrupos fueron probadas sin lograr superar el umbral KS.")

p("La interpretación del resultado KS debe contextualizarse metodológicamente: con "
  "N ≈ 1 800 muestras, el valor crítico D para α = 0.05 es ≈ 0.045, un umbral "
  "excepcionalmente bajo que hace que el test sea hipersensible a diferencias triviales "
  "en la forma de la distribución. Razali & Wah (2011) documentan que, con muestras "
  "grandes, prácticamente cualquier diferencia distribucional, aunque sea prácticamente "
  "irrelevante, alcanza significancia estadística. Las métricas ΔF1 y PSI que miden "
  "directamente la brecha de rendimiento entre conjunto visto y no visto son más "
  "relevantes para el uso práctico del sistema y ambas cumplen sus umbrales con "
  "holgura.", indent_first=1.25)

heading("4.4 Hallazgo Adicional: Sesgo de Dominio en el Reconocimiento del Abecedario", level=2)
p("Un hallazgo adicional, posterior al cierre formal del Sprint 27, surgió al investigar "
  "por qué el sistema no reconocía el abecedario dactilológico a través de cámara en vivo "
  "o video de cuerpo completo, pese a que las letras aisladas figuran entre las clases de "
  "mejor F1 del modelo (W, F, U, I, D; F1 > 0.85, ver Anexo B). La causa raíz se identificó "
  "mediante inspección directa de los datos de entrenamiento: el 100 % de las muestras de "
  "la clase abecedario provienen de un conjunto externo de fotografías de mano en primer "
  "plano sin cuerpo visible, por lo que los 33 landmarks de pose del vector de "
  "características son sistemáticamente cero para esa clase durante el entrenamiento. El "
  "modelo aprendió, de forma válida dentro de esa distribución pero espuria en general, a "
  "usar «pose ≈ 0» como señal de que la seña observada es una letra.", indent_first=1.25)
p("Se verificó la hipótesis causalmente: sobre clips reales de cámara de cuerpo completo, "
  "forzar a cero el bloque de pose en el vector de entrada cambia la predicción del modelo "
  "de palabras genéricas de alta frecuencia (IR, ORIGINAL, DORMIR) a letras del abecedario, "
  "aunque no siempre la correcta. Sobre 120 muestras genuinas de entrenamiento, el modelo "
  "alcanza 88 % de exactitud Top-1, confirmando que el reconocimiento de la forma de la "
  "mano no está deteriorado — el problema es enteramente de dominio, no de arquitectura ni "
  "de capacidad del modelo.", indent_first=1.25)
p("Como mitigación parcial, se corrigió el flujo de reconocimiento de imagen estática: "
  "cuando el detector de landmarks no encuentra ninguna mano — situación frecuente en "
  "fotografías de mano sola, porque el recorte de región de interés de MediaPipe Holistic "
  "depende de una estimación de pose que resulta errónea sin cuerpo visible — se ejecuta "
  "como respaldo el modelo independiente mediapipe.solutions.hands, que no depende de "
  "pose; y si tampoco se detecta rostro, se descarta la pose estimada antes de construir "
  "el vector de características, replicando la distribución de entrenamiento. Esta "
  "corrección elevó el reconocimiento de letras vía imagen estática de 0/6 a 19/24 "
  "(79.2 %, IC95 Wilson [59.5 %, 90.8 %]). La misma corrección se evaluó — y se decidió no "
  "aplicar — sobre los flujos de cámara en vivo y video, porque en esos casos la ausencia "
  "de rostro puede deberse a que la mano cubre momentáneamente el rostro durante una seña "
  "real, no a que la imagen sea de mano sola; descartar la pose ahí degradaría la "
  "clasificación de palabras reales. La Figura 7 resume la evidencia cuantitativa.", indent_first=1.25)

figure_label(7, "Slices problemáticos — proporción de acierto con intervalo de confianza Wilson 95%")
figure_image(FIGS / "fig_slices_ic.png", width_in=6.2)
figure_note("El reconocimiento del abecedario por imagen estática mejora de 0 % a 79.2 % "
            "tras la corrección de dominio. Las dos barras inferiores corresponden a un "
            "hallazgo independiente ya documentado (Tabla 4, Riesgo R10): video sin "
            "segmentar rinde peor que clips ya aislados por el muestreo uniforme de "
            "/predict/video sobre archivos largos.")

p("Resultado parcial: la corrección para cámara en vivo y video de cuerpo completo queda "
  "como trabajo futuro. Ya existe en el repositorio material real de cuerpo completo de "
  "las 24 letras con landmarks de pose correctos (carpeta vocabulario_lsp_p_pkl/"
  "LETRAS-ABECEDARIO, con anotación ELAN de timestamps exactos por letra) nunca utilizado "
  "para entrenar, identificado como el camino directo para resolver esta limitación "
  "mediante fine-tuning del checkpoint activo.", indent_first=1.25)

heading("4.5 Hallazgo Adicional: Traducción de Narración Continua", level=2)
p("El sistema fue entrenado y evaluado sobre clips ya aislados —una seña por clip— pero "
  "el caso de uso de comunicación asistida real requiere reconocer señas dentro de "
  "narración continua, sin cortes manuales. Esta limitación, documentada desde etapas "
  "tempranas del proyecto (Anexo D), motivó una línea de trabajo dedicada a medirla y "
  "mejorarla con evidencia cuantitativa por primera vez en el proyecto, mediante la "
  "métrica estándar de reconocimiento de habla/lengua de señas continua: Word Error Rate "
  "(WER), la distancia de edición de Levenshtein entre la secuencia de glosas predicha y "
  "la secuencia real, normalizada por la longitud de la referencia.", indent_first=1.25)
p("Se identificaron dos recursos existentes en el repositorio nunca cruzados entre sí: "
  "4 092 glosas anotadas manualmente con marca de tiempo exacta en milisegundos sobre 27 "
  "videos narrativos completos (subtítulos SRT del corpus PUCP-DGI156), y keypoints ya "
  "extraídos con MediaPipe en ventanas deslizantes de 30 fotogramas sobre 26 de esos "
  "mismos videos, generados en un sprint previo pero nunca etiquetados a nivel de glosa "
  "individual. Cruzando ambos recursos se construyó un dataset de contexto narrativo real "
  "—a diferencia de los clips ya aislados del corpus principal—, reservando 5 videos "
  "completos, nunca utilizados en ningún entrenamiento, exclusivamente para medir WER.", indent_first=1.25)
p("Un primer modelo entrenado de forma aislada sobre este dataset (1 538 muestras, 51 "
  "clases) obtuvo peor WER que el sistema de producción (2.827 frente a 1.763), con una "
  "caída de generalización severa entre validación cruzada y videos genuinamente no "
  "vistos (F1 de 0.174 a 0.061) atribuible al bajo número de señantes de entrenamiento "
  "efectivos (15). Combinando estas muestras de contexto narrativo con el corpus grande "
  "de clips ya aislados —mismas fuentes que el corpus principal, reutilizando el "
  "mecanismo de rebalanceo de grupos por fuente cruzada ya validado en sprints anteriores "
  "para evitar fuga de datos— se ensambló un dataset combinado de 270 clases y 20 689 "
  "muestras (Tabla 10), sobre el cual dos reentrenamientos sucesivos redujeron el WER a "
  "1.134 y finalmente 1.027 — un 41.8 % por debajo del sistema de producción (Tabla 11, "
  "Figura 8).", indent_first=1.25)

table_label(10, "Fuentes de datos del dataset combinado utilizado para la línea de mejora de narración continua")
make_table(["Fuente", "Origen", "Muestras", "Clases"], [
    ["dgi156", "PUCP-DGI156 — narrativa", "3 642", "28"],
    ["dgi156_gloss", "PUCP-DGI156 — glosa individual", "2 688", "195"],
    ["vineta", "PUCP-DGI156 — narrativa (2.ª colección)", "3 684", "27"],
    ["vineta_gloss", "PUCP-DGI156 — glosa individual", "2 734", "195"],
    ["aec", "Aprendo en Casa / PeruSIL", "1 454", "123"],
    ["vocabulario_lsp_p", "PUCP-305", "252", "57"],
    ["abecedario", "Colección externa (imágenes de mano)", "3 600", "24"],
    ["glosa", "Colección interna del proyecto", "134", "61"],
    ["s31_continuo", "SRT × keypoints (construido en esta línea)", "2 501", "195"],
    ["Total (mín. 15 muestras/clase)", "—", "20 689", "270"],
], col_widths=[4.5, 6.5, 2.5, 2.5])
table_note("Excluida explícitamente: LSA64 (Lengua de Señas Argentina, no peruana). El "
           "modelo de producción (Tabla 1) usa un subconjunto filtrado a 96 clases; el "
           "dataset de 270 clases es exclusivo de esta línea de mejora.")

table_label(11, "WER real sobre 5 videos narrativos nunca vistos en entrenamiento — 3 corridas consecutivas vs. producción")
make_table(["Modelo", "WER medio", "N videos"], [
    ["S31 (aislado, 51 clases)", "2.827", "5"],
    ["S32 (combinado, split único)", "1.134", "5"],
    ["S33 (combinado, KFold(5) completo)", "1.027", "5"],
    ["Producción (ensemble v4+S29, 96 clases)", "1.763", "5"],
], col_widths=[9, 4, 3])
table_note("WER = distancia de Levenshtein entre secuencia de glosas predicha y real, "
           "normalizada por longitud de referencia. WER = 1.0 equivale a tantos errores "
           "como palabras reales. Valores por video en data/s33_wer_resultados.json.")

figure_label(8, "WER real — 3 corridas consecutivas de la línea de narración continua vs. producción")
figure_image(FIGS / "fig_wer_progresion.png", width_in=6.0)
figure_note("Mejora monótona en las 3 corridas (S31→S32→S33), sin retrocesos. WER=1.027 "
            "sigue por encima de 1.0 — la narración continua no queda resuelta, pero es la "
            "primera vez que un intento dedicado supera al sistema de producción en esta "
            "tarea.")

p("Resultado parcial: WER = 1.027 confirma que la traducción de narración continua sigue "
  "sin resolverse en sentido estricto (más errores que palabras de referencia en "
  "promedio), pero la mejora sostenida de tres corridas consecutivas (−63.7 % de WER "
  "acumulado) valida el enfoque de combinar contexto narrativo real con el corpus de "
  "clips aislados como vía de trabajo futuro, en vez de una limitación arquitectónica sin "
  "salida. El modelo de esta línea no reemplaza al ensemble de producción: su F1 sobre "
  "clasificación aislada (0.209, 270 clases) es inferior al del sistema activo (0.4424, "
  "96 clases) — es una mejora específica y medida para narración, no una actualización "
  "general del sistema.", indent_first=1.25)

heading("4.6 Discusión General", level=2)
p("Los resultados sitúan al sistema en un nivel de madurez comparable con la literatura "
  "para lenguas de señas de bajo recurso. El F1-macro de 0.4426 es inferior al estado "
  "del arte para ASL o DGS donde los mejores modelos superan el 90 % con corpus de "
  "decenas de miles de muestras, pero coherente con lo esperable para un corpus de 96 "
  "clases con ~44 muestras promedio por clase. Rastgoo et al. (2021) documentan que el "
  "desempeño en SLR está fuertemente correlacionado con el tamaño y la diversidad del "
  "corpus, y que los sistemas con menos de 10 000 muestras raramente superan el 65–70 % "
  "en vocabularios de más de 50 clases.", indent_first=1.25)
p("El cumplimiento del objetivo de latencia con margen de ~145 ms es un resultado sólido "
  "que se mantiene incluso en el peor caso observado (118.2 ms), coherente con los "
  "hallazgos de Jiang et al. (2021), quienes identifican la estimación de pose corporal "
  "como el cuello de botella dominante en pipelines de SLR basados en esqueleto.", indent_first=1.25)
p("La validación de extremo a extremo con demo interactiva, API WebSocket, suite de 6 "
  "tests automatizados (6/6 pasan) y despliegue público en HuggingFace Spaces representa "
  "una contribución de ingeniería que demuestra la viabilidad de un sistema accesible "
  "sin instalación para usuarios finales.", indent_first=1.25)
page_break()

print("Conclusiones y recomendaciones...")
# ══════════════════════════════════════════════════════════════════════════
# CONCLUSIONES
# ══════════════════════════════════════════════════════════════════════════
p("Conclusiones", bold=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=16)
c1 = doc.add_paragraph()
rich(c1, [("Objetivo General: ", True, False), ("Se desarrolló un sistema integral de "
          "comunicación inclusiva basado en Deep Learning para la traducción automática "
          "de la LSP a texto castellano, con cinco módulos funcionales validados de "
          "extremo a extremo: captura, extracción de características, segmentación, "
          "clasificación BiLSTM-ONNX e interfaz web con despliegue público.", False, False)])
c2 = doc.add_paragraph()
rich(c2, [("OE1 Precisión (parcialmente cumplido): ", True, False), ("El modelo BiLSTM v4 "
          "alcanzó F1-macro = 0.4426 (Top-3: 57.2 %, Top-5: 64.4 %), el mejor resultado "
          "histórico tras 27 sprints. La meta de F1 ≥ 0.70 no fue alcanzada; la brecha se "
          "atribuye a la cantidad de muestras por clase, no a la arquitectura.", False, False)])
c3 = doc.add_paragraph()
rich(c3, [("OE2 Latencia (cumplido): ", True, False), ("La latencia mediana del pipeline "
          "E2E es 54.7 ms (p95: 58.7 ms, máx: 118.2 ms), cumpliendo el umbral de 200 ms "
          "con margen de ~145 ms. El cuello de botella es MediaPipe Holistic (~55 ms), no "
          "el modelo ONNX (< 1 ms).", False, False)])
c4 = doc.add_paragraph()
rich(c4, [("OE3 Generalización (parcialmente cumplido): ", True, False), ("ΔF1 = 0.0406 "
          "(margen 3.7×) y PSI = 0.0288 (margen 7×) cumplen sus umbrales. El estadístico "
          "KS no supera el umbral de significancia (p = 0.0011) en ninguna de las 4 "
          "corridas, documentado como gap metodológico atribuible a hipersensibilidad "
          "estadística ante N ≈ 1 800 muestras.", False, False)])
page_break()

p("Recomendaciones", bold=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=16)
bullet_rich([("Ampliación del corpus LSP: ", True, False), ("Diseñar campaña de "
             "recolección con señantes de distintas regiones (Lima, Cusco, Arequipa, "
             "Puno), apuntando a ≥ 500 muestras por clase para superar la barrera de "
             "F1 = 0.70.", False, False)])
bullet_rich([("Exploración de arquitecturas Transformer: ", True, False), ("Evaluar "
             "Transformers sobre el mismo corpus siguiendo la metodología de Camgoz et "
             "al. (2020) para verificar si la brecha de F1 puede reducirse con la "
             "arquitectura sin ampliar el corpus.", False, False)])
bullet_rich([("Riesgo R9 (resuelto): ", True, False), ("Los tres ajustes de calidad "
             "(segmentación por pausas, filtro HISTORIAS_VINETAS_*, umbral de confianza "
             "0.20) fueron portados de demo/app_gradio.py a api/main.py y verificados con "
             "cliente WebSocket real (2026-07-18). Pendiente de seguimiento: mantener "
             "ambos flujos sincronizados ante futuros cambios de calidad.", False, False)])
bullet_rich([("Versionado del checkpoint ONNX: ", True, False), ("Versionar "
             "bilstm_s27.onnx con nombre de corrida (no nombre fijo) y publicarlo como "
             "artefacto en GitHub Releases o almacenamiento externo para garantizar la "
             "reproducibilidad.", False, False)])
bullet_rich([("Hardening del backend para producción: ", True, False), ("Restringir CORS, "
             "agregar límite de tamaño en UploadFile, implementar logging estructurado, "
             "configurar CI con pytest y realizar build real del contenedor Docker.", False, False)])
bullet_rich([("Continuar la línea de narración continua (§4.5): ", True, False), ("El WER "
             "bajó de 2.827 a 1.027 en tres corridas, pero sigue por encima de 1.0. "
             "Próximos pasos ya identificados: incorporar más videos narrativos con SRT "
             "disponibles sin extraer (data/external_lsp/dgi156_full/SRT.tar) y correr "
             "búsqueda de hiperparámetros real — las tres corridas de esta línea heredaron "
             "los hiperparámetros de un sprint muy anterior (S13) sin optimizar para este "
             "dataset específico.", False, False)])
bullet_rich([("Fine-tuning del abecedario con datos de cuerpo completo (§4.4): ", True, False),
             ("Ya existe en el repositorio material real de las 24 letras con pose "
              "correcta y anotación ELAN de timestamps (vocabulario_lsp_p_pkl/"
              "LETRAS-ABECEDARIO) nunca usado para entrenar — extraerlo y hacer "
              "fine-tuning del checkpoint activo es el camino directo para resolver el "
              "sesgo de dominio en cámara en vivo y video.", False, False)])
bullet_rich([("Exploración de arquitecturas Transformer: ", True, False), ("Evaluar "
             "Transformers sobre el mismo corpus siguiendo la metodología de Camgoz et "
             "al. (2020) para verificar si la brecha de F1 puede reducirse con la "
             "arquitectura sin ampliar el corpus.", False, False)])
page_break()

print("Referencias...")
# ══════════════════════════════════════════════════════════════════════════
# REFERENCIAS
# ══════════════════════════════════════════════════════════════════════════
p("Referencias Bibliográficas", bold=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=16)
referencias = [
    "Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly "
    "learning to align and translate. 3rd International Conference on Learning "
    "Representations (ICLR 2015). https://arxiv.org/abs/1409.0473",
    "Bai, J., Lu, F., & Zhang, K. (2019). ONNX: Open neural network exchange. GitHub. "
    "https://github.com/onnx/onnx",
    "Bragg, D., Koller, O., Bellard, M., Berke, L., Boudreault, P., Braffort, A., "
    "Caselli, N., Huenerfauth, M., Kacorri, H., Verhoef, T., Vogler, C., & Morris, M. R. "
    "(2019). Sign language recognition, generation, and translation: An "
    "interdisciplinary perspective. Proceedings of the 21st International ACM SIGACCESS "
    "Conference on Computers and Accessibility (ASSETS '19) (pp. 16–31). ACM. "
    "https://doi.org/10.1145/3308561.3353774",
    "Camgoz, N. C., Koller, O., Hadfield, S., & Bowden, R. (2020). Sign language "
    "transformers: Joint end-to-end sign language recognition and translation. "
    "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition "
    "(CVPR 2020) (pp. 10023–10033). https://doi.org/10.1109/CVPR42600.2020.01004",
    "De Coster, M., Van Herreweghe, M., & Dambre, J. (2020). Sign language recognition "
    "with transformer networks. Proceedings of the 12th International Conference on "
    "Language Resources and Evaluation (LREC 2020) (pp. 6018–6024). ELRA.",
    "Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural "
    "Computation, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735",
    "Instituto Nacional de Estadística e Informática. (2017). Primera Encuesta Nacional "
    "Especializada sobre Discapacidad 2012. INEI.",
    "Jiang, S., Sun, B., Wang, L., Bai, Y., Li, K., & Fu, Y. (2021). Skeleton aware "
    "multi-modal sign language recognition. Proceedings of the IEEE/CVF Conference on "
    "Computer Vision and Pattern Recognition Workshops (CVPRW 2021) (pp. 3413–3423). "
    "https://doi.org/10.1109/CVPRW53098.2021.00380",
    "Koller, O., Camgoz, N. C., Ney, H., & Bowden, R. (2020). Weakly supervised learning "
    "with multi-stream CNN-LSTM-HMM for sign language recognition and translation. IEEE "
    "Transactions on Pattern Analysis and Machine Intelligence, 42(11), 2793–2806. "
    "https://doi.org/10.1109/TPAMI.2019.2911077",
    "LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521, 436–444. "
    "https://doi.org/10.1038/nature14539",
    "Ley N.° 29535. (2010, 21 de mayo). Ley que otorga reconocimiento oficial a la "
    "Lengua de Señas Peruana. Congreso de la República del Perú.",
    "Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., Zhang, F., "
    "Chang, C. L., Yong, M. G., Lee, J., Chang, W. T., Hua, W., Georg, M., & Grundmann, "
    "M. (2019). MediaPipe: A framework for building perception pipelines. arXiv. "
    "https://arxiv.org/abs/1906.08172",
    "Nielsen, J. (1993). Usability engineering. Academic Press.",
    "Organización de las Naciones Unidas. (2006). Convención sobre los Derechos de las "
    "Personas con Discapacidad. ONU.",
    "Quionero-Candela, J., Sugiyama, M., Schwaighofer, A., & Lawrence, N. D. (Eds.). "
    "(2009). Dataset shift in machine learning. MIT Press.",
    "Rastgoo, R., Kiani, K., & Escalera, S. (2021). Sign language recognition: A deep "
    "survey. Expert Systems with Applications, 164, Article 113794. "
    "https://doi.org/10.1016/j.eswa.2020.113794",
    "Razali, N. M., & Wah, Y. B. (2011). Power comparisons of Shapiro-Wilk, "
    "Kolmogorov-Smirnov, Lilliefors and Anderson-Darling tests. Journal of Statistical "
    "Modeling and Analytics, 2(1), 21–33.",
    "Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. "
    "IEEE Transactions on Signal Processing, 45(11), 2673–2681. "
    "https://doi.org/10.1109/78.650093",
]
for ref in referencias:
    para = doc.add_paragraph()
    para.paragraph_format.left_indent = Cm(1.25)
    para.paragraph_format.first_line_indent = Cm(-1.25)
    para.paragraph_format.space_after = Pt(10)
    r = para.add_run(ref)
    r.font.name = "Times New Roman"; r.font.size = Pt(12)
page_break()

print("Anexos...")
# ══════════════════════════════════════════════════════════════════════════
# ANEXOS
# ══════════════════════════════════════════════════════════════════════════
p("Anexos", bold=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=16)
heading("Anexo A. Matriz de Consistencia", level=2)
p("La Matriz de Consistencia completa del estudio se presenta a continuación.")
tbl = doc.add_table(rows=1, cols=3)
tbl.style = "Table Grid"
hdr = tbl.rows[0].cells
for i, h in enumerate(["PROBLEMAS", "OBJETIVOS", "HIPÓTESIS"]):
    hdr[i].text = ""
    r = hdr[i].paragraphs[0].add_run(h)
    r.font.name = "Times New Roman"; r.font.size = Pt(10); r.bold = True
    set_cell_shading(hdr[i], GREY_HEAD)

consistencia = [
    ("Problema General\n¿En qué medida un sistema integral basado en Deep Learning "
     "permite realizar la traducción automática de la Lengua de Señas Peruana (LSP) a "
     "texto castellano?",
     "Objetivo General\nDesarrollar un sistema integral basado en Deep Learning para la "
     "traducción automática de la Lengua de Señas Peruana (LSP) a texto castellano.",
     "Hipótesis General\nEl sistema integral basado en Deep Learning permite realizar la "
     "traducción automática de la Lengua de Señas Peruana (LSP) a texto castellano "
     "mediante un desempeño cuantificable del modelo y del proceso de inferencia."),
    ("PO1: ¿En qué medida un modelo basado en Deep Learning permite mejorar el desempeño "
     "de la traducción automática de la Lengua de Señas Peruana (LSP) a texto "
     "castellano, alcanzando un F1-score ≥ 0.70?",
     "OE1: Diseñar y desarrollar un modelo basado en Deep Learning para la traducción "
     "automática de la Lengua de Señas Peruana (LSP) a texto castellano, alcanzando un "
     "desempeño medido mediante F1-score.",
     "HE1: El modelo basado en Deep Learning permite mejorar el desempeño de la "
     "traducción automática de la Lengua de Señas Peruana (LSP) a texto castellano, "
     "logrando un F1-score ≥ 0.70 durante la evaluación experimental."),
    ("PO2: ¿En qué medida la mejora del pipeline de inferencia basado en Deep Learning "
     "permite realizar la traducción automática de la Lengua de Señas Peruana (LSP) en "
     "tiempo real con una latencia inferior a 200 ms?",
     "OE2: Mejorar el pipeline de inferencia basado en Deep Learning para realizar la "
     "traducción automática de la Lengua de Señas Peruana (LSP) en tiempo real con una "
     "latencia inferior a 200 ms.",
     "HE2: La mejora del pipeline de inferencia basado en Deep Learning permite realizar "
     "la traducción automática de la Lengua de Señas Peruana (LSP) en tiempo real, "
     "alcanzando una latencia inferior a 200 ms sin afectar significativamente el "
     "desempeño del modelo."),
    ("PO3: ¿En qué medida la mejora de la capacidad de generalización del modelo basado "
     "en Deep Learning permite mantener el desempeño de la traducción automática de la "
     "Lengua de Señas Peruana (LSP) ante usuarios y datos no observados durante el "
     "entrenamiento?",
     "OE3: Mejorar la capacidad de generalización del modelo basado en Deep Learning "
     "para mantener el desempeño de la traducción automática de la Lengua de Señas "
     "Peruana (LSP) ante usuarios y datos no observados durante el entrenamiento.",
     "HE3: La mejora de la capacidad de generalización del modelo basado en Deep "
     "Learning permite mantener el desempeño de la traducción automática de la Lengua de "
     "Señas Peruana (LSP) ante usuarios y datos no observados durante el "
     "entrenamiento."),
]
for prob_, obj_, hip_ in consistencia:
    cells = tbl.add_row().cells
    for i, txt in enumerate([prob_, obj_, hip_]):
        cells[i].text = ""
        r = cells[i].paragraphs[0].add_run(txt)
        r.font.name = "Times New Roman"; r.font.size = Pt(9)

page_break()
heading("Matriz de Operacionalización de Variables", level=2)
tbl2 = doc.add_table(rows=1, cols=6)
tbl2.style = "Table Grid"
hdr2 = tbl2.rows[0].cells
for i, h in enumerate(["Variable", "Definición conceptual", "Definición operacional",
                        "Dimensiones", "Indicadores", "Escala / Instrumento"]):
    hdr2[i].text = ""
    r = hdr2[i].paragraphs[0].add_run(h)
    r.font.name = "Times New Roman"; r.font.size = Pt(9); r.bold = True
    set_cell_shading(hdr2[i], GREY_HEAD)

operacionalizacion = [
    ("Variable Independiente (VI): Sistema integral basado en Deep Learning",
     "Sistema inteligente que integra modelos de aprendizaje profundo, procesamiento de "
     "datos visuales e inferencia automática para aprender patrones espacio-temporales "
     "de la LSP y generar predicciones automáticas de traducción a texto castellano.",
     "Se implementa mediante un sistema basado en Deep Learning entrenado con datos de "
     "LSP, incorporando etapas de extracción de características, entrenamiento, "
     "validación e inferencia.",
     "Arquitectura del modelo",
     "Tipo de arquitectura, capas, parámetros entrenables, mecanismo de atención",
     "Nominal/Razón — ficha técnica y código fuente"),
    ("", "", "", "Proceso de entrenamiento",
     "Épocas, función de pérdida, optimizador, tasa de aprendizaje",
     "Razón — historial de entrenamiento"),
    ("", "", "", "Proceso de inferencia",
     "Tiempo de inferencia, latencia, rendimiento computacional",
     "Razón (ms) — benchmark de rendimiento"),
    ("Variable Dependiente (VD): Traducción automática de la LSP a texto castellano",
     "Capacidad de un sistema inteligente para interpretar secuencias de LSP y generar "
     "automáticamente una representación textual en castellano con precisión, rapidez y "
     "capacidad de adaptación.",
     "Se evalúa mediante el desempeño del modelo en la generación automática de texto "
     "castellano a partir de secuencias de señas, con métricas de aprendizaje "
     "automático y pruebas de robustez.",
     "Desempeño predictivo",
     "F1-score, Accuracy, Precisión, Recall, matriz de confusión",
     "Razón (%) — dataset de prueba y métricas del modelo"),
    ("", "", "", "Eficiencia en tiempo real",
     "Latencia de respuesta, tiempo total de inferencia, FPS",
     "Razón (ms, fps) — benchmark del sistema"),
    ("", "", "", "Capacidad de generalización",
     "Variación del desempeño ante usuarios/datos no observados (ΔF1, PSI, KS)",
     "Razón (%) — dataset externo, pruebas estadísticas"),
]
for row in operacionalizacion:
    cells = tbl2.add_row().cells
    for i, txt in enumerate(row):
        cells[i].text = ""
        r = cells[i].paragraphs[0].add_run(txt)
        r.font.name = "Times New Roman"; r.font.size = Pt(8.5)
page_break()

heading("Anexo B. Métricas Detalladas por Clase", level=2)
p("Las métricas de F1-score por cada una de las 96 clases del vocabulario LSP fueron "
  "recomputadas en vivo reproduciendo el split de test determinista (mismo utilizado en "
  "el entrenamiento de v4). Las Figuras 9 y 10 presentan la matriz de confusión completa "
  "y el detalle de las 15 mejores y 15 peores clases por F1 — ambas generadas a partir "
  "de inferencia real del checkpoint checkpoints/bilstm_s27.onnx, no de datos "
  "ilustrativos.", indent_first=1.25)

figure_label(9, "Matriz de confusión normalizada — 96 clases (test holdout real, N=1823)")
figure_image(FIGS / "fig_matriz_confusion.png", width_in=5.5)
figure_note("La diagonal dominante confirma que el modelo aprendió estructura de clase "
            "real. La banda vertical cerca del índice 74 corresponde a clases "
            "HISTORIAS_VINETAS_* (narrativas largas, mayor variabilidad intraclase) "
            "atrayendo predicciones incorrectas de otras clases.")

figure_label(10, "Mejores y peores 15 clases por F1-score (test holdout real)")
figure_image(FIGS / "fig_f1_por_clase.png", width_in=6.2)
figure_note("Las mejores 15 clases son mayormente letras del abecedario con seña muy "
            "distintiva (W, F, U, I, D — F1 > 0.85). Las peores incluyen clases con muy "
            "pocas muestras de entrenamiento (ORIGINAL) y vocabulario abstracto (PENSAR, "
            "NO, VER, QUÉ). La tabla completa de precisión/recall/F1 por las 96 clases "
            "(classification_report de scikit-learn) está disponible en "
            "logs/runs.csv del repositorio del proyecto.")

heading("Anexo C. Contratos de la API y Plan de Despliegue Completo", level=2)
p("Los esquemas de entrada y salida de todos los endpoints, el análisis completo de "
  "riesgos y la hoja de ruta de tareas pendientes se documentan en el Entregable Plan de "
  "Despliegue S13 (ENTREGABLE_PLAN_DE_DESPLIEGUE_S13.docx), adjunto a la presente tesis.")

heading("Anexo D. Declaración de Limitaciones", level=2)
bullet("El checkpoint v3 (mejor estadístico KS, F1 = 0.4563) fue sobrescrito durante el "
       "entrenamiento de v4 y no es recuperable.")
bullet("El sistema reconoce señas aisladas con segmentación por pausas; el "
       "reconocimiento continuo sin pausas está fuera del alcance de este estudio. "
       "Verificado con evidencia adicional (2026-07-18): sobre un video real de 112s "
       "con guion SRT de 109 glosas (≈58 señas/minuto), el segmentador cerró 33 "
       "segmentos de los cuales 24 se descartaron (7 por clase narrativa, 17 por "
       "confianza insuficiente) — ninguna de las palabras mostradas coincidió con el "
       "guion real. Además, de las 79 glosas únicas del video, solo 11 pertenecen al "
       "vocabulario de 96 clases del modelo, por lo que gran parte del contenido real "
       "es irreconocible incluso con segmentación perfecta. A partir de este hallazgo se "
       "abrió una línea de mejora dedicada (§4.5), con la primera medición WER real del "
       "proyecto: tres corridas consecutivas redujeron el WER de 2.827 a 1.027 sobre 5 "
       "videos narrativos nunca vistos, superando al sistema de producción (WER=1.763) "
       "— mejora real y medida, aunque la narración continua no queda resuelta en "
       "sentido estricto (WER sigue > 1.0).")
bullet("El estadístico KS no supera el umbral de significancia (p = 0.0011 en v4); esta "
       "limitación está documentada y analizada en §4.3.")
bullet("Los ajustes de calidad de la demo (segmentación por pausas, filtro narrativas, "
       "umbral 0.20) fueron portados a api/main.py y verificados con cliente WebSocket "
       "real (Riesgo R9, resuelto 2026-07-18). Persiste como limitación de capacidad, "
       "no de infraestructura, el hallazgo del punto anterior sobre narración continua.")
bullet("El sistema activo pasó de un modelo único (v4) a un ensemble de dos modelos "
       "(v4 + S29). El ensemble mejora el F1 promedio y la generalización, pero no "
       "garantiza mejorar cada predicción individual — un caso real observado en pruebas "
       "en vivo (clip de la clase NIÑO) que v4 clasificaba correctamente pasó a "
       "clasificarse mal con el ensemble, aunque el conjunto de clips de referencia "
       "(golden set) siguió clasificando correctamente. Es el comportamiento esperado de "
       "un ensemble por promedio, documentado con honestidad y no ocultado.")
bullet("El abecedario no se reconocía correctamente en cámara en vivo ni en video de "
       "cuerpo completo. El diagnóstico inicial (el segmentador por pausas interpreta la "
       "letra sostenida como silencio) resultó parcial; el análisis causal completo (§4.4) "
       "identificó como causa dominante un sesgo de dominio: el 100 % de las muestras de "
       "entrenamiento del abecedario provienen de fotografías de mano sin cuerpo, por lo "
       "que el modelo aprendió «pose ≈ 0» como atajo para reconocer una letra — señal "
       "ausente en cámara real. Corregido parcialmente para imagen estática (0 % → 79.2 %, "
       "§4.4); pendiente para video/cámara en vivo. No es una limitación del modelo — las "
       "letras aisladas alcanzan F1 > 0.85 y 88 % de exactitud Top-1 sobre datos de su "
       "propio dominio de entrenamiento, de las mejores clases del sistema — sino de la "
       "distribución de los datos de entrenamiento de esa clase específica.")

print(f"Guardando {OUT.name}...")
doc.save(str(OUT))
size_kb = OUT.stat().st_size / 1024
print(f"OK — {OUT.name} ({size_kb:.0f} KB) en {OUT}")
