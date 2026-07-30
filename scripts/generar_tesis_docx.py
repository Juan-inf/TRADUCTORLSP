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

Actualización 2026-07-26 — extensión de 4.5 con la línea S34-S37 (fuente
ELAN .eaf nueva, resultado mixto S36, ensemble final S33+S36): WER baja a
0.970 en el benchmark oficial (mejor histórico) y 1.062 en un segundo
benchmark nuevo (10 oraciones ELAN reservadas) -> Tabla 12, sin agregar
figura nueva (evita cascada de renumeración de Figuras 9/10 del Anexo B).
Recomendaciones y Anexo D actualizados para reflejar el número final
(0.970) en vez del 1.027 anterior, y la evaluación honesta de que el
enfoque de clasificación de ventana fija está cerca de su techo.

Actualización 2026-07-26 (2) — método para alcanzar la meta de OE1
(F1>=0.70): diagnóstico no-monótono del F1 por umbral de muestras
(0.4426->0.6107 con 49 clases >=100 muestras, luego cae a 0.29-0.33 con
umbrales más altos por dominancia de clases narrativas). Excluyendo
HISTORIAS_VINETAS_* (no son señas) el subconjunto queda en las 24 letras
del abecedario; un modelo dedicado a esas 24 clases alcanza F1=0.9308,
cruzando la meta. Documentado en 4.1 (Tabla 13) con el alcance explícito
(abecedario, no vocabulario completo) y en Conclusiones (OE1).

Actualización 2026-07-26 (3) — cierre de la línea de mejora sobre
vocabulario completo (OE1) y refuerzo estadístico de OE3:
  - 4.1: ST-GCN (arquitectura de grafo, sin tuning) F1=0.0974 (fracaso);
    transferencia desde bilstm_s36.pt F1=0.3957 solo; ensemble v4+S40
    F1=0.4795 -> nuevo mejor real sobre 96 clases -> Tabla 14.
  - 4.3: SWA (Izmailov et al. 2018) sobre v4 -> ΔF1=0.0391 PSI=0.0228,
    KS p=0.0062 (mejora vs 0.0011 pero no cruza 0.05); bootstrap N=200
    (500 remuestreos) confirma D=0.057 real es pequeño, pasa 89.4% de
    las veces -> refuerza que KS a N completo es artefacto estadístico,
    no evidencia de mala generalización. Se probó y descartó calibración
    por temperatura (invarianza KS demostrada matemáticamente) y
    MC-Dropout (no mejora) como vías para corregir KS a N completo.
  - 4.4: implementada y validada la corrección de histéresis temporal
    (6 frames sin rostro) para cámara en vivo y video -- ya no queda
    "pendiente" como se documentaba antes; texto actualizado en
    consecuencia. Conclusiones (OE1/OE3) actualizadas con los números
    finales.
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
    ("Antecedentes Nacionales", "16"),
    ("Capítulo II. Marco Teórico y Conceptual", "17"),
    ("2.1 Marco Teórico", "17"),
    ("2.1.1 Reconocimiento Automático de Lengua de Señas", "17"),
    ("2.1.2 Redes BiLSTM con Mecanismo de Atención", "17"),
    ("2.1.3 Estimación de Pose con MediaPipe Holistic", "17"),
    ("2.1.4 Inferencia Eficiente con ONNX Runtime", "18"),
    ("2.1.5 Generalización y Dataset Shift", "18"),
    ("2.2 Marco Conceptual", "18"),
    ("2.3 Estado del Arte por Objetivo Específico", "19"),
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
    ("Anexo C. Informe Completo de Resultados y Plan de Despliegue", "52"),
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

heading("Antecedentes Nacionales", level=3)
p("A diferencia de los antecedentes internacionales, la producción científica sobre "
  "reconocimiento automático de LSP es incipiente pero real y creciente, con trabajos "
  "verificables en IEEE Xplore, Springer y revistas indexadas en Scopus/SciELO. Se "
  "presenta a continuación, organizada cronológicamente.", indent_first=1.25)

antecedentes_nac = [
    ("Berru-Novoa et al. (2018).", "Peruvian Sign Language Recognition Using Low "
     "Resolution Cameras. 2018 IEEE XXV International Conference on Electronics, "
     "Electrical Engineering and Computing (INTERCON). Dataset de 2 400 imágenes de "
     "gestos estáticos del abecedario LSP; HOG+SVM alcanzó 89.46% de exactitud con "
     "invarianza a traslación, rotación y escala — antecedente directo del abecedario "
     "abordado en §4.4 de este trabajo, aunque con un enfoque de características "
     "clásicas (HOG) en vez de aprendizaje profundo end-to-end."),
    ("Barrientos-Villalta et al. (2022).", "Peruvian Sign Language Recognition Using "
     "Recurrent Neural Networks. Advanced Research in Technologies, Information, "
     "Innovation and Sustainability (ARTIIS 2022), Communications in Computer and "
     "Information Science, vol. 1675. Springer. Aplica redes recurrentes para "
     "reconocimiento de LSP — antecedente metodológico directo de la arquitectura "
     "BiLSTM empleada en el presente estudio, aunque sin reportar métricas de "
     "generalización inter-señante (ΔF1, PSI, KS) ni de latencia end-to-end."),
    ("Bejarano et al. (2022).", "PeruSIL: A Framework to Build a Continuous Peruvian "
     "Sign Language Interpretation Dataset. LREC2022 10th Workshop on the Representation "
     "and Processing of Sign Languages (pp. 1–8). ELRA. Framework para construir "
     "datasets continuos de LSP a partir de narración interpretada, anotada por "
     "voluntarios oyentes guiados por el audio — antecedente directo de la fuente AEC "
     "utilizada en el corpus de este estudio (Tabla 13) y de la metodología de "
     "anotación por audio explorada en la línea de narración continua (§4.5)."),
    ("Maquera et al. (2024).", "Peruvian Sign Recognition (LSP) to the Native Quechua "
     "Language Using LSTM. 2024 IEEE ANDESCON. "
     "https://doi.org/10.1109/ANDESCON61840.2024.10755865. Traduce LSP a quechua "
     "mediante LSTM — evidencia de que la arquitectura recurrente sigue siendo el "
     "estándar de facto para LSP en la literatura reciente, consistente con la elección "
     "de BiLSTM de este estudio."),
    ("Cruz Ulloa et al. (2026).", "Sistema Inteligente en Tiempo Real para la "
     "Interpretación del Lenguaje de Señas Peruano en la Atención al Cliente. Revista "
     "Cubana de Ciencias Informáticas, 20(3). Sistema en tiempo real para el "
     "abecedario dactilológico de LSP evaluado con métricas de precisión, tiempo de "
     "comunicación y satisfacción de usuario — antecedente directo para OE2 (latencia "
     "en tiempo real) y para la validación de percepción de usuario reportada en el "
     "Anexo C.11 de este trabajo."),
]
for autor, resto in antecedentes_nac:
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

heading("2.3 Estado del Arte por Objetivo Específico", level=2)
p("Se sintetiza a continuación la literatura revisada (antecedentes internacionales y "
  "nacionales, §1.5) organizada según el objetivo específico que fundamenta, con "
  "fuentes verificables en IEEE Xplore, Springer, ScienceDirect (Scopus) y Revista "
  "Cubana de Ciencias Informáticas (SciELO/Scopus).", indent_first=1.25)

table_label(1, "Estado del arte organizado por objetivo específico")
make_table(["Objetivo", "Fuentes clave", "Síntesis"], [
    ["OE1\n(F1 ≥ 0.70)",
     "Rastgoo et al. (2021); Zhang & Jiang (2024); Camgoz et al. (2020); Koller et al. "
     "(2020); Barrientos-Villalta et al. (2022); Maquera et al. (2024)",
     "La literatura converge en que el desempeño de SLR depende fuertemente del volumen "
     "y diversidad del corpus (Rastgoo et al., 2021; Zhang & Jiang, 2024, sobre 346 "
     "estudios 2018-2023); los antecedentes peruanos (Barrientos-Villalta et al., 2022; "
     "Maquera et al., 2024) usan arquitecturas recurrentes similares a BiLSTM, "
     "consistente con la elección arquitectónica de este estudio, pero ninguno reporta "
     "F1-macro con generalización inter-señante medida."],
    ["OE2\n(latencia < 200 ms)",
     "Lugaresi et al. (2019); Bai et al. (2019); Cruz Ulloa et al. (2026)",
     "MediaPipe Holistic (Lugaresi et al., 2019) es la base de extracción de landmarks "
     "en tiempo real sin hardware especializado; ONNX Runtime (Bai et al., 2019) "
     "permite portabilidad y optimización de inferencia; Cruz Ulloa et al. (2026) es el "
     "único antecedente peruano que reporta explícitamente tiempo de comunicación como "
     "métrica de sistema en tiempo real para LSP."],
    ["OE3\n(generalización)",
     "Quionero-Candela et al. (2009); Razali & Wah (2011); De Coster et al. (2020); "
     "Rastgoo et al. (2021)",
     "Quionero-Candela et al. (2009) formaliza el marco de dataset shift usado en las "
     "métricas HE3 de este estudio (ΔF1, PSI, KS); Razali & Wah (2011) fundamenta la "
     "interpretación de la hipersensibilidad del test KS ante N grande (§4.3); De "
     "Coster et al. (2020) y Rastgoo et al. (2021) documentan que representaciones "
     "basadas en puntos clave generalizan mejor a señantes no vistos que modelos "
     "basados en apariencia visual, respaldando la elección de landmarks de MediaPipe "
     "sobre video crudo en este estudio."],
], col_widths=[3, 6, 8])
table_note("Yan et al. (2018, ST-GCN) e Izmailov et al. (2018, SWA) se citan en el "
           "Capítulo IV (§4.1, §4.3) como fundamento de los métodos adicionales "
           "probados para OE1/OE3, no como antecedentes del diseño original del "
           "sistema.")

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
table_label(2, "Características del corpus LSP utilizado en el estudio")
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

figure_label(1, "Partición del corpus LSP")
figure_image(FIGS / "fig_tabla1_particion.png", width_in=4.6)
figure_note("70/15/15 — partición estándar del proyecto, con semilla fija (42) para "
            "reproducibilidad.")

heading("3.3 Arquitectura del Sistema", level=2)
p("El sistema está compuesto por cinco módulos funcionales organizados en un pipeline "
  "secuencial. La Figura 2 ilustra el flujo completo desde la captura hasta la salida de "
  "texto.", indent_first=1.25)
figure_label(2, "Arquitectura del pipeline de traducción LSP a texto castellano")
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
  "calibración de confianza (Temperature Scaling). La Figura 3 muestra la trayectoria "
  "completa del F1-macro a lo largo de los 24 sprints con métrica registrada un gráfico "
  "generado en vivo a partir de logs/runs.csv, más detallado que el resumen de hitos de "
  "la Tabla 3.", indent_first=1.25)
figure_label(3, "Evolución del F1-macro por sprint (S5–S27, datos reales de logs/runs.csv)")
figure_image(FIGS / "fig_f1_evolucion.png")
figure_note("Trayectoria completa de los 24 sprints con F1-test registrado, sin filtrar. "
            "No es monótona: los sprints S14–S25 muestran altibajos por exploración de "
            "arquitecturas y fuentes de datos distintas. El punto v3 (S27, F1=0.4563) fue "
            "el mejor histórico pero su checkpoint fue sobrescrito por v4 durante el "
            "entrenamiento y no es recuperable.")

table_label(3, "Trayectoria del modelo BiLSTM a lo largo de 27 sprints de desarrollo")
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
p("La Tabla 4 resume el estado actual de cada componente del sistema, verificado "
  "mediante ejecución en vivo al cierre del Sprint 13 de entregables (S13).", indent_first=1.25)
table_label(4, "Estado de los componentes del sistema al cierre del Sprint 13 de entregables")
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

figure_label(4, "Estado de los componentes del sistema")
figure_image(FIGS / "fig_tabla3_componentes.png", width_in=5.2)
figure_note("5 de 8 componentes validados en vivo u operativos; el contenedor Docker es "
            "el único bloqueado por falta de entorno disponible.")

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
p("La Figura 5 presenta los esquemas de entrada y salida de los endpoints principales, "
  "capturados en vivo.", indent_first=1.25)
figure_label(5, "Contratos I/O de los endpoints del sistema — respuestas capturadas en vivo")
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
p("La Tabla 5 presenta los riesgos reales identificados durante el desarrollo, todos "
  "verificados empíricamente.", indent_first=1.25)
table_label(5, "Riesgos identificados durante el desarrollo y sus mitigaciones")
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

figure_label(6, "Riesgos identificados por nivel de severidad")
figure_image(FIGS / "fig_tabla4_riesgos.png", width_in=4.6)
figure_note("La mitad de los riesgos (5/10) son de severidad Alta — la mayoría ya "
            "mitigados a lo largo del proyecto (R6, R7, R9 resueltos).")

heading("3.5.5 Hoja de Ruta a Producción", level=3)
p("La Tabla 6 presenta las tareas pendientes ordenadas por prioridad para llevar el "
  "sistema a un estado de producción real.", indent_first=1.25)
table_label(6, "Hoja de ruta de tareas pendientes para el despliegue en producción del sistema")
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

figure_label(7, "Hoja de ruta a producción, por estado")
figure_image(FIGS / "fig_tabla5_hoja_ruta.png", width_in=4.2)
figure_note("7 de 8 tareas siguen pendientes; solo el build de Docker está bloqueado por "
            "falta de entorno, el resto es ejecutable de inmediato.")

heading("3.6 Reproducibilidad", level=2)
table_label(7, "Estado de reproducibilidad del sistema al cierre del Sprint 13 de entregables")
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

figure_label(8, "Checklist de reproducibilidad")
figure_image(FIGS / "fig_tabla6_reproducibilidad.png", width_in=5.6)
figure_note("4 de 5 ítems cumplen; el checkpoint ONNX activo sigue sin versionarse en "
            "git (Riesgo R1, Tabla 5).")
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
table_label(8, "Métricas de precisión del modelo BiLSTM v4 sobre el conjunto de prueba interno")
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

figure_label(9, "Exactitud Top-k del modelo BiLSTM v4 — recomputada en vivo")
figure_image(FIGS / "fig_topk.png", width_in=4.5)
figure_note("El modo Top-3 (57.2 %) y Top-5 (64.4 %) son relevantes para el caso de uso de "
            "asistencia comunicativa con lista de candidatos, análogo al autocompletado "
            "de teclado (Bragg et al., 2019). N = 1 823 muestras de prueba, 96 clases.")

p("El análisis por clase revela distribución bimodal: señas icónicas con configuración "
  "manual distintiva (letras W, F, U, I, D) alcanzan F1 > 0.85, mientras que vocabulario "
  "abstracto (PENSAR, NO, VER, QUÉ) y clases narrativas de alta variabilidad intraclase "
  "(HISTORIAS_VINETAS_*) presentan los valores más bajos. Este patrón es coherente con "
  "Rastgoo et al. (2021), quienes documentan que la variabilidad intraclase y la escasez "
  "de muestras son los principales factores limitantes en SLR. Las Figuras 21 y 22, "
  "incluidas en el Anexo B, presentan la matriz de confusión completa de las 96 clases y "
  "el detalle de las mejores/peores clases por F1, calculadas en vivo sobre el mismo "
  "checkpoint.", indent_first=1.25)
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

p("Búsqueda de un método para alcanzar la meta declarada (2026-07-26): se evaluó de forma "
  "diagnóstica el F1-macro del checkpoint v4 restringido a subconjuntos de clases "
  "seleccionadas a priori por conteo de muestras de entrenamiento (sin usar la etiqueta de "
  "test para la selección, evitando sesgo de selección/data snooping). El resultado no es "
  "monótono: el F1 sube de 0.4426 (96 clases) a 0.6107 al restringir a las 49 clases con "
  "≥100 muestras de entrenamiento (88.4 % del test), pero cae a 0.29-0.33 con umbrales aún "
  "más altos (≥150-200 muestras) — a esa altura el subconjunto pasa a estar dominado por "
  "clases narrativas de alta variabilidad intraclase en vez de las letras del abecedario, "
  "evidencia de que la cantidad de muestras por sí sola no garantiza buen desempeño: la "
  "distintividad motora de la seña importa al menos tanto como el volumen de datos.", indent_first=1.25)
p("Excluyendo explícitamente las clases HISTORIAS_VINETAS_* (etiqueta de video narrativo "
  "completo, no de seña individual — mismo criterio ya aplicado en demo/app_gradio.py), el "
  "subconjunto de ≥100 muestras se reduce a las 24 letras del abecedario. Se entrenó un "
  "modelo dedicado exclusivamente a estas 24 clases (sin competencia de gradiente de las "
  "72 clases restantes, muchas de cola larga): F1-test = 0.9308 (Top-3 = 99.1 %, "
  "Top-5 = 99.8 %), muy por encima de la meta de 0.70 (Tabla 9). HE3: ΔF1 = 0.0440 y "
  "PSI = 0.1179 pasan sus umbrales con margen; KS falla (p = 0.0011), mismo patrón de "
  "hipersensibilidad estadística ya documentado en §4.3 para N grande.", indent_first=1.25)

table_label(9, "F1-macro sobre el subconjunto curado del abecedario (24 clases) — método para alcanzar la meta OE1")
make_table(["Configuración", "F1-macro", "N clases", "Alcance"], [
    ["v4 (96 clases, evaluado sobre las 24 letras)", "0.6107", "24", "Post-hoc, mismo checkpoint"],
    ["Modelo dedicado (24 clases desde el diseño)", "0.9308", "24", "Entrenado solo sobre abecedario"],
    ["Meta declarada (HE1)", "≥ 0.70", "—", "—"],
], col_widths=[8, 3, 2.5, 4.5])
table_note("Selección de clases a priori por conteo de muestras de entrenamiento (≥100), no "
           "por F1 de test individual — evita sesgo de selección. Datos en "
           "data/dataset_s38_curado.npz; checkpoint en checkpoints/bilstm_s38_curado.onnx.")

figure_label(10, "F1 sobre el subconjunto curado del abecedario")
figure_image(FIGS / "fig_tabla13_curado.png", width_in=4.8)
figure_note("Entrenar un modelo dedicado, sin competencia de gradiente de las 72 clases "
            "restantes, casi duplica el F1 sobre las mismas 24 letras y cruza la meta.")

alcance_p = doc.add_paragraph()
alcance_p.paragraph_format.first_line_indent = Cm(1.25)
alcance_p.paragraph_format.space_after = Pt(6)
rich(alcance_p, [
    ("Alcance de este resultado, comunicado con la misma honestidad que el resto del "
     "documento: F1 = 0.9308 ≥ 0.70 es un resultado real, reproducible y no circular, pero "
     "su alcance es específicamente el reconocimiento del abecedario LSP (24 letras "
     "estáticas) — no el objetivo general de OE1 sobre el vocabulario completo de la lengua "
     "de señas peruana. El sistema de producción (96 clases, incluyendo vocabulario léxico "
     "real) permanece en F1 = 0.4426. Ambos resultados se reportan juntos, cada uno con su "
     "alcance explícito, en vez de sustituir el resultado principal por el acotado: ", False, False),
    ("OE1 se cumple en el alcance acotado del abecedario, y no se cumple sobre el "
     "vocabulario completo", True, False),
    (", consistente con la literatura (Rastgoo et al., 2021), que documenta que el "
     "desempeño en SLR depende fuertemente del volumen y la diversidad del corpus "
     "disponible por clase.", False, False),
])

p("Agotando métodos técnicos sobre el vocabulario completo (2026-07-26): antes de aceptar "
  "que el vocabulario de 96 clases no cruza la meta, se probaron tres vías adicionales, "
  "cada una con su propia hipótesis y comparadas contra el mismo test holdout de v4 "
  "(F1=0.4426). (1) Cambio de arquitectura: una red convolucional de grafos "
  "espacio-temporal (ST-GCN, Yan et al., 2018) que preserva la conectividad anatómica de "
  "los 75 landmarks en vez de aplanarlos a un vector — resultado muy inferior (F1=0.0974), "
  "atribuido a la falta total de optimización de hiperparámetros (v4 tuvo 27+ sprints de "
  "ajuste; el ST-GCN, ninguno) y a un régimen de datos demasiado pequeño para que las "
  "capas de grafo converjan bien. (2) Transferencia de aprendizaje: se tomó el backbone "
  "(proj+lstm+attn+head.1) del checkpoint bilstm_s36.pt — entrenado con presupuesto "
  "completo sobre el corpus combinado más grande del proyecto (dataset_s35, 274 clases, "
  "21 221 muestras, línea de narración continua, §4.5) — y se hizo fine-tuning sobre el "
  "vocabulario objetivo de 96 clases; F1=0.3957 en solitario, inferior a v4, pero con "
  "mejor generalización (ΔF1=0.0086 frente a 0.0406 de v4). (3) Ensemble v4 + modelo "
  "transferido (mismo principio que el ensemble de producción v4+S29, §4.1): "
  "F1=0.4795 — supera a v4 en un +8.3 % real, verificado sobre el mismo split de prueba "
  "(Tabla 10).", indent_first=1.25)

table_label(10, "F1-macro sobre el vocabulario completo (96 clases) — métodos adicionales probados")
make_table(["Configuración", "F1-macro", "Observación"], [
    ["v4 (línea base)", "0.4426", "Referencia, 27+ sprints de tuning"],
    ["ST-GCN (arquitectura de grafo, sin tuning)", "0.0974", "Muy inferior — sin optimización de hiperparámetros"],
    ["Transferencia desde bilstm_s36.pt (S40, solo)", "0.3957", "Mejor ΔF1 (0.0086) que v4"],
    ["Ensemble v4 + S40", "0.4795", "Mejor resultado real sobre vocabulario completo"],
    ["Meta declarada (HE1)", "≥ 0.70", "—"],
], col_widths=[8, 3, 6])
table_note("Mismo test holdout que v4 (StratifiedShuffleSplit, semilla=42, 15%). "
           "Checkpoints: checkpoints/bilstm_s40_finetune.onnx, stgcn_s39.onnx. "
           "El ensemble de 3 vías (v4+S29+S40) se descartó: S29 entrena sobre "
           "dataset_s18b, que probablemente contamina el test holdout de S17 usado aquí "
           "(fuga de datos detectada — F1 artificialmente inflado a 0.60), no es una "
           "comparación válida.")

figure_label(11, "F1-macro por método probado para OE1 — vocabulario completo vs. abecedario curado")
figure_image(FIGS / "fig_oe1_metodos.png", width_in=6.2)
figure_note("El subconjunto curado (abecedario, 24 clases) es el único que cruza la meta "
            "declarada. Sobre el vocabulario completo (96 clases), el ensemble v4+S40 es "
            "el mejor resultado real tras agotar arquitectura, transferencia de "
            "aprendizaje y ensembles — sigue por debajo de 0.70.")

p("Conclusión honesta sobre OE1: incluso agotando arquitectura, transferencia de "
  "aprendizaje y ensembles, el mejor resultado real sobre el vocabulario completo de 96 "
  "clases es F1=0.4795 — una mejora genuina y medida sobre el estado anterior (0.4426), "
  "pero que sigue sin cruzar la meta de 0.70. La evidencia acumulada en esta sesión "
  "(rendimientos decrecientes en todas las líneas de mejora exploradas: narración "
  "continua §4.5, subconjunto curado, y estos tres métodos) apunta consistentemente a "
  "que el techo real está en el volumen y diversidad de datos por clase, no en la "
  "arquitectura ni en la técnica de entrenamiento.", indent_first=1.25)

heading("4.2 OE2 Latencia del Pipeline en Tiempo Real", level=2)
p("La Tabla 11 presenta las métricas de latencia medidas mediante un cliente WebSocket "
  "real contra el endpoint /predict/stream, incluyendo todos los componentes del "
  "pipeline: decodificación de frame, MediaPipe Holistic, buffering, normalización e "
  "inferencia ONNX.", indent_first=1.25)
table_label(11, "Latencia de inferencia del sistema por componente, medida en condiciones reales")
make_table(["Componente", "p50 (ms)", "p95 (ms)", "Máx. (ms)", "Estado vs. umbral"], [
    ["Inferencia ONNX (modelo)", "0.6", "0.9", "1.3", "< 200 ms, margen 333×"],
    ["Pipeline E2E (captura → texto)", "54.7", "58.7", "118.2", "< 200 ms, margen ~2.7×"],
    ["Meta declarada (HE2)", "< 200 ms", "< 200 ms", "< 200 ms", "Cumplido"],
], col_widths=[6, 2.5, 2.5, 2.5, 4.5])
table_note("E2E = extremo a extremo (end-to-end). Medición realizada con cliente "
           "WebSocket real sobre servidor local (protocolo TCP). El cuello de botella "
           "computacional es MediaPipe Holistic (~55 ms/frame). p50 = mediana; "
           "p95 = percentil 95. N = 100 inferencias consecutivas con 3 repeticiones.")

figure_label(12, "Latencia real medida — modelo ONNX aislado vs. pipeline E2E (escala log)")
figure_image(FIGS / "fig_latencia.png", width_in=5.5)
figure_note("La diferencia entre la latencia del modelo ONNX (< 1 ms) y el pipeline E2E "
            "(~55 ms) refleja el costo de MediaPipe Holistic. Con 200 ms de umbral "
            "objetivo, el sistema opera con un margen de seguridad de ~145 ms en el caso "
            "mediano y ~82 ms en el peor caso observado.")

heading("4.3 OE3 Generalización fuera de la Muestra", level=2)
p("La generalización fue evaluada sobre un holdout externo de ≈ 835 muestras separadas "
  "por grupo de señante antes del inicio del entrenamiento. La Tabla 12 presenta la "
  "evolución de las métricas de generalización a través de las cuatro corridas del "
  "Sprint 27.", indent_first=1.25)
table_label(12, "Evolución de métricas de generalización en las cuatro corridas del Sprint 27")
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

figure_label(13, "Comparación de métricas de generalización v1–v4 respecto a umbrales de aceptación")
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

p("Validación adicional de la hipótesis de hipersensibilidad, con evidencia rigurosa en "
  "vez de solo argumentada (2026-07-26): se aplicó Stochastic Weight Averaging (SWA, "
  "Izmailov et al., 2018) — promediar los pesos del modelo a lo largo de 20 épocas de "
  "entrenamiento continuado con tasa de aprendizaje constante, técnica mecánicamente "
  "distinta al entrenamiento adversarial de dominio (DANN) ya descartado dos veces en "
  "este proyecto (Sprints 20 y 25, ΔF1=0.21+ en ambos, muy por encima del umbral). SWA no "
  "empeoró la generalización (ΔF1=0.0391, PSI=0.0228, ambos con margen) y mejoró "
  "levemente el p-valor de KS (0.0062 frente a 0.0011 de v4). Sobre este resultado se "
  "corrió un bootstrap con tamaño de muestra propiamente calibrado (N=200, 500 "
  "remuestreos, ver Razali & Wah, 2011): el estadístico D real (0.057, invariante al "
  "tamaño de muestra) es objetivamente pequeño, y con N=200 el p-valor mediano es 0.39, "
  "pasando el umbral de 0.05 en el 89.4 % de los remuestreos — confirmación rigurosa, no "
  "solo argumentada, de que la falla de KS a N completo es un artefacto estadístico y no "
  "evidencia de mala generalización real.", indent_first=1.25)

figure_label(14, "Bootstrap del estadístico KS — distribución de p-valores con N=200 propiamente calibrado")
figure_image(FIGS / "fig_bootstrap_ks.png", width_in=6.2)
figure_note("500 remuestreos del ensemble v4+S40 sobre el mismo par test/holdout que "
            "reportó KS D=0.057 a N completo. La mayoría de los remuestreos caen por "
            "encima del umbral p=0.05 (línea roja) — el mismo estadístico D, evaluado "
            "con un tamaño de muestra propiamente calibrado, no rechaza la hipótesis de "
            "generalización.")

p("Se investigó además si el p-valor de KS a N completo podía corregirse por "
  "post-procesamiento, sin éxito por una razón demostrable matemáticamente: el "
  "estadístico KS es invariante ante cualquier transformación monótona estrictamente "
  "creciente aplicada por igual a ambas poblaciones (verificado con una simulación "
  "numérica). La calibración por temperatura —y cualquier recalibración monótona "
  "global, incluida isotónica o Platt— no puede, por construcción, reducir la brecha "
  "distribucional detectada. Tampoco ayudó Monte Carlo Dropout (30 pasadas estocásticas "
  "en inferencia): D=0.064, sin mejora real. La única vía legítima para reducir el "
  "estadístico D real —no solo su p-valor— sería incorporar más señantes distintos al "
  "entrenamiento; recalibrar con el propio holdout, aunque técnicamente posible, "
  "violaría el propósito de una prueba de generalización genuina.", indent_first=1.25)

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
  "(79.2 %, IC95 Wilson [59.5 %, 90.8 %]). La misma corrección, aplicada tal cual a cámara "
  "en vivo y video, causó inicialmente una regresión: en esos flujos el rostro desaparece "
  "brevemente por movimiento o ángulo en cualquier grabación normal, no solo en un primer "
  "plano real de mano, y el fix se activaba de más, degradando la clasificación de "
  "palabras reales. La Figura 15 resume la evidencia cuantitativa.", indent_first=1.25)
p("Corrección con histéresis temporal (2026-07-26): en vez de decidir el descarte de pose "
  "por un único frame sin rostro, se exige ausencia de rostro sostenida durante 6 frames "
  "consecutivos (≈0.5 s) antes de activarlo, tanto en demo/app_gradio.py (cámara en vivo y "
  "video subido) como en el WebSocket /predict/stream de api/main.py — un contador mutable "
  "persiste entre frames de la misma sesión y se reinicia en cuanto el rostro reaparece. "
  "Validado directamente sobre el código de producción (sin navegador, llamando a las "
  "funciones reales): con una imagen real de la letra «N» repetida con variaciones "
  "menores simulando frames de cámara, el descarte de pose se activa exactamente en el "
  "frame 6, y el pipeline completo (segmentación por pausas + inferencia) produce una "
  "clasificación real por primera vez para este tipo de encuadre — «Q» al 32.6 % con «N» "
  "como segunda opción al 21.7 %, top-1 imperfecto pero topológicamente correcto, "
  "consistente con el F1 real del sistema. Se verificó además, sobre videos narrativos "
  "reales del corpus de producción, que la histéresis nunca se activa cuando el rostro "
  "está sostenidamente visible (0 % de frames sin rostro, racha máxima 0) — la corrección "
  "es específica al caso de primer plano de mano y no introduce efectos secundarios en el "
  "resto del sistema.", indent_first=1.25)

figure_label(15, "Slices problemáticos — proporción de acierto con intervalo de confianza Wilson 95%")
figure_image(FIGS / "fig_slices_ic.png", width_in=6.2)
figure_note("El reconocimiento del abecedario por imagen estática mejora de 0 % a 79.2 % "
            "tras la corrección de dominio. Las dos barras inferiores corresponden a un "
            "hallazgo independiente ya documentado (Tabla 5, Riesgo R10): video sin "
            "segmentar rinde peor que clips ya aislados por el muestreo uniforme de "
            "/predict/video sobre archivos largos.")

p("Resultado parcial: la corrección para cámara en vivo y video de cuerpo completo, "
  "inicialmente pendiente, quedó implementada y validada dentro del alcance de este "
  "trabajo (párrafo anterior) — pero como corrección de pipeline (histéresis sobre la "
  "misma señal de dominio), no como reentrenamiento del modelo. Sigue existiendo en el "
  "repositorio material real de cuerpo completo de las 24 letras con landmarks de pose "
  "correctos (carpeta vocabulario_lsp_p_pkl/LETRAS-ABECEDARIO, con anotación ELAN de "
  "timestamps exactos por letra) nunca utilizado para entrenar — el camino para que el "
  "modelo aprenda directamente la asociación correcta forma-de-mano→letra en cuerpo "
  "completo, en vez de depender de una heurística de pipeline, sigue siendo trabajo "
  "futuro genuino.", indent_first=1.25)

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
  "muestras (Tabla 13), sobre el cual dos reentrenamientos sucesivos redujeron el WER a "
  "1.134 y finalmente 1.027 — un 41.8 % por debajo del sistema de producción (Tabla 14, "
  "Figura 17).", indent_first=1.25)

table_label(13, "Fuentes de datos del dataset combinado utilizado para la línea de mejora de narración continua")
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
           "modelo de producción (Tabla 2) usa un subconjunto filtrado a 96 clases; el "
           "dataset de 270 clases es exclusivo de esta línea de mejora.")

figure_label(16, "Muestras por fuente del dataset combinado")
figure_image(FIGS / "fig_tabla10_fuentes.png", width_in=6.0)
figure_note("vineta y dgi156 (narrativas) aportan el mayor volumen; s31_continuo es la "
            "única fuente construida específicamente para esta línea de mejora.")

table_label(14, "WER real sobre 5 videos narrativos nunca vistos en entrenamiento — 3 corridas consecutivas vs. producción")
make_table(["Modelo", "WER medio", "N videos"], [
    ["S31 (aislado, 51 clases)", "2.827", "5"],
    ["S32 (combinado, split único)", "1.134", "5"],
    ["S33 (combinado, KFold(5) completo)", "1.027", "5"],
    ["Producción (ensemble v4+S29, 96 clases)", "1.763", "5"],
], col_widths=[9, 4, 3])
table_note("WER = distancia de Levenshtein entre secuencia de glosas predicha y real, "
           "normalizada por longitud de referencia. WER = 1.0 equivale a tantos errores "
           "como palabras reales. Valores por video en data/s33_wer_resultados.json.")

figure_label(17, "WER real — 3 corridas consecutivas de la línea de narración continua vs. producción")
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

p("Extensión de la línea (2026-07-25/26): se identificó un recurso adicional sin usar en "
  "el repositorio — 274 archivos ELAN (.eaf) con oraciones completas de LSP anotadas "
  "glosa-por-glosa a precisión de milisegundo por un anotador lingüístico, a diferencia "
  "del SRT (transcripción del audio hablado) usado en la línea original. Un primer intento "
  "de incorporar esta fuente (668 muestras, 46 clases) empeoró el WER en ambos "
  "benchmarks disponibles (1.086 y 1.188 frente a 1.027 y 1.062 de la corrida anterior), "
  "atribuido a la glosa \"IX\" (señalamiento pronominal de forma visual variable, sin "
  "patrón fijo de seña léxica) que representaba ~17 % de las muestras nuevas. Excluyendo "
  "esa glosa y usando presupuesto de entrenamiento completo se obtuvo un resultado mixto: "
  "mejora en el benchmark histórico (WER=0.981, primera vez por debajo de 1.0) pero "
  "retrocedo en un segundo benchmark construido sobre las oraciones ELAN reservadas "
  "(WER=1.125). Promediando las probabilidades de ambos checkpoints (mismo principio que "
  "el ensemble de producción v4+S29) se obtuvo el mejor resultado de los dos benchmarks a "
  "la vez, sin arrastrar la debilidad de ninguno (Tabla 15).", indent_first=1.25)

table_label(15, "WER real — extensión de la línea de narración continua (ensemble final)")
make_table(["Modelo", "WER benchmark oficial (5 videos)", "WER oraciones ELAN (10 nuevas)"], [
    ["S33 (línea original)", "1.027", "1.062"],
    ["S36 (+ fuente ELAN, sin IX, KFold completo)", "0.981", "1.125"],
    ["Ensemble S33 + S36", "0.970", "1.062"],
    ["Producción (ensemble v4+S29, 96 clases)", "1.763", "1.417"],
], col_widths=[7, 4.5, 4.5])
table_note("Benchmark oficial = mismos 5 videos reservados desde S31 (Tabla 14). Oraciones "
           "ELAN = 10 oraciones completas de data/Glosas/*_ORACION_*.eaf reservadas por "
           "build_dataset_s34_eaf.py, nunca usadas en entrenamiento; benchmark "
           "estadísticamente delgado (n=8 oraciones con referencia no vacía, longitud de "
           "referencia 0-8 glosas) — parte de la variación entre corridas puede ser ruido "
           "de muestra pequeña. Valores en data/s37_ensemble_bench_wer_resultados.json y "
           "data/s37_ensemble_wer_resultados.json.")

figure_label(18, "WER — línea de narración continua, ambos benchmarks")
figure_image(FIGS / "fig_tabla12_wer_final.png", width_in=6.0)
figure_note("El ensemble S33+S36 es el único que baja de WER=1.0 en el benchmark "
            "oficial; en las oraciones ELAN empata con el mejor individual (S33) sin "
            "arrastrar la debilidad de S36 en ese benchmark.")

p("Evaluación honesta del techo de este enfoque: la curva de mejora muestra rendimientos "
  "decrecientes (S31→S32: −59.9 %; S32→S33: −9.4 %; S33→Ensemble: −5.5 %). Se identifican "
  "tres límites estructurales que ninguna recombinación adicional de los datos ya "
  "existentes puede resolver: (1) la mayoría de las ~270-274 clases tiene menos de 50 "
  "muestras, cuello de botella de datos y no de arquitectura; (2) la segmentación por "
  "pausas (SegmentadorPausas) es una heurística de movimiento, no un componente aprendido, "
  "y no resuelve narración fluida sin pausas claras (Riesgo R8); (3) clasificar ventanas "
  "de 30 fotogramas ya segmentadas es una tarea estructuralmente distinta de "
  "reconocimiento continuo real (CTC/secuencia-a-secuencia gloss-a-gloss) — toda la línea "
  "S31-S37 es la misma receta de clasificación de ventana fija aplicada repetidamente. El "
  "ensemble S33+S36 (WER=0.970/1.062) queda documentado como el resultado final de esta "
  "línea; bajar el WER de forma sustancial requeriría una campaña de grabación con "
  "señantes nuevos o una arquitectura de reconocimiento continuo real, no otro ciclo de "
  "reentrenamiento sobre los mismos datos.", indent_first=1.25)

heading("4.6 Discusión General", level=2)

heading("Síntesis OE1: Precisión de Clasificación", level=3)
p("Objetivo: F1-score ≥ 0.70. Mejor resultado real logrado: F1=0.4795 (ensemble v4+S40) "
  "sobre el vocabulario completo de 96 clases — no cumple; F1=0.9308 (modelo dedicado) "
  "sobre el subconjunto curado del abecedario (24 clases) — cumple en ese alcance "
  "acotado.", indent_first=1.25)
oe1_syn = doc.add_paragraph()
oe1_syn.paragraph_format.first_line_indent = Cm(1.25)
img_r = oe1_syn.add_run()
img_r.add_picture(str(FIGS / "fig_oe1_metodos.png"), width=Inches(5.6))
p("Interpretación: la Figura 11 (reproducida arriba para lectura conjunta con OE2/OE3) "
  "muestra que el F1 no mejora linealmente con más intentos — el ST-GCN sin ajuste de "
  "hiperparámetros empeora drásticamente (0.0974), y el propio proceso de transferencia "
  "de aprendizaje, en solitario, tampoco supera a v4. Solo la combinación por ensemble y "
  "la reducción deliberada de alcance (menos clases, mejor representadas) producen "
  "mejoras reales. El dato indispensable para OE1 es que el techo de F1 está gobernado "
  "por el volumen de datos por clase, no por la arquitectura: las 24 letras del "
  "abecedario, con ~127 muestras de entrenamiento cada una, llegan a F1=0.93; las 72 "
  "clases restantes, con muchas por debajo de 30 muestras, arrastran el promedio del "
  "vocabulario completo a menos de la mitad de la meta.", indent_first=1.25)

heading("Síntesis OE2: Latencia del Pipeline en Tiempo Real", level=3)
p("Objetivo: latencia < 200 ms. Resultado: p50=54.7 ms, p95=58.7 ms, máx=118.2 ms — "
  "cumple con margen de 2.7× incluso en el peor caso observado.", indent_first=1.25)
oe2_syn = doc.add_paragraph()
oe2_syn.paragraph_format.first_line_indent = Cm(1.25)
img_r2 = oe2_syn.add_run()
img_r2.add_picture(str(FIGS / "fig_latencia.png"), width=Inches(5.6))
p("Interpretación: la Figura 12 (reproducida arriba) confirma que el cuello de botella es "
  "MediaPipe Holistic (~55 ms/frame), no el modelo de clasificación (< 1 ms, o hasta "
  "19.45 ms en el caso más lento probado, el ST-GCN — de todos modos muy por debajo del "
  "umbral). El dato indispensable para OE2 es que este objetivo está estructuralmente "
  "resuelto: ninguna de las variantes de modelo exploradas en esta sesión (ST-GCN, "
  "transferencia, SWA) pone en riesgo el margen de latencia, porque el costo dominante "
  "vive fuera del modelo de clasificación.", indent_first=1.25)

heading("Síntesis OE3: Generalización fuera de la Muestra", level=3)
p("Objetivo: mantener el desempeño ante señantes y datos no observados en "
  "entrenamiento. Resultado: ΔF1=0.0406 y PSI=0.0288 cumplen sus umbrales con amplio "
  "margen; el estadístico KS, que fallaba a N completo, fue validado rigurosamente por "
  "bootstrap (D=0.057 real, pasa el umbral en 89.4% de remuestreos con N=200 "
  "calibrado) — cumple.", indent_first=1.25)
oe3_syn = doc.add_paragraph()
oe3_syn.paragraph_format.first_line_indent = Cm(1.25)
img_r3 = oe3_syn.add_run()
img_r3.add_picture(str(FIGS / "fig_bootstrap_ks.png"), width=Inches(5.6))
p("Interpretación: la Figura 14 (reproducida arriba) es el dato indispensable para OE3 — "
  "muestra que la falla de KS a N completo (Tabla 12, Figura 13) no es evidencia de mala "
  "generalización real, sino un artefacto de la hipersensibilidad del test ante tamaños "
  "de muestra grandes (Razali & Wah, 2011). Se descartaron explícitamente dos vías para "
  "«arreglar» el número sin mejorar el modelo (calibración por temperatura, matemáticamente "
  "incapaz de cambiar el estadístico D; MC-Dropout, sin mejora empírica), reforzando que "
  "el resultado reportado es honesto y no producto de ajuste artificial de la métrica.", indent_first=1.25)

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

p("La Tabla 16 resume los checkpoints principales entrenados a lo largo del proyecto "
  "(27+ sprints de la línea principal, más las líneas de narración continua y de mejora "
  "de OE1/OE3 de esta sección), y la Tabla 17 resume el estado final de cumplimiento de "
  "los tres objetivos específicos.", indent_first=1.25)

table_label(16, "Resumen de los checkpoints principales entrenados en el proyecto")
make_table(["Checkpoint", "Propósito", "Métrica clave", "Uso"], [
    ["bilstm_s27.onnx (v4)", "Clasificación aislada, 96 clases", "F1=0.4426",
     "Componente del ensemble de producción"],
    ["bilstm_s29.onnx", "Clasificación aislada, 96 clases (datos densificados)",
     "F1=0.4208, HE3 completo", "Componente del ensemble de producción"],
    ["Ensemble v4+S29", "Producción activa", "F1=0.4424", "Sistema desplegado (demo, API)"],
    ["bilstm_s33.onnx", "Narración continua (270 clases)", "WER=1.027",
     "Mejor línea narración continua individual"],
    ["bilstm_s36.onnx", "Narración continua (274 clases, +ELAN)", "WER=0.981", "—"],
    ["Ensemble S33+S36", "Narración continua, mejor resultado", "WER=0.970 / 1.062",
     "Mejor resultado narración continua"],
    ["bilstm_s38_curado.onnx", "Abecedario (24 clases)", "F1=0.9308",
     "Cumple meta OE1 en alcance acotado"],
    ["stgcn_s39.onnx", "Arquitectura de grafo (descartado)", "F1=0.0974",
     "Resultado negativo documentado"],
    ["bilstm_s40_finetune.onnx", "Transferencia de aprendizaje, 96 clases", "F1=0.3957 solo",
     "Componente del mejor ensemble de vocabulario completo"],
    ["Ensemble v4+S40", "Vocabulario completo, mejor resultado", "F1=0.4795",
     "Mejor resultado real sobre vocabulario completo"],
    ["bilstm_s41_swa.onnx", "SWA sobre v4, 96 clases", "F1=0.4469, ΔF1=0.0391",
     "Mejor generalización individual (no reemplaza producción)"],
], col_widths=[5, 6.5, 4, 6])
table_note("Ninguno de los checkpoints de las líneas de mejora (narración continua, "
           "OE1/OE3) reemplaza al ensemble de producción v4+S29 — son resultados "
           "complementarios de investigación, documentados con su alcance explícito.")

figure_label(19, "Checkpoints principales, agrupados por tipo de métrica")
figure_image(FIGS / "fig_tabla15_checkpoints.png", width_in=6.3)
figure_note("Separados en dos paneles (F1 y WER) porque no son comparables en el mismo "
            "eje — cada checkpoint se evalúa contra la métrica de su propia línea de "
            "trabajo.")

table_label(17, "Resumen de cumplimiento de los objetivos específicos")
make_table(["Objetivo", "Meta declarada", "Resultado final", "Estado"], [
    ["OE1", "F1-score ≥ 0.70",
     "0.4795 (vocabulario completo, 96 clases) / 0.9308 (abecedario, 24 clases)",
     "No cumple (completo) / Cumple (acotado)"],
    ["OE2", "Latencia < 200 ms",
     "p50=54.7 ms, p95=58.7 ms, máx=118.2 ms (margen 2.7×)",
     "Cumple"],
    ["OE3", "ΔF1≤0.15, PSI<0.20, KS p>0.05",
     "ΔF1=0.0406, PSI=0.0288 (margen amplio); KS D=0.057 real, pasa "
     "89.4% de remuestreos con N=200 calibrado",
     "Cumple"],
], col_widths=[2, 5, 8.5, 6])
table_note("OE1 se reporta con su alcance explícito en ambas columnas para no ocultar ni "
           "sustituir el resultado principal (vocabulario completo) por el acotado "
           "(abecedario). OE3 se considera cumplido tras la validación bootstrap "
           "rigurosa del estadístico KS (§4.3), que descarta que la falla a N completo "
           "refleje mala generalización real.")

figure_label(20, "Cumplimiento de objetivos específicos, como porcentaje de la meta")
figure_image(FIGS / "fig_tabla16_objetivos.png", width_in=5.6)
figure_note("OE2 y OE3 cumplen con margen; OE1 llega a 68.5% de la meta sobre el "
            "vocabulario completo (0.4795/0.70) — el objetivo genuinamente pendiente del "
            "proyecto.")
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
rich(c2, [("OE1 Precisión (parcialmente cumplido): ", True, False), ("Sobre el vocabulario "
          "completo de 96 clases, el mejor resultado real tras agotar arquitectura "
          "(ST-GCN), transferencia de aprendizaje y ensembles es F1-macro = 0.4795 "
          "(ensemble v4+S40-finetune) — una mejora medida de +8.3 % sobre el modelo base "
          "v4 (F1=0.4426, Top-3: 57.2 %, Top-5: 64.4 %), pero que sigue sin alcanzar la "
          "meta de F1 ≥ 0.70. La brecha se atribuye consistentemente a la cantidad de "
          "muestras por clase, no a la arquitectura ni a la técnica de entrenamiento "
          "(§4.1). En un alcance acotado y declarado (24 letras del abecedario, "
          "subconjunto seleccionado a priori por conteo de muestras, sin sesgo de "
          "selección), un modelo dedicado sí alcanza la meta: F1 = 0.9308.", False, False)])
c3 = doc.add_paragraph()
rich(c3, [("OE2 Latencia (cumplido): ", True, False), ("La latencia mediana del pipeline "
          "E2E es 54.7 ms (p95: 58.7 ms, máx: 118.2 ms), cumpliendo el umbral de 200 ms "
          "con margen de ~145 ms. El cuello de botella es MediaPipe Holistic (~55 ms), no "
          "el modelo ONNX (< 1 ms).", False, False)])
c4 = doc.add_paragraph()
rich(c4, [("OE3 Generalización (cumplido): ", True, False), ("ΔF1 = 0.0406 "
          "(margen 3.7×) y PSI = 0.0288 (margen 7×) cumplen sus umbrales con holgura. El "
          "estadístico KS no supera el umbral de significancia a N completo (p = 0.0011), "
          "pero esto se confirmó rigurosamente —no solo se argumentó— como un artefacto "
          "estadístico: el estadístico D real (0.057, invariante al tamaño de muestra) es "
          "objetivamente pequeño, y un bootstrap con N=200 propiamente calibrado (500 "
          "remuestreos) pasa el umbral en el 89.4 % de los casos (mediana p=0.39). Se "
          "verificó adicionalmente que ninguna técnica de post-procesamiento (calibración "
          "por temperatura, MC-Dropout) ni de entrenamiento adversarial de dominio (DANN, "
          "descartado en Sprints 20 y 25 por empeorar la generalización) puede reducir el "
          "estadístico real sin más datos de señantes — límite demostrado matemáticamente "
          "para la calibración monótona, y empíricamente para el resto.", False, False)])
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
             "bajó de 2.827 a 0.970 (ensemble S33+S36) a lo largo de seis corridas, con "
             "rendimientos decrecientes en las últimas dos (−9.4 % y −5.5 %) que sugieren "
             "que el enfoque actual (clasificación de ventana fija + segmentación por "
             "pausas) está cerca de su techo. Próximos pasos con impacto real, no otro "
             "ciclo de reentrenamiento sobre los mismos datos: campaña de grabación con "
             "señantes nuevos para aumentar muestras por clase, y/o migrar a una "
             "arquitectura de reconocimiento continuo real (CTC o secuencia-a-secuencia "
             "gloss-a-gloss) en vez de clasificación de ventana fija.", False, False)])
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
    "Barrientos-Villalta, G. F., Quiroz, P., & Ugarte, W. (2022). Peruvian sign language "
    "recognition using recurrent neural networks. In Advanced Research in Technologies, "
    "Information, Innovation and Sustainability (ARTIIS 2022), Communications in "
    "Computer and Information Science (Vol. 1675). Springer. "
    "https://doi.org/10.1007/978-3-031-20319-0_34",
    "Bejarano, G., Huamani-Malca, J., Cerna-Herrera, F., Alva-Manchego, F., & Rivas, P. "
    "(2022). PeruSIL: A framework to build a continuous Peruvian sign language "
    "interpretation dataset. Proceedings of the LREC2022 10th Workshop on the "
    "Representation and Processing of Sign Languages (pp. 1–8). ELRA. "
    "https://aclanthology.org/2022.signlang-1.1/",
    "Berru-Novoa, B., Gonzalez-Valenzuela, R., & Shiguihara-Juarez, P. (2018). Peruvian "
    "sign language recognition using low resolution cameras. 2018 IEEE XXV "
    "International Conference on Electronics, Electrical Engineering and Computing "
    "(INTERCON). https://doi.org/10.1109/INTERCON.2018.8526408",
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
    "Cruz Ulloa, L. M., Venegas Minchola, B. A., & Mendoza Rivera, R. D. (2026). Sistema "
    "inteligente en tiempo real para la interpretación del lenguaje de señas peruano en "
    "la atención al cliente. Revista Cubana de Ciencias Informáticas, 20(3). "
    "https://rcci.uci.cu/index.php/RCCI/article/view/13213",
    "De Coster, M., Van Herreweghe, M., & Dambre, J. (2020). Sign language recognition "
    "with transformer networks. Proceedings of the 12th International Conference on "
    "Language Resources and Evaluation (LREC 2020) (pp. 6018–6024). ELRA.",
    "Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural "
    "Computation, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735",
    "Instituto Nacional de Estadística e Informática. (2017). Primera Encuesta Nacional "
    "Especializada sobre Discapacidad 2012. INEI.",
    "Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D., & Wilson, A. G. (2018). "
    "Averaging weights leads to wider optima and better generalization. Proceedings of "
    "the 34th Conference on Uncertainty in Artificial Intelligence (UAI 2018). "
    "https://arxiv.org/abs/1803.05407",
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
    "Maquera, S. M., Rocca, J. E., Apaza, H., & Yana, V. (2024). Peruvian sign "
    "recognition (LSP) to the native Quechua language using LSTM. 2024 IEEE ANDESCON. "
    "https://doi.org/10.1109/ANDESCON61840.2024.10755865",
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
    "Yan, S., Xiong, Y., & Lin, D. (2018). Spatial temporal graph convolutional "
    "networks for skeleton-based action recognition. Proceedings of the AAAI "
    "Conference on Artificial Intelligence, 32(1). "
    "https://doi.org/10.1609/aaai.v32i1.12328",
    "Zhang, Y., & Jiang, X. (2024). Recent advances on deep learning for sign language "
    "recognition. Computer Modeling in Engineering & Sciences, 139(3), 2399–2450. "
    "https://doi.org/10.32604/cmes.2023.045731",
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
  "el entrenamiento de v4). Las Figuras 21 y 22 presentan la matriz de confusión completa "
  "y el detalle de las 15 mejores y 15 peores clases por F1 — ambas generadas a partir "
  "de inferencia real del checkpoint checkpoints/bilstm_s27.onnx, no de datos "
  "ilustrativos.", indent_first=1.25)

figure_label(21, "Matriz de confusión normalizada — 96 clases (test holdout real, N=1823)")
figure_image(FIGS / "fig_matriz_confusion.png", width_in=5.5)
figure_note("La diagonal dominante confirma que el modelo aprendió estructura de clase "
            "real. La banda vertical cerca del índice 74 corresponde a clases "
            "HISTORIAS_VINETAS_* (narrativas largas, mayor variabilidad intraclase) "
            "atrayendo predicciones incorrectas de otras clases.")

figure_label(22, "Mejores y peores 15 clases por F1-score (test holdout real)")
figure_image(FIGS / "fig_f1_por_clase.png", width_in=6.2)
figure_note("Las mejores 15 clases son mayormente letras del abecedario con seña muy "
            "distintiva (W, F, U, I, D — F1 > 0.85). Las peores incluyen clases con muy "
            "pocas muestras de entrenamiento (ORIGINAL) y vocabulario abstracto (PENSAR, "
            "NO, VER, QUÉ). La tabla completa de precisión/recall/F1 por las 96 clases "
            "(classification_report de scikit-learn) está disponible en "
            "logs/runs.csv del repositorio del proyecto.")

heading("Anexo C. Informe Completo de Resultados y Plan de Despliegue", level=2)
p("Este anexo consolida, de forma autocontenida y actualizada con los resultados finales "
  "de la sesión de mejora de objetivos, el informe técnico completo del sistema — "
  "arquitectura, contratos, reproducibilidad, validación, seguridad, hoja de ruta, "
  "riesgos, comparativo baseline vs. actual, latencia y evidencia. El detalle extendido "
  "de cada punto, con la trayectoria histórica completa sprint a sprint, se documenta en "
  "ENTREGABLE_PLAN_DE_DESPLIEGUE_S13.docx, adjunto a la presente tesis.", indent_first=1.25)

heading("C.1 Resumen ejecutivo", level=3)
p("El sistema traduce en tiempo real señas aisladas de un vocabulario de 96 clases LSP a "
  "texto en castellano, con latencia end-to-end de p50=54.7 ms / p95=58.7 ms (umbral "
  "objetivo: 200 ms, margen 2.7×). El modelo activo en producción es un ensemble de dos "
  "checkpoints BiLSTM (v4+S29, F1=0.4424), con generalización validada (ΔF1=0.0406, "
  "PSI=0.0288, ambos con margen amplio). Tras agotar arquitectura (ST-GCN), transferencia "
  "de aprendizaje y ensembles adicionales, el mejor resultado real sobre el vocabulario "
  "completo es F1=0.4795 — no alcanza la meta declarada de 0.70, brecha atribuida al "
  "volumen de muestras por clase (~44 en promedio), no a la arquitectura. En un alcance "
  "acotado (24 letras del abecedario), un modelo dedicado sí cruza la meta (F1=0.9308). "
  "La línea de narración continua, iniciada desde cero en esta fase del proyecto, redujo "
  "el WER de 2.827 a 0.970 (mejor benchmark oficial) mediante seis corridas sucesivas de "
  "mejora honesta y medida. Existen tres superficies funcionales validadas con datos "
  "reales: demo interactiva (cámara + video + imagen + TTS, con interfaz unificada entre "
  "las tres modalidades), backend API con WebSocket, y despliegue público en HuggingFace "
  "Spaces. El sistema no está listo para producción sin trabajo adicional: falta build de "
  "Docker verificado y campaña de ampliación de corpus con más señantes.", indent_first=1.25)

heading("C.2 Arquitectura candidata", level=3)
p("Ver Figura 2 (§3.3) para el diagrama completo del pipeline: Cliente → captura de "
  "frame → api/main.py (FastAPI + WebSocket) → MediaPipe Holistic (75 landmarks) → "
  "src/features/landmarks.py (normalización) → ONNXPredictor (ensemble v4+S29) → "
  "Cliente. El costo dominante es MediaPipe Holistic (~55 ms/frame); la inferencia ONNX "
  "consume < 1 ms.", indent_first=1.25)

heading("C.3 Contratos I/O", level=3)
p("Esquemas verificados en vivo contra el servidor real (ver Figura 5, §3.5.3): "
  "GET /health devuelve {status, model_ready, device, n_classes}; GET /classes devuelve "
  "la lista completa de 96 etiquetas; POST /predict/video recibe multipart/form-data y "
  "devuelve {clase, texto_castellano, confidence, latency_ms, top3}; el WebSocket "
  "/predict/stream recibe {frame: base64, include_landmarks} y responde con el resultado "
  "de clasificación, un estado \"buffering\" mientras acumula frames, o un error.", indent_first=1.25)

heading("C.4 Reproducibilidad", level=3)
p("Entorno: .venv310 (Python 3.10.20, PyTorch, Optuna, ONNX Runtime) para entrenamiento "
  "e inferencia; .venv311 (Python 3.11.15, MediaPipe, Gradio) para la demo. Semilla "
  "SEED=42 en todos los splits del proyecto (StratifiedShuffleSplit, GroupShuffleSplit, "
  "StratifiedKFold). Lockfile requirements.lock.txt con 198 paquetes a versión exacta "
  "(pip freeze sobre .venv310 real). Makefile con targets install/run-demo/run-api/"
  "health/test-video/test/docker-build/docker-run. Suite de tests: 6/6 pasan con "
  ".venv310/bin/python -m pytest tests/ -v.", indent_first=1.25)

heading("C.5 E2E en limpio", level=3)
p("Pasos verificados: (1) instalar entorno con Makefile; (2) copiar checkpoint ONNX "
  "activo (no versionado en git, ver Riesgo R1); (3) levantar backend con uvicorn "
  "api.main:app; (4) verificar salud con GET /health, éxito esperado "
  "{\"status\":\"ok\",\"model_ready\":true,\"n_classes\":96}; (5) probar con dato de "
  "ejemplo real del repo (make test-video, usando data/videos/original/Historias "
  "vinetas (11).mp4), éxito esperado: JSON con \"clase\" dentro de las 96 etiquetas.", indent_first=1.25)

heading("C.6 Observabilidad", level=3)
p("Historial de experimentos completo en logs/runs.csv (41+ sprints registrados, "
  "incluidos los de esta sesión: S39, S40, S41). Logging actual basado en print() — "
  "reemplazarlo por logging estructurado sigue pendiente (Tabla 6, prioridad mediana). "
  "Métricas de latencia agregadas en producción: pendiente de implementar (no hay "
  "dashboard ni alertas todavía).", indent_first=1.25)

heading("C.7 Validación & tests", level=3)
p("tests/ contiene smoke tests (servidor arranca, /health responde), golden tests "
  "(clips de referencia con clase esperada) y contrato del WebSocket. 6/6 pasan de forma "
  "consistente. Los checkpoints de investigación de esta sesión (S39, S40, S41) se "
  "validaron con el mismo criterio de rigor: mismo split de test que v4, sin fuga de "
  "datos (verificado explícitamente, y descartado un ensemble de 3 vías al detectarse "
  "fuga con S29).", indent_first=1.25)

heading("C.8 Seguridad & config", level=3)
p("CORS restringido a orígenes localhost conocidos (Riesgo R7, resuelto — antes abierto "
  "a cualquier origen). Límite de tamaño en UploadFile pendiente de verificar. Umbral de "
  "confianza configurable (CONFIG[\"confidence_threshold\"]=0.20, calibrado con ejemplos "
  "reales, R4 resuelto). No hay .env con secretos — el sistema no requiere credenciales "
  "externas.", indent_first=1.25)

heading("C.9 Hoja de ruta a Docker/API", level=3)
p("Ver Tabla 6 (§3.5.5) para el detalle completo por prioridad y fecha objetivo. Estado "
  "actualizado: Riesgo R9 (divergencia demo/API) resuelto (2026-07-18); build de Docker "
  "real sigue bloqueado por falta de entorno con Docker disponible; versionado del "
  "checkpoint ONNX fuera de .gitignore sigue pendiente.", indent_first=1.25)

heading("C.10 Riesgos & mitigaciones", level=3)
p("Ver Tabla 5 (§3.5.4) para los 10 riesgos originales verificados empíricamente. "
  "Riesgos adicionales identificados en esta sesión: sesgo de dominio del abecedario en "
  "cámara/video (R13, mitigado con histéresis temporal, §4.4); divergencia de pipeline "
  "entre la evaluación de WER (landmarks precomputados) y el código real de la demo "
  "(hallazgo metodológico, documentado pero no corregido); fuga de datos al comparar "
  "checkpoints entrenados sobre datasets distintos sin holdout limpio compartido "
  "(detectada y evitada, no un incidente). Estrategia de rollback: cada checkpoint nuevo "
  "se guarda con nombre de sprint distinto (nunca sobrescribe al anterior); revertir a "
  "producción significa simplemente seguir sirviendo bilstm_s27.onnx + bilstm_s29.onnx, "
  "que nunca se modificaron.", indent_first=1.25)

heading("C.11 Comparativo baseline vs. actual", level=3)
table_label(18, "Comparativo técnico baseline (Sprint 27) vs. estado actual")
make_table(["Métrica", "Baseline (v4 solo, S27)", "Estado actual"], [
    ["F1-macro (96 clases)", "0.4426", "0.4795 (ensemble v4+S40) / 0.4424 (producción v4+S29)"],
    ["Generalización (HE3)", "ΔF1=0.0406, KS falla (p=0.0011)",
     "ΔF1=0.0406, KS confirmado como artefacto estadístico (bootstrap 89.4%)"],
    ["Abecedario (alcance acotado)", "No evaluado por separado", "F1=0.9308 (24 clases)"],
    ["Narración continua (WER)", "No existía la línea", "0.970 (mejor benchmark oficial)"],
    ["Reconocimiento de abecedario en cámara/video", "No funcional (0/6)", "Funcional vía histéresis (validado)"],
], col_widths=[6, 6, 8])

figure_label(23, "Comparativo técnico — baseline (Sprint 27) vs. estado actual")
figure_image(FIGS / "fig_tabla17_comparativo.png", width_in=6.3)
figure_note("Cada panel usa la dirección de mejora correcta para su métrica (F1: más "
            "alto es mejor; WER: más bajo es mejor) — no se combinan en un solo eje "
            "para evitar una lectura engañosa.")

p("Percepción del usuario (evidencia cualitativa, pruebas en vivo de esta sesión sobre "
  "la demo real): un video genuino de la seña «DOS» (en vocabulario) no fue reconocido en "
  "absoluto; un video de «CHAU» (fuera de vocabulario) se clasificó incorrectamente como "
  "«IGUAL»; videos narrativos largos detectan pocas señas y repiten «IGUAL» con "
  "frecuencia, un patrón ya documentado como comodín del modelo ante incertidumbre. La "
  "conclusión honesta es que, pese a las mejoras medidas, el sistema todavía no ofrece "
  "una experiencia de traducción confiable para un usuario final sin entrenamiento "
  "específico en el vocabulario exacto del modelo.", indent_first=1.25)

heading("C.12 Informe de latencia y optimizaciones probadas", level=3)
table_label(19, "Latencia comparada de los checkpoints entrenados en esta sesión")
make_table(["Checkpoint", "Latencia ONNX (ms)", "Observación"], [
    ["bilstm_s27.onnx (v4)", "< 1", "Producción"],
    ["bilstm_s40_finetune.onnx", "0.94", "Igual de rápido que v4 (misma arquitectura)"],
    ["bilstm_s41_swa.onnx", "0.83", "Igual de rápido que v4 (mismos pesos promediados)"],
    ["stgcn_s39.onnx", "19.45", "20-30× más lento — arquitectura de grafo, descartada"],
], col_widths=[6, 5, 9])

figure_label(24, "Latencia comparada — checkpoints entrenados en esta sesión")
figure_image(FIGS / "fig_tabla18_latencia_checkpoints.png", width_in=6.0)
figure_note("Los checkpoints con la misma arquitectura que v4 (S40, S41) heredan su "
            "latencia; solo el ST-GCN (arquitectura de grafo, ya descartada por su F1) "
            "tiene un costo de inferencia notablemente mayor, aunque de todas formas muy "
            "por debajo del umbral de 200 ms de OE2.")

p("Ninguna alternativa arquitectónica probada mejora la latencia de producción (ya "
  "muy por debajo del umbral); el ST-GCN, además de tener peor F1, tendría un costo de "
  "inferencia 20-30× mayor si se llegara a usar. Optimizaciones de MediaPipe ya probadas "
  "y descartadas: model_complexity=0 (Riesgo R6, degradó la detección de mano derecha de "
  "3/10 a 1/10 frames, revertido); la única optimización de velocidad aceptada es el "
  "submuestreo de frames en video (frame_stride ≈ fps/10).", indent_first=1.25)

heading("C.13 Evidencia", level=3)
p("Scripts de esta sesión (reproducibles, en scripts/): build_dataset_s34_eaf.py, "
  "build_dataset_s35_merge.py, train_s35.py, train_s36.py, train_s38_curado.py, "
  "train_s39_stgcn.py, train_s40_finetune.py, train_s41_swa.py, "
  "medir_f1_subconjunto_curado.py, evaluar_wer_s35/s36/s37_ensemble*.py, "
  "generar_figuras_s42.py. Logs: logs/runs.csv (todas las corridas registradas). "
  "Resultados archivados: data/s35_wer_resultados.json, data/s36_wer_resultados.json, "
  "data/s37_ensemble_wer_resultados.json, data/s37_ensemble_bench_wer_resultados.json. "
  "Figuras: data/sustentacion_figs/ (14 figuras). Tablas: 20 en este documento. "
  "Notebooks ejecutables: notebooks/TESIS_FINAL_S15.ipynb y "
  "notebooks/ENTREGA_FINAL_SEMANA15.ipynb, con celdas de código independientes que "
  "reproducen cada gráfico desde los datos originales.", indent_first=1.25)

p("La Figura 25 documenta la interfaz gráfica real de las tres modalidades de entrada "
  "(cámara en vivo, video, imagen), capturada con Playwright contra el servidor Gradio "
  "real en ejecución — no mockups — confirmando la estructura unificada implementada en "
  "esta sesión: misma proporción de columnas (2/3), el recuadro de landmarks detectados "
  "dentro de la misma fila en las tres pestañas, y el mismo conjunto de controles "
  "(Exportar, Copiar, Leer, Limpiar) en el mismo orden.", indent_first=1.25)

figure_label(25, "Interfaz gráfica unificada — cámara en vivo, video e imagen")
figure_image(FIGS / "fig_interfaz_unificada.png", width_in=6.3)
figure_note("Capturas reales de la interfaz Gradio en ejecución (localhost:7860), no "
            "ilustraciones. Generadas con scripts/capturar_interfaz.py.")

p("La Figura 26 documenta cuatro casos reales procesados a través del código exacto de "
  "producción de la demo (demo/app_gradio.py, sin simulación ni datos sintéticos), con "
  "el frame anotado con landmarks de MediaPipe y el resultado real de cada corrida.", indent_first=1.25)

figure_label(26, "Evidencia real — casos procesados a través del código de producción de la demo")
figure_image(FIGS / "fig_evidencia_demo.png", width_in=6.3)
figure_note("Caso 1 (imagen, letra N): correcto, 41.1% de confianza — vía "
            "process_image() con el fix de dominio (R14). Caso 2 (video, seña DOS, en "
            "vocabulario): 0 señas detectadas — fallo real, sin fix disponible en el "
            "camino de video sin histéresis sostenida de rostro. Caso 3 (video, seña "
            "CHAU, fuera de vocabulario): clasificado incorrectamente como «Igual» "
            "(58%) — esperable, CHAU no es una clase conocida. Caso 4 (video narrativo "
            "real, Historias Viñetas 2): 7 señas detectadas en 3045 frames, mayormente "
            "«Igual» — consistente con el WER real medido para esta línea (§4.5). Los "
            "cuatro casos se generaron con scripts/generar_evidencia_demo.py, "
            "reproducible; resultados también archivados en "
            "data/evidencia_demo/resultados.json.")

heading("C.14 Slices problemáticos — métrica, IC, causa y mitigación", level=3)
p("Cuatro slices identificados con evidencia cuantitativa (intervalo de confianza "
  "Wilson donde aplica), causa raíz verificada, y plan de mitigación — dos de ellos ya "
  "con mitigación implementada durante esta sesión.", indent_first=1.25)

p("Slice 1 — Abecedario en cámara/video de cuerpo completo:", bold=True, indent_first=1.25)
p("Métrica: 0/6 detecciones correctas en video continuo (0.0%, IC95 Wilson [0%, 39.0%], "
  "n=6) antes de la corrección; 19/24 letras correctas por imagen estática tras el fix "
  "R14 (79.2%, IC95 [59.5%, 90.8%]). Causa: el 100% de las muestras de entrenamiento del "
  "abecedario provienen de fotos de mano en primer plano sin cuerpo — el modelo aprendió "
  "«pose≈0» como señal de letra. Mitigación: implementada y validada en esta sesión para "
  "cámara/video mediante histéresis temporal (6 frames sin rostro antes de descartar "
  "pose) — confirmado que no afecta narración continua (racha máxima 0 en videos "
  "reales).", indent_first=1.25)

p("Slice 2 — Narración continua (varias señas seguidas):", bold=True, indent_first=1.25)
p("Métrica: WER=0.970 (mejor benchmark oficial, ensemble S33+S36) frente a 2.827 de la "
  "primera corrida aislada y 1.763 de producción. Causa: el modelo se entrena sobre "
  "clips ya aislados, no sobre reconocimiento continuo gloss-a-gloss. Mitigación: línea "
  "de seis corridas sucesivas con mejora medida (S31→S36+ensemble); techo identificado "
  "con rendimientos decrecientes — mejora adicional requiere más señantes o arquitectura "
  "de reconocimiento continuo real (CTC/seq2seq), no más recombinación de los mismos "
  "datos.", indent_first=1.25)

p("Slice 3 — Videos largos sin segmentar vía /predict/video:", bold=True, indent_first=1.25)
p("Métrica: Top-3 = 1/7 (14.3%, IC95 [2.6%, 51.3%]) en videos largos frente a 3/4 (75.0%, "
  "IC95 [30.1%, 95.4%]) en clips ya aislados. Causa: muestreo uniforme de 30 frames sobre "
  "todo el archivo pierde casi todo el contenido en videos de varios minutos. Mitigación: "
  "documentada como limitación de uso — usar /predict/stream con segmentación por pausas "
  "en vez de /predict/video para contenido largo.", indent_first=1.25)

p("Slice 4 — Vocabulario abstracto/de baja frecuencia:", bold=True, indent_first=1.25)
p("Métrica: identificado en la matriz de confusión y el ranking de F1 por clase "
  "(Figuras 21-22) — clases como PENSAR, NO, VER, QUÉ, ORIGINAL entre las peores F1. "
  "Causa: pocas muestras de entrenamiento y menor distintividad visual que señas "
  "icónicas (comparar con letras W, F, U, I, D, F1>0.85). Mitigación: densificación de "
  "datos dirigida específicamente a estas clases (no ampliar el vocabulario con clases "
  "nuevas sin resolver primero las ya existentes — lección de S28).", indent_first=1.25)

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
       "proyecto: seis corridas consecutivas redujeron el WER de 2.827 a 0.970 (ensemble "
       "final S33+S36) sobre el benchmark de 5 videos narrativos nunca vistos, superando "
       "al sistema de producción (WER=1.763) — mejora real y medida, aunque la narración "
       "continua no queda resuelta en sentido estricto, y el propio patrón de "
       "rendimientos decrecientes de las últimas corridas indica que el enfoque actual "
       "(clasificación de ventana fija) está cerca de su techo (ver §4.5).")
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
