"""
EDA local adaptado al dataset real: 'Historias vinetas' (videos continuos LSP)
Ejecutar: python3 scripts/run_eda_local.py
"""
import cv2, os, json, glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────
VIDEO_DIR = Path("data/videos/original")
DATA_DIR  = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Escaneo recursivo de MP4 ───────────────────────────────────────────
print("═" * 60)
print("1. ESCANEO DE VIDEOS MP4")
print("═" * 60)

videos = sorted(VIDEO_DIR.glob("*.mp4")) + sorted(VIDEO_DIR.glob("*.MP4"))
if not videos:
    print(f"ERROR: No se encontraron videos en {VIDEO_DIR}")
    exit(1)
print(f"  Videos encontrados: {len(videos)}")

# ── 2. Extracción de metadatos ────────────────────────────────────────────
print("\n2. EXTRACCIÓN DE METADATOS")
records = []
for vp in videos:
    cap = cv2.VideoCapture(str(vp))
    fps      = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    dur_seg = n_frames / fps if fps > 0 else 0
    valido  = n_frames > 0 and fps > 0

    # Extraer número de la vineta del nombre
    nombre = vp.stem  # "Historias vinetas (11)"
    match  = re.search(r'\((\d+)\)', nombre)
    num    = int(match.group(1)) if match else -1

    records.append({
        'ruta':       str(vp),
        'nombre':     vp.name,
        'num_vineta': num,
        'clase':      f'vineta_{num:03d}',
        'n_frames':   n_frames,
        'fps':        fps,
        'ancho':      width,
        'alto':       height,
        'resolucion': f'{width}x{height}',
        'duracion_seg': round(dur_seg, 2),
        'size_mb':    round(vp.stat().st_size / 1e6, 2),
        'valido':     valido,
    })
    print(f"  {vp.name:40s} | {fps:.0f}fps | {n_frames:5d}f | "
          f"{width}x{height} | {dur_seg:.1f}s | {vp.stat().st_size/1e6:.1f}MB")

df = pd.DataFrame(records).sort_values('num_vineta').reset_index(drop=True)
df_valid = df[df['valido']]
print(f"\n  Total: {len(df)} | Válidos: {len(df_valid)} | "
      f"Inválidos: {len(df)-len(df_valid)}")

# ── 3. Estadísticas generales ─────────────────────────────────────────────
print("\n3. ESTADÍSTICAS GENERALES")
print(f"  Duración total:     {df_valid['duracion_seg'].sum()/60:.1f} min")
print(f"  Duración media:     {df_valid['duracion_seg'].mean():.1f}s "
      f"± {df_valid['duracion_seg'].std():.1f}s")
print(f"  Duración min/max:   {df_valid['duracion_seg'].min():.1f}s / "
      f"{df_valid['duracion_seg'].max():.1f}s")
print(f"  FPS únicos:         {sorted(df_valid['fps'].unique())}")
print(f"  Resoluciones:       {df_valid['resolucion'].value_counts().to_dict()}")
print(f"  Tamaño total:       {df_valid['size_mb'].sum():.1f} MB")
print(f"  Frames totales:     {df_valid['n_frames'].sum():,}")
print(f"  Frames @30fps/30f ≈ {df_valid['n_frames'].sum()//30:,} segmentos posibles")

# Estimación de señas potenciales (asumiendo ~2s por seña)
segs_2s = int(df_valid['duracion_seg'].sum() / 2)
print(f"  Señas estimadas (~2s/seña): {segs_2s}")

# ── 4. Guardado manifest ──────────────────────────────────────────────────
manifest_path = DATA_DIR / "manifest.csv"
df.to_csv(manifest_path, index=False)
print(f"\n4. Manifest guardado: {manifest_path}")

# label2idx basado en número de vineta
labels = sorted(df_valid['clase'].unique())
label2idx = {lbl: i for i, lbl in enumerate(labels)}
idx2label = {i: lbl for lbl, i in label2idx.items()}

with open(DATA_DIR / "label2idx.json", "w") as f:
    json.dump(label2idx, f, indent=2, ensure_ascii=False)
with open(DATA_DIR / "idx2label.json", "w") as f:
    json.dump(idx2label, f, indent=2, ensure_ascii=False)
print(f"   label2idx.json: {len(label2idx)} clases guardadas")

# class_weights uniformes (1 video/clase → dataset muy pequeño)
class_weights = {lbl: 1.0 for lbl in labels}
with open(DATA_DIR / "class_weights.json", "w") as f:
    json.dump(class_weights, f, indent=2)

# ── 5. Visualizaciones ────────────────────────────────────────────────────
print("\n5. GENERANDO VISUALIZACIONES...")
fig = plt.figure(figsize=(18, 12))
fig.suptitle("EDA Dataset LSP — Historias Vinetas", fontsize=16, fontweight='bold', y=0.98)

# 5.1 Duración por vineta
ax1 = fig.add_subplot(2, 3, 1)
bars = ax1.barh(df_valid['nombre'].str.replace('.mp4','',regex=False),
                df_valid['duracion_seg'], color='steelblue', edgecolor='navy', alpha=0.8)
ax1.set_xlabel("Duración (segundos)")
ax1.set_title("Duración por Video", fontweight='bold')
ax1.axvline(df_valid['duracion_seg'].mean(), color='red', ls='--', lw=1.5,
            label=f"Media: {df_valid['duracion_seg'].mean():.0f}s")
ax1.legend(fontsize=8)
for bar, val in zip(bars, df_valid['duracion_seg']):
    ax1.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.0f}s',
             va='center', fontsize=7)
ax1.tick_params(axis='y', labelsize=7)

# 5.2 Histograma de duración
ax2 = fig.add_subplot(2, 3, 2)
ax2.hist(df_valid['duracion_seg'], bins=10, color='coral', edgecolor='darkred', alpha=0.85)
ax2.set_xlabel("Duración (s)")
ax2.set_ylabel("Frecuencia")
ax2.set_title("Distribución de Duraciones", fontweight='bold')
ax2.axvline(df_valid['duracion_seg'].mean(), color='blue', ls='--', lw=2,
            label=f"μ={df_valid['duracion_seg'].mean():.0f}s")
ax2.axvline(df_valid['duracion_seg'].median(), color='green', ls=':', lw=2,
            label=f"med={df_valid['duracion_seg'].median():.0f}s")
ax2.legend(fontsize=9)

# 5.3 Resoluciones (pie)
ax3 = fig.add_subplot(2, 3, 3)
res_counts = df_valid['resolucion'].value_counts()
colors = plt.cm.Set3(np.linspace(0, 1, len(res_counts)))
ax3.pie(res_counts.values, labels=res_counts.index, autopct='%1.0f%%',
        colors=colors, startangle=90)
ax3.set_title("Distribución de Resoluciones", fontweight='bold')

# 5.4 Frames totales por video
ax4 = fig.add_subplot(2, 3, 4)
ax4.bar(range(len(df_valid)), df_valid['n_frames'], color='mediumseagreen',
        edgecolor='darkgreen', alpha=0.8)
ax4.set_xticks(range(len(df_valid)))
ax4.set_xticklabels([f"V{r}" for r in df_valid['num_vineta']], rotation=45, fontsize=8)
ax4.set_xlabel("Vineta")
ax4.set_ylabel("Número de Frames")
ax4.set_title("Frames Totales por Video", fontweight='bold')
ax4.axhline(900, color='orange', ls='--', lw=1.5, label='30s @30fps')
ax4.axhline(1800, color='red', ls='--', lw=1.5, label='60s @30fps')
ax4.legend(fontsize=8)

# 5.5 Tamaño en MB
ax5 = fig.add_subplot(2, 3, 5)
ax5.scatter(df_valid['duracion_seg'], df_valid['size_mb'],
            c=df_valid['ancho'], cmap='viridis', s=80, alpha=0.8, edgecolors='k')
ax5.set_xlabel("Duración (s)")
ax5.set_ylabel("Tamaño (MB)")
ax5.set_title("Tamaño vs Duración", fontweight='bold')
for _, row in df_valid.iterrows():
    ax5.annotate(f"V{row['num_vineta']}", (row['duracion_seg'], row['size_mb']),
                 textcoords="offset points", xytext=(3,3), fontsize=7)

# 5.6 Tabla resumen
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')
summary_data = [
    ["Total videos",          str(len(df_valid))],
    ["Duración total",        f"{df_valid['duracion_seg'].sum()/60:.1f} min"],
    ["Duración media",        f"{df_valid['duracion_seg'].mean():.0f}s"],
    ["Frames totales",        f"{df_valid['n_frames'].sum():,}"],
    ["FPS",                   "30"],
    ["Resoluciones",          ', '.join(res_counts.index.tolist())],
    ["Tamaño total",          f"{df_valid['size_mb'].sum():.0f} MB"],
    ["Señas ~2s/seña",        f"~{int(df_valid['duracion_seg'].sum()/2)} posibles"],
    ["Segmentos 30f/50%",     f"~{int(df_valid['n_frames'].sum()/(30*0.5)):,}"],
]
table = ax6.table(cellText=summary_data, colLabels=["Métrica", "Valor"],
                  cellLoc='left', loc='center', bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(9)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#D9E1F2')
ax6.set_title("Resumen del Dataset", fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(DATA_DIR / "eda_visualizaciones.png", dpi=150, bbox_inches='tight')
print(f"   Guardado: data/eda_visualizaciones.png")
plt.show()

# ── 6. Extracción de frames de muestra ───────────────────────────────────
print("\n6. EXTRAYENDO FRAMES DE MUESTRA...")
n_sample = min(6, len(df_valid))
sample_df = df_valid.sample(n=n_sample, random_state=42).sort_values('num_vineta')

fig2, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, (_, row) in enumerate(sample_df.iterrows()):
    cap = cv2.VideoCapture(row['ruta'])
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)  # frame central
    ret, frame = cap.read()
    cap.release()

    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        axes[i].imshow(frame_rgb)
    else:
        axes[i].set_facecolor('#f0f0f0')
        axes[i].text(0.5, 0.5, 'No disponible', ha='center', va='center',
                     transform=axes[i].transAxes)

    axes[i].set_title(
        f"Vineta {row['num_vineta']} — frame central\n"
        f"{row['resolucion']} | {row['duracion_seg']:.0f}s | {row['n_frames']}f",
        fontsize=9
    )
    axes[i].axis('off')

plt.suptitle("Frames Centrales — Muestra del Dataset LSP", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(DATA_DIR / "eda_frames_muestra.png", dpi=120, bbox_inches='tight')
print(f"   Guardado: data/eda_frames_muestra.png")
plt.show()

# ── 7. Alerta: dataset continuo → requiere segmentación ──────────────────
print("\n" + "╔" + "═"*60 + "╗")
print("║  ANÁLISIS DEL TIPO DE DATASET                              ║")
print("╠" + "═"*60 + "╣")
print(f"║  Tipo detectado: VIDEOS CONTINUOS (historias en LSP)       ║")
print(f"║  Duración media: {df_valid['duracion_seg'].mean():.0f}s (esperado <5s para señas aisladas)     ║")
print("╠" + "═"*60 + "╣")
print("║  OPCIONES PARA EL PIPELINE:                                ║")
print("║  A) Sliding Window (sin anotaciones):                       ║")
print("║     - Ventanas de 30f (1s) con stride de 15f (50% overlap)  ║")
print("║     - Clase = número de vineta (27 clases)                  ║")
print(f"║     - Genera ~{int(df_valid['n_frames'].sum()/(15)):,} segmentos para entrenamiento     ║")
print("║  B) Segmentación manual/automática:                         ║")
print("║     - Anotar tiempos inicio/fin de cada seña                ║")
print("║     - Usar herramienta ELAN o similar                       ║")
print("║     - Requiere anotadores expertos en LSP                   ║")
print("║  C) Reconocimiento continuo (CTC):                          ║")
print("║     - Modelo seq2seq con CTC loss                           ║")
print("║     - Requiere transcripciones a nivel de seña              ║")
print("╠" + "═"*60 + "╣")
print("║  RECOMENDACIÓN: Opción A (Sliding Window) para POC rápido   ║")
print("║  + Opción B para sistema de producción real                 ║")
print("╚" + "═"*60 + "╝")

print("\n✓ EDA COMPLETO")
print(f"  Archivos generados en data/:")
print(f"  - manifest.csv ({len(df)} filas)")
print(f"  - label2idx.json ({len(label2idx)} clases)")
print(f"  - idx2label.json")
print(f"  - class_weights.json")
print(f"  - eda_visualizaciones.png")
print(f"  - eda_frames_muestra.png")
