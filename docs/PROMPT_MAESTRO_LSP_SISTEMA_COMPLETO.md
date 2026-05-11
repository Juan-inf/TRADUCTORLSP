# PROMPT MAESTRO — SISTEMA INTEGRAL LSP → TEXTO EN CASTELLANO
## Traducción en Tiempo Real con Cámara + Interfaz Visual

---

## `<SISTEMA>`

**ROL:** Eres un arquitecto senior de sistemas de IA especializado en visión por computadora, Deep Learning y comunicación inclusiva. Tu misión es diseñar, implementar y desplegar un sistema de traducción de Lengua de Señas Peruana (LSP) a texto en castellano, operando en tiempo real desde cámara web y sobre videos pregrabados, con interfaz visual donde el usuario vea simultáneamente la imagen con las señas detectadas y el texto traducido.

**ENFOQUE:** El sistema debe ser inclusivo, robusto, desplegable en instituciones educativas y de salud del Perú, y accesible para personas sordas y oyentes sin conocimiento técnico previo.

---

## `<OBJETIVO>`

Desarrollar un sistema integral de comunicación inclusiva que:

1. Capture señas LSP en tiempo real desde cámara web (webcam, laptop, tablet, smartphone)
2. Procese videos pregrabados en formato MP4, AVI o MOV con señas LSP
3. Traduzca automáticamente las señas a texto en castellano con alta precisión y latencia <200ms
4. Muestre en pantalla la cámara con landmarks de manos superpuestos + el texto traducido simultáneamente
5. Sea entrenado con datasets reales de LSP: videos MP4 + imágenes + PDFs de señas peruanas
6. Sea desplegable como aplicación web accesible sin instalación por el usuario final

---

## `<DATASET_LSP>` — FUENTE ÚNICA OFICIAL

### 🗂️ UN SOLO GOOGLE DRIVE — CONTIENE TODO EL DATASET

```
URL ÚNICA DE ACCESO AL DATASET COMPLETO:
https://drive.google.com/drive/u/0/folders/1JjakUGGrAn9YwdHrkAUNCmuzyS8jdCse
```

**Esta carpeta compartida en Google Drive ES el dataset principal.**
Dentro de ella se encuentran los **3 archivos TAR** que conforman el 100% de los datos:

```
LSP_Dataset/  ← carpeta raíz del Google Drive compartido
├── Videos.tar        1.0 GB    │ Videos MP4 de señas LSP (fuente visual)
├── Keypoints.tar   690.2 MB    │ Landmarks pre-extraídos (input directo al modelo)
└── SRT.tar         191.5 KB    │ Subtítulos .srt en castellano (ground truth)
```

> ⚠️ NO buscar los datos en otras fuentes.
> TODO el entrenamiento, validación y evaluación se hace con estos 3 archivos.

---

### PASO 0 — MONTAR DRIVE Y EXTRAER LOS 3 TAR (obligatorio antes de todo)

```python
from google.colab import drive
import os, tarfile

# 1. Montar el Google Drive compartido
drive.mount('/content/drive')

# 2. Ajustar esta ruta al nombre real de la carpeta en tu Drive
#    Buscar la carpeta con ID: 1JjakUGGrAn9YwdHrkAUNCmuzyS8jdCse
DATASET_ROOT = "/content/drive/MyDrive/"  # cambiar si es necesario

# 3. Verificar que los 3 TAR existen
archivos_tar = {
    "Videos.tar":    {"size": "1.0 GB",   "md5": "3bd...e21", "descargas": 226},
    "Keypoints.tar": {"size": "690.2 MB", "md5": "706...75c", "descargas": 179},
    "SRT.tar":       {"size": "191.5 KB", "md5": "f31...129", "descargas": 109},
}

for nombre, meta in archivos_tar.items():
    encontrado = False
    for root, dirs, files in os.walk(DATASET_ROOT):
        if nombre in files:
            ruta = os.path.join(root, nombre)
            size_real = os.path.getsize(ruta) / (1024**3)
            print(f"✅ {nombre}: {ruta}  ({size_real:.3f} GB) — esperado: {meta['size']}")
            encontrado = True
            break
    if not encontrado:
        print(f"❌ {nombre} NO encontrado — verificar acceso al Drive compartido")

# 4. Extraer los 3 TAR en /content/lsp_dataset/
EXTRACT_DIR = "/content/lsp_dataset/"
os.makedirs(EXTRACT_DIR, exist_ok=True)

for nombre in archivos_tar:
    tar_path = None
    for root, dirs, files in os.walk(DATASET_ROOT):
        if nombre in files:
            tar_path = os.path.join(root, nombre)
            break

    if tar_path:
        dest = os.path.join(EXTRACT_DIR, nombre.replace(".tar", ""))
        os.makedirs(dest, exist_ok=True)
        print(f"📦 Extrayendo {nombre}...")
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=dest)
        print(f"   ✅ → {dest}")
    else:
        print(f"⚠️  {nombre}: no se pudo extraer")

# 5. Inspeccionar la estructura real extraída
print("\n📂 Estructura del dataset extraído:")
for root, dirs, files in os.walk(EXTRACT_DIR):
    nivel = root.replace(EXTRACT_DIR, '').count(os.sep)
    if nivel > 3: continue
    indent = '  ' * nivel
    print(f"{indent}{os.path.basename(root)}/")
    for f in sorted(files)[:3]:
        print(f"{indent}  {f}")
    if len(files) > 3:
        print(f"{indent}  ... ({len(files)} archivos total)")
```

---

### ARCHIVO 1 — `Videos.tar` | 1.0 GB | MD5: 3bd...e21 | 226 descargas

**Contenido:** Videos de señas LSP organizados por clase/palabra

**Uso en el sistema:**
- Fuente visual para el EDA (ver frames por clase, detectar drift de iluminación/fondo)
- Re-extraer landmarks con MediaPipe si Keypoints.tar no cubre alguna clase
- Validación visual del overlay de landmarks en la interfaz en tiempo real
- Segmentar clips usando timestamps del SRT.tar

```python
import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic

def extraer_landmarks_frame(frame, holistic_model):
    """Extrae 1662 landmarks de un frame: manos(21x2) + pose(33) + cara(468)"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = holistic_model.process(rgb)
    coords = []
    for landmark_list in [
        result.left_hand_landmarks,
        result.right_hand_landmarks,
        result.pose_landmarks,
        result.face_landmarks
    ]:
        if landmark_list:
            for lm in landmark_list.landmark:
                coords.append([lm.x, lm.y, lm.z])
        else:
            n = {0: 21, 1: 21, 2: 33, 3: 468}[len(coords) // 21 if coords else 0]
            coords.extend([[0.0, 0.0, 0.0]] * n)
    return np.array(coords)  # shape [1662, 3]

def procesar_video_lsp(video_path, n_frames=30):
    """Lee un MP4 del Videos.tar y devuelve secuencia de landmarks"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = set(np.linspace(0, total - 1, n_frames, dtype=int))
    secuencia = []
    frame_idx = 0

    with mp_holistic.Holistic(min_detection_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in indices:
                lm = extraer_landmarks_frame(frame, holistic)
                secuencia.append(lm)
            frame_idx += 1
    cap.release()

    while len(secuencia) < n_frames:
        secuencia.append(np.zeros((1662, 3)))

    return np.array(secuencia[:n_frames])  # shape [30, 1662, 3]
```

---

### ARCHIVO 2 — `Keypoints.tar` | 690.2 MB | MD5: 706...75c | 179 descargas

**Contenido:** Landmarks/keypoints pre-extraídos de todos los videos LSP

> ⚡ **USAR ESTE ARCHIVO PRIMERO.** Evita re-procesar el 1.0 GB de videos con MediaPipe.
> Cargar directamente como input del modelo LSTM / ST-GCN.

```python
import numpy as np
import os, glob, json
import pandas as pd

KEYPOINTS_DIR = "/content/lsp_dataset/Keypoints/"

def inspeccionar_keypoints(keypoints_dir):
    """Paso 1: inspeccionar la estructura real antes de cargar"""
    archivos = list(glob.glob(os.path.join(keypoints_dir, "**/*"), recursive=True))
    ext_count = {}
    for f in archivos:
        if os.path.isfile(f):
            ext = os.path.splitext(f)[1].lower()
            ext_count[ext] = ext_count.get(ext, 0) + 1

    print("Extensiones encontradas:", ext_count)
    print("Total archivos:", len(archivos))
    carpetas = set(os.path.basename(os.path.dirname(f))
                   for f in archivos if os.path.isfile(f))
    print(f"Clases/carpetas únicas: {len(carpetas)}")
    print("Muestra de clases:", list(carpetas)[:10])
    return ext_count

def cargar_keypoints_lsp(keypoints_dir):
    """Carga los keypoints según el formato real del archivo"""
    ext_count = inspeccionar_keypoints(keypoints_dir)
    X, y, rutas = [], [], []

    if ".npy" in ext_count:
        print("\nFormato detectado: .npy")
        for npy_path in sorted(glob.glob(os.path.join(keypoints_dir, "**/*.npy"), recursive=True)):
            clase = os.path.basename(os.path.dirname(npy_path))
            kp = np.load(npy_path)
            X.append(kp)
            y.append(clase)
            rutas.append(npy_path)

    elif ".csv" in ext_count:
        print("\nFormato detectado: .csv")
        for csv_path in sorted(glob.glob(os.path.join(keypoints_dir, "**/*.csv"), recursive=True)):
            clase = os.path.basename(os.path.dirname(csv_path))
            kp = pd.read_csv(csv_path).values
            X.append(kp)
            y.append(clase)
            rutas.append(csv_path)

    elif ".json" in ext_count:
        print("\nFormato detectado: .json")
        for json_path in sorted(glob.glob(os.path.join(keypoints_dir, "**/*.json"), recursive=True)):
            clase = os.path.basename(os.path.dirname(json_path))
            with open(json_path) as jf:
                data = json.load(jf)
            kp = np.array(data) if isinstance(data, list) else np.array(data.get("keypoints", []))
            X.append(kp)
            y.append(clase)

    print(f"\nTotal muestras cargadas: {len(X)}")
    print(f"Clases únicas: {len(set(y))}")
    if X:
        print(f"Shape de una muestra: {np.array(X[0]).shape}")
    return X, y, rutas

X_kp, y_kp, rutas_kp = cargar_keypoints_lsp(KEYPOINTS_DIR)

def normalizar_keypoints(kp_sequence):
    """Centra las coordenadas respecto a la muñeca derecha para invarianza posicional"""
    kp = np.array(kp_sequence)
    referencia = kp[:, 0:1, :]  # shape [T, 1, 3]
    kp_normalizado = kp - referencia
    return kp_normalizado

X_norm = np.array([normalizar_keypoints(kp) for kp in X_kp])
print(f"Shape final del dataset: {X_norm.shape}")
```

---

### ARCHIVO 3 — `SRT.tar` | 191.5 KB | MD5: f31...129 | 109 descargas

**Contenido:** Archivos `.srt` con timestamps y transcripciones en castellano

> 🎯 **ESTE ES EL GROUND TRUTH.** Sin los SRT no se puede calcular WER ni BLEU Score.
> Alinear cada `.srt` con su video/keypoints por nombre de archivo base.

```python
import re, os, glob

SRT_DIR = "/content/lsp_dataset/SRT/"

def parsear_srt(srt_path):
    """Parsea un archivo .srt → lista de segmentos con timestamp y texto castellano"""
    with open(srt_path, "r", encoding="utf-8", errors="replace") as f:
        contenido = f.read()

    patron = r"(\d+)\n(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\n([\s\S]*?)(?=\n\n|\Z)"
    segmentos = re.findall(patron, contenido.strip())

    resultado = []
    for num, inicio, fin, texto in segmentos:
        resultado.append({
            "id":               int(num),
            "inicio":           inicio,
            "fin":              fin,
            "texto_castellano": texto.strip().replace("\n", " "),
        })
    return resultado

def construir_ground_truth_completo(srt_dir):
    """Crea dict {nombre_video: {transcripcion_completa, segmentos, n_segmentos}}"""
    ground_truth = {}
    srt_files = glob.glob(os.path.join(srt_dir, "**/*.srt"), recursive=True)

    for srt_path in sorted(srt_files):
        nombre = os.path.splitext(os.path.basename(srt_path))[0]
        segs = parsear_srt(srt_path)
        ground_truth[nombre] = {
            "transcripcion_completa": " ".join(s["texto_castellano"] for s in segs),
            "segmentos":              segs,
            "n_segmentos":            len(segs),
            "srt_path":               srt_path,
        }

    print(f"Total SRT cargados: {len(ground_truth)}")
    muestra = list(ground_truth.items())[:2]
    for nombre, datos in muestra:
        print(f"  {nombre}: '{datos['transcripcion_completa'][:60]}...'")
    return ground_truth

ground_truth = construir_ground_truth_completo(SRT_DIR)

def calcular_wer(referencia, hipotesis):
    """Word Error Rate entre transcripción SRT y predicción del modelo"""
    import numpy as np
    ref_words = referencia.lower().split()
    hyp_words = hipotesis.lower().split()
    d = np.zeros((len(ref_words)+1, len(hyp_words)+1))
    for i in range(len(ref_words)+1): d[i][0] = i
    for j in range(len(hyp_words)+1): d[0][j] = j
    for i in range(1, len(ref_words)+1):
        for j in range(1, len(hyp_words)+1):
            cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)
```

---

### PIPELINE COMPLETO — 3 TAR INTEGRADOS

```
Google Drive (URL única):
https://drive.google.com/drive/u/0/folders/1JjakUGGrAn9YwdHrkAUNCmuzyS8jdCse
│
├── Videos.tar (1.0 GB)
│   └── extraer → /lsp_dataset/Videos/[clase]/video.mp4
│       ├── EDA visual: distribución de clases, duración, fps, muestra de frames
│       ├── Detectar riesgos: drift de iluminación, variación de fondos, desbalance
│       └── Backup: re-extraer landmarks si Keypoints.tar no cubre alguna clase
│
├── Keypoints.tar (690.2 MB) ← INPUT PRINCIPAL DEL MODELO
│   └── extraer → /lsp_dataset/Keypoints/[clase]/kp.npy (o .csv o .json)
│       ├── Cargar directamente como tensores [T, N_keypoints, 3]
│       ├── Normalizar respecto a muñeca derecha
│       ├── Splits 70/15/15 estratificados
│       └── Entrenar: LSTM Bidireccional / ST-GCN / MLP (baseline)
│
└── SRT.tar (191.5 KB) ← GROUND TRUTH DE TRADUCCIÓN
    └── extraer → /lsp_dataset/SRT/[video].srt
        ├── Parsear timestamps + texto castellano
        ├── Alinear con Keypoints/Videos por nombre de archivo
        ├── Calcular WER (Word Error Rate) del sistema
        └── Calcular BLEU Score de la traducción a frases completas
```

**Verificación MD5 de integridad (ejecutar antes de entrenar):**

```python
import hashlib

def verificar_md5_prefijo(filepath, md5_prefix):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    md5_real = hash_md5.hexdigest()
    ok = md5_real.startswith(md5_prefix)
    status = "✅" if ok else "❌"
    print(f"{status} {os.path.basename(filepath)}: MD5={md5_real[:8]}... (esperado: {md5_prefix}...)")
    return ok

for nombre, prefijo in [("Videos.tar","3bd"), ("Keypoints.tar","706"), ("SRT.tar","f31")]:
    for root, dirs, files in os.walk(DATASET_ROOT):
        if nombre in files:
            verificar_md5_prefijo(os.path.join(root, nombre), prefijo)
            break
```

---

### FUENTES COMPLEMENTARIAS (opcionales — solo si el dataset TAR no cubre ciertas señas)

- **Imágenes:** MINEDU Perú, CPAL, FENAL, Kaggle "Peruvian Sign Language"
  - Estructura: `/dataset/imagenes/[SEÑA]/img_001.jpg`
  - Procesamiento: MediaPipe Hands → landmarks `[63,]` por imagen
- **PDFs LSP:** MINEDU, CPAL, FENAL → diccionario JSON con `pdfplumber`
  - Uso: fallback semántico cuando confianza del modelo < 60%

---

## `<ARQUITECTURA_DL>`

### MÓDULO 1 — CAPTURA Y DETECCIÓN
- **OpenCV:** `cv2.VideoCapture(0)` para webcam o lectura de video archivo
- **MediaPipe Holistic:** manos (21 pts×2), pose (33 pts), cara (468 pts) por frame
- **YOLOv8-pose:** detección de persona + bounding box para recorte ROI
- **Normalización:** landmarks respecto a muñeca derecha (punto 0) → invarianza posicional

### MÓDULO 2 — CLASIFICACIÓN DE SEÑAS

**2A. Señas estáticas (alfabeto dactilológico, números):**
- MLP sobre vector de landmarks de 1 frame → salida: letra/número con confianza

**2B. Señas dinámicas (palabras y frases LSP):**
- Secuencia `[30 frames × 1662 keypoints × 3 coords]`
- → LSTM Bidireccional + Attention → clase de seña
- → Alternativa: ST-GCN sobre grafo anatómico de manos
- → Alternativa avanzada: VideoMAE fine-tuneado en dataset LSP

### MÓDULO 3 — POST-PROCESAMIENTO LINGÜÍSTICO
- Secuencia de clases predichas → BERT español (HuggingFace)
- Corrección gramatical: ajustar orden SOV (LSP) a SVO (castellano)
- Suavizado temporal: buffer de confianza deslizante (evitar parpadeo de texto)
- Fallback: si confianza < 0.60 → consultar diccionario LSP extraído de PDFs

### MÓDULO 4 — INTERFAZ EN TIEMPO REAL
- **Frontend:** React + TailwindCSS + Canvas API para overlay de landmarks
- **Panel izquierdo:** cámara/video con skeleton de manos superpuesto
- **Panel derecho:** texto traducido en castellano (≥18px), actualización fluida
- **Comunicación:** WebSocket (FastAPI) para streaming de predicciones en vivo
- **Overlay:** bounding box de manos, conexiones de dedos, nombre de seña, confianza %

### MODELOS A COMPARAR

| Señas dinámicas | Señas estáticas |
|---|---|
| LSTM Bidireccional + Attention (baseline DL) | MLP sobre landmarks planos (baseline) |
| ST-GCN — grafo anatómico de manos | Random Forest sobre vectores keypoints |
| VideoMAE — ViT preentrenado en Kinetics | EfficientNet sobre imagen recortada |
| CNN-LSTM — ResNet50 + LSTM temporal | KNN (k=5) como referencia mínima |

---

## `<INTERFAZ_UI>` — ESPECIFICACIÓN COMPLETA

### DISEÑO DE PANTALLA PRINCIPAL

```
┌─────────────────────────────────────────────────────────────┐
│  🤟 Sistema LSP → Castellano | EN VIVO ● [CÁMARA] [VIDEO]  │
├────────────────────────────┬────────────────────────────────┤
│  📷 CÁMARA — SEÑAS LSP     │  📝 TRADUCCIÓN EN CASTELLANO   │
│                            │                                │
│  [imagen con skeleton      │  "Hola, buenos días.           │
│   de manos superpuesto     │   ¿Cómo estás hoy?"           │
│   en tiempo real]          │                                │
│                            │  Confianza: ████████░░ 87%     │
│  🟢 Manos detectadas       │  ● 24 fps  ● 143ms latencia   │
│  Seña: BUENOS_DIAS         │                                │
├────────────────────────────┴────────────────────────────────┤
│  📜 HISTORIAL: "Hola" | "Me llamo..." | "Necesito ayuda"   │
│  [🔊 Leer en voz alta] [📋 Copiar texto] [💾 Exportar TXT] │
└─────────────────────────────────────────────────────────────┘
```

### COMPONENTES OBLIGATORIOS

**Panel izquierdo — Cámara con señas:**
- Video en vivo o video subido con overlay Canvas de skeleton de manos
- Puntos de landmarks (círculos) + conexiones de dedos (líneas)
- Bounding box alrededor de la mano con nombre de seña actual
- Indicador de estado: 🟢 Manos detectadas / 🔴 Sin detección / 🟡 Procesando

**Panel derecho — Texto en castellano:**
- Texto traducido en fuente ≥18px, actualización suave sin parpadeo
- Barra de confianza con color: verde >75% | amarillo 50-75% | rojo <50%
- Historial de últimas 10 frases, scrolleable
- Botón TTS: leer en voz alta con Web Speech API (nativa del navegador)
- Exportar historial como .txt o .pdf

**Controles y modos:**
- `[ Cámara en vivo ]` → webcam activa, traducción continua a 24+ fps
- `[ Subir video ]` → procesar .mp4 guardado, transcripción con timestamps
- `[ Modo Educativo ]` → nombre de la seña + imagen de referencia del diccionario PDF
- `[ Pausar / Reanudar ]` | `[ Limpiar historial ]`

**Barra de métricas en vivo:**
- FPS actual | Latencia de inferencia (ms) | Confianza de predicción (%)

**Implementación técnica:**
- Frontend: React + TailwindCSS + Canvas API para el overlay de landmarks
- WebSocket: FastAPI → streaming de predicciones en tiempo real al frontend
- TTS: Web Speech API nativa del navegador (sin costo, funciona offline)

---

## `<ENTRENAMIENTO>`

**Preprocesamiento — orden de prioridad:**
1. `Keypoints.tar` (690 MB) → **usar directamente** sin re-procesar → shape `[n_frames, 1662, 3]` ⚡
2. `Videos.tar` (1.0 GB) → re-extraer landmarks solo si Keypoints.tar no cubre todas las clases
3. `SRT.tar` (191 KB) → parsear `.srt` → ground truth castellano → calcular WER y BLEU
4. Imágenes complementarias → MediaPipe Hands → landmarks `.npy` shape `[63,]`
5. PDFs LSP → `pdfplumber` → diccionario `.json` para validación de etiquetas y fallback

**Aumentación específica para LSP:**
- Flip horizontal (solo señas no lateralizadas/asimétricas)
- Variación de velocidad: ×0.75 y ×1.25
- Jitter brillo/contraste: ±20%
- Rotación leve: ±10°
- Ruido gaussiano sobre landmarks: σ=0.01

**Splits:**
- Train 70% | Validación 15% | Test 15%
- Estratificación por clase, sin mezcla de sujetos entre splits

**Configuración de entrenamiento:**
- Épocas: 50-100 con early stopping (paciencia=10)
- Optimizador: AdamW (lr=1e-4, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR
- Loss: CrossEntropyLoss con pesos por clase (desbalance)
- Batch size: 16-32 | Mixed precision: `torch.cuda.amp.autocast()`
- Logging: Weights & Biases (wandb)

**Baseline mínimo (antes del modelo DL):**
- Flatten landmarks → KNN + Naive Bayes + Regresión Logística
- Métrica: F1-score macro (multiclase desbalanceada)
- El modelo DL debe superar este baseline claramente

---

## `<STACK_TECNOLOGICO>`

| Categoría | Herramientas |
|---|---|
| Captura de video | OpenCV (cv2), decord, ffmpeg-python |
| Detección de landmarks | MediaPipe Holistic, YOLOv8-pose (Ultralytics) |
| Deep Learning | PyTorch 2.x, torchvision, HuggingFace Transformers |
| Modelos | LSTM Bidireccional, ST-GCN (MMAction2), VideoMAE, MLP |
| NLP post-proceso | BERT español (HuggingFace), corrección SOV→SVO |
| PDFs y datos | pdfplumber, PyMuPDF, pandas, numpy |
| Backend | FastAPI + WebSocket, ONNX Runtime |
| Frontend | React + TailwindCSS, Canvas API, Web Speech API (TTS) |
| Experiment tracking | Weights & Biases (wandb), MLflow |
| Despliegue | HuggingFace Spaces, Docker, Google Colab Pro |
| **Latencia objetivo** | **< 200ms por seña** |
| **FPS objetivo** | **≥ 24 fps en interfaz** |

---

## `<METRICAS_EVALUACION>`

- **F1-score macro y weighted** — métrica principal (multiclase, posible desbalance)
- Accuracy top-1 y top-5 | Precision y Recall por clase de seña
- Word Error Rate (WER) para secuencias de señas continuas
- BLEU Score para frases completas traducidas
- Latencia de inferencia: objetivo <200ms por seña
- FPS de la interfaz: objetivo ≥24 fps para experiencia fluida
- Tasa de detección de manos: % de frames con manos correctamente detectadas
- Matriz de confusión normalizada por grupos de señas similares

---

## `<ENTREGABLES_CURSO>`

| # | Entregable | Descripción |
|---|------------|-------------|
| 1 | **README.md** | Descripción del problema, dataset, instalación con versiones exactas, cómo ejecutar, tabla de resultados, estructura de carpetas |
| 2 | **01_eda_lsp.ipynb** | EDA: estadísticas de los 3 TAR, distribución de clases, leakage + desbalance + drift, visualizaciones |
| 3 | **02_preprocessing.ipynb** | Pipeline unificado MP4 + imágenes + PDFs, landmarks, normalización, aumentación, splits |
| 4 | **03_baseline.ipynb** | KNN + Naive Bayes + LogReg sobre landmarks planos, F1-macro, justificación de métrica |
| 5 | **04_deep_learning.ipynb** | Mínimo 2 arquitecturas comparadas, tabla de métricas, checkpoint .pt + ONNX |
| 6 | **05_resultados.ipynb** | Curvas de aprendizaje, matriz de confusión normalizada, tabla Modelo/Accuracy/F1/Latencia |
| 7 | **api/main.py** | FastAPI + WebSocket, inferencia en tiempo real desde cámara o video, latencia <200ms |
| 8 | **frontend/** | Cámara + overlay landmarks + texto castellano + TTS + exportar + modos |
| 9 | **Modelo optimizado** | Checkpoint .pt + versión ONNX para despliegue rápido |
| 10 | **demo_script.md** | Guion demo 5-10 min: EDA → baseline → pipeline → logs W&B → demo cámara real |

---

## `<INSTRUCCION_FINAL>`

Desarrolla el sistema completo en este orden estricto:

1. **Descomprimir y verificar MD5** de los 3 archivos TAR: `Videos.tar` (1.0 GB) + `Keypoints.tar` (690 MB) + `SRT.tar` (191 KB)
2. **EDA completo:** explorar los 3 archivos TAR (estructura, clases, duración), estadísticas descriptivas, riesgos (leakage, desbalance, drift)
3. **Pipeline de preprocesamiento:** cargar Keypoints.tar directamente, parsear SRT.tar como ground truth, alinear con Videos.tar por nombre de archivo
4. **Entrenar baseline** (KNN/LogReg) y luego el **modelo DL** (mínimo 2 arquitecturas comparadas)
5. **Construir la API FastAPI** con WebSocket para streaming de predicciones
6. **Construir la interfaz web:** panel cámara izquierdo con overlay de landmarks + panel texto derecho en castellano
7. **Exportar el modelo a ONNX** y optimizar para latencia <200ms a ≥24 fps
8. **Preparar la demo** ejecutable sin errores de arriba a abajo para presentación al curso

**El sistema final debe ser funcional, inclusivo, reproducible y desplegable en instituciones educativas y de salud del Perú.**

**La interfaz es el corazón del sistema:** el usuario debe ver la cámara con las señas detectadas y el texto traducido en castellano simultáneamente, en tiempo real, sin necesidad de conocimiento técnico previo.
