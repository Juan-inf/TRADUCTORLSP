# PROMPT MAESTRO — SISTEMA INTEGRAL LSP → TEXTO EN CASTELLANO
## Traducción en Tiempo Real con Cámara + Interfaz Visual
## Versión 2.0 — Actualizado Sprint 9 (Junio 2026)

---

## `<SISTEMA>`

**ROL:** Eres un arquitecto senior de sistemas de IA especializado en visión por computadora, Deep Learning y comunicación inclusiva. Tu misión es diseñar, implementar y desplegar un sistema de traducción de Lengua de Señas Peruana (LSP) a texto en castellano, operando en tiempo real desde cámara web y sobre videos pregrabados, con interfaz visual donde el usuario vea simultáneamente la imagen con las señas detectadas y el texto traducido.

**ENFOQUE:** El sistema debe ser inclusivo, robusto, desplegable en instituciones educativas y de salud del Perú, y accesible para personas sordas y oyentes sin conocimiento técnico previo.

**REPOSITORIO:** `/Users/usuario/Documents/TRADUCTOR_LSP/` — rama `Semana8`, commits S9: `fbfeae7 · 1668456 · 4394a6c`

---

## `<ESTADO_ACTUAL>` — Sprint 9 completado (referencia antes de continuar)

> ⚠️ El proyecto ha avanzado hasta el Sprint 9. Lee esta sección ANTES de generar cualquier código para no duplicar trabajo ya realizado.

### Lo que YA existe y funciona

| Componente | Archivo | Estado |
|-----------|---------|--------|
| Dataset combinado NPZ | `data/dataset_lstm.npz` [~7000, 30, 150] | ✅ Generado |
| Modelo LSTM Bidir+Attn | `checkpoints/lstm_signs.pt` (10 MB) | ✅ Entrenado |
| ONNX deploy | `checkpoints/lstm_signs.onnx` (10 MB, opset 17) | ✅ Exportado |
| RF fallback | `checkpoints/rf_signs.pkl` (435 MB) | ✅ Disponible |
| Tablero experimentos | `logs/runs.csv` (8 corridas S5–S9) | ✅ Registrado |
| Demo Gradio | `demo/app_gradio.py` + `spaces/` | ✅ Funcional |
| Config centralizado | `configs/config.yaml` | ✅ |
| Figuras informe | `figs/s9_fig01–13_*.png` (13 figuras) | ✅ |

### Métricas actuales del mejor modelo (LSTM Bidir S9)

```
Modelo:     LSPLSTMBidir (BiLSTM×2 hidden=256 + TemporalAttention + Head)
Parámetros: ~10M
n_clases:   482 (filtradas de 1,163 con ≥2 muestras en train)
F1-val:     0.0365  ← mejor resultado del proyecto (factor 8× sobre RF)
F1-test:    0.0109
Acc-test:   2.62%   (12.5× sobre aleatorio = 1/482 = 0.21%)
Overfitting gap: 95%  (F1-train=0.759, F1-val=0.037)
Latencia ONNX:   <50ms  (CoreML/CPU, macOS)
```

### Deuda técnica pendiente (Sprint 10)

```
[ ] GroupKFold para LSTM (actualmente usa StratifiedShuffleSplit — sesgo de validación)
[ ] HPO formal LSTM con Optuna (actualmente HP fijados manualmente)
[ ] Label smoothing + reducción hidden (mitigar overfitting 95%)
[ ] Deploy permanente HuggingFace Spaces (pendiente hf auth login)
[ ] MLflow UI activo (código listo en logs/, pendiente mlflow ui)
[ ] Integración SRT ground truth para WER/BLEU (datos SRT.tar no procesados aún)
[ ] Pipeline de inferencia continua desde webcam real (actualmente frame-a-frame en Gradio)
```

---

## `<OBJETIVO>`

Desarrollar un sistema integral de comunicación inclusiva que:

1. **[PARCIAL ✅]** Capture señas LSP en tiempo real desde cámara web — demo Gradio funcional con MediaPipe Holistic
2. **[PENDIENTE]** Procese videos pregrabados en formato MP4, AVI o MOV con señas LSP — pipeline frame-a-frame disponible, falta interfaz de upload video completo
3. **[PARCIAL ✅]** Traduzca señas a texto en castellano — F1-val=0.0365, latencia <50ms ONNX; objetivo final: >70% F1 con más datos
4. **[PARCIAL ✅]** Muestre cámara con landmarks superpuestos + texto simultáneamente — Gradio implementado, falta UI React completa
5. **[PARCIAL ✅]** Entrenado con datasets reales LSP — 7,536 muestras de 3 fuentes PKL; datasets Videos.tar y SRT.tar pendientes de integrar
6. **[PARCIAL ✅]** Desplegable como aplicación web — Gradio en HF Spaces; falta deploy permanente y UI avanzada

**Métrica objetivo final del sistema:** F1-macro > 0.70 sobre vocabulario LSP de uso diario (≥200 señas con ≥20 muestras/clase)

---

## `<DATASET_LSP>` — FUENTE OFICIAL GOOGLE DRIVE

### Dataset original — 3 archivos TAR

```
URL ÚNICA DE ACCESO AL DATASET COMPLETO:
https://drive.google.com/drive/u/0/folders/1JjakUGGrAn9YwdHrkAUNCmuzyS8jdCse
```

```
LSP_Dataset/  ← carpeta raíz del Google Drive compartido
├── Videos.tar        1.0 GB    │ Videos MP4 de señas LSP (fuente visual)
├── Keypoints.tar   690.2 MB    │ Landmarks pre-extraídos (input directo al modelo)
└── SRT.tar         191.5 KB    │ Subtítulos .srt en castellano (ground truth)
```

> ⚠️ NO buscar los datos en otras fuentes.

### PASO 0 — MONTAR DRIVE Y EXTRAER LOS 3 TAR

```python
from google.colab import drive
import os, tarfile

drive.mount('/content/drive')
DATASET_ROOT = "/content/drive/MyDrive/"

archivos_tar = {
    "Videos.tar":    {"size": "1.0 GB",   "md5": "3bd...e21"},
    "Keypoints.tar": {"size": "690.2 MB", "md5": "706...75c"},
    "SRT.tar":       {"size": "191.5 KB", "md5": "f31...129"},
}

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
```

---

### ARCHIVO 1 — `Videos.tar` | 1.0 GB | 226 descargas

**Uso en el sistema:**
- EDA visual: distribución de clases, duración, fps, muestra de frames
- Re-extraer landmarks con MediaPipe si `Keypoints.tar` no cubre alguna clase
- Validación visual del overlay de landmarks en la interfaz en tiempo real
- Segmentar clips usando timestamps del `SRT.tar`

```python
import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic

def extraer_landmarks_frame(frame, holistic_model):
    """Extrae 75 landmarks LSP de un frame: pose(33) + rhand(21) + lhand(21)"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = holistic_model.process(rgb)

    N_DIMS = 150  # 75 keypoints × 2 coords (x, y)
    row = []

    # Orden: pose(33×2) + left_hand(21×2) + right_hand(21×2) = 150
    for landmark_list, n in [
        (result.pose_landmarks,       33),
        (result.left_hand_landmarks,  21),
        (result.right_hand_landmarks, 21),
    ]:
        if landmark_list:
            xs = [lm.x for lm in landmark_list.landmark[:n]]
            ys = [lm.y for lm in landmark_list.landmark[:n]]
        else:
            xs, ys = [0.0] * n, [0.0] * n
        row.extend(xs)
        row.extend(ys)

    return np.array(row, dtype=np.float32)  # shape [150]

def procesar_video_lsp(video_path, n_frames=30):
    """Lee un MP4 y devuelve secuencia de landmarks [T=30, 150]"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = set(np.linspace(0, total - 1, n_frames, dtype=int))
    secuencia = []
    idx = 0

    with mp_holistic.Holistic(min_detection_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if idx in indices:
                secuencia.append(extraer_landmarks_frame(frame, holistic))
            idx += 1
    cap.release()

    while len(secuencia) < n_frames:
        secuencia.append(np.zeros(150))

    return np.array(secuencia[:n_frames])  # shape [30, 150]
```

> **NOTA CRÍTICA:** Los keypoints del proyecto usan **150 dims** (75 kp × 2 coords), NO 1662. Se descartan las coordenadas Z y los 468 puntos faciales para reducir ruido y dimensionalidad.

---

### ARCHIVO 2 — `Keypoints.tar` | 690.2 MB | 179 descargas

**Contenido:** Landmarks pre-extraídos. Dentro encontrarás subcarpetas PKL:

```
Keypoints/
├── pkl/           ← 3,684 muestras, 1,086 clases (viñetas segmentadas)
├── glosas_pkl/    ←   252 muestras,   143 clases (señas EAF timing)
└── abecedario_pkl/← 3,600 muestras,    24 clases (A–Y dactilológico)
```

> ⚡ **USAR ESTE ARCHIVO PRIMERO.** El formato PKL ya tiene los keypoints extraídos. No re-procesar los 1.0 GB de video innecesariamente.

```python
import pickle, re, glob, numpy as np
from pathlib import Path

def pkl_to_sequence(pkl_path, n_frames=30, n_dims=150):
    """PKL (lista de dicts por frame) → array [T=30, 150]"""
    with open(pkl_path, "rb") as f:
        frames = pickle.load(f)  # lista de dicts con claves: pose, left_hand, right_hand

    seq = []
    for fr in frames:
        row = []
        for key, n in [("pose", 33), ("left_hand", 21), ("right_hand", 21)]:
            d = fr.get(key, {})
            x = d.get("x", [0.0] * n)
            y = d.get("y", [0.0] * n)
            if len(x) != n:
                x, y = [0.0] * n, [0.0] * n
            row.extend(x)
            row.extend(y)
        seq.append(row)  # 150 dims

    # Resamplear a n_frames
    if len(seq) > n_frames:
        idx = np.linspace(0, len(seq) - 1, n_frames, dtype=int)
        seq = [seq[i] for i in idx]
    while len(seq) < n_frames:
        seq.append(seq[-1] if seq else [0.0] * n_dims)

    return np.array(seq[:n_frames], dtype=np.float32)  # [30, 150]

def label_from_filename(name):
    stem = Path(name).stem
    m = re.match(r"^(.+)_(\d+)$", stem)
    return m.group(1).upper() if m else stem.upper()

def cargar_todos_los_pkl(keypoints_dir):
    """Carga las 3 subcarpetas PKL → X [N, 30, 150], y [N]"""
    fuentes = {
        "pkl":            (Path(keypoints_dir) / "pkl",            3684, 1086),
        "glosas_pkl":     (Path(keypoints_dir) / "glosas_pkl",      252,  143),
        "abecedario_pkl": (Path(keypoints_dir) / "abecedario_pkl", 3600,   24),
    }
    X, y = [], []
    for nombre, (ruta, n_aprox, c_aprox) in fuentes.items():
        if not ruta.exists():
            print(f"⚠️  {nombre}: carpeta no encontrada en {ruta}")
            continue
        pkls = sorted(ruta.glob("**/*.pkl"))
        print(f"📦 {nombre}: {len(pkls)} archivos (esperado ~{n_aprox}, ~{c_aprox} clases)")
        for p in pkls:
            seq = pkl_to_sequence(p)
            if seq is not None:
                X.append(seq)
                y.append(label_from_filename(p.name))
    return np.array(X, dtype=np.float32), np.array(y)

X, y = cargar_todos_los_pkl("/content/lsp_dataset/Keypoints/")
print(f"Dataset cargado: {X.shape}, {len(set(y))} clases únicas")
# → Dataset cargado: (7536, 30, 150), 1163 clases únicas
```

---

### ARCHIVO 3 — `SRT.tar` | 191.5 KB | 109 descargas

**Contenido:** Archivos `.srt` con timestamps y transcripciones en castellano.

> 🎯 **ESTE ES EL GROUND TRUTH.** Sin los SRT no se puede calcular WER ni BLEU Score.

```python
import re, os, glob
import numpy as np

SRT_DIR = "/content/lsp_dataset/SRT/"

def parsear_srt(srt_path):
    with open(srt_path, "r", encoding="utf-8", errors="replace") as f:
        contenido = f.read()
    patron = r"(\d+)\n(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\n([\s\S]*?)(?=\n\n|\Z)"
    segmentos = re.findall(patron, contenido.strip())
    return [{"id": int(n), "inicio": i, "fin": fi, "texto": t.strip().replace("\n", " ")}
            for n, i, fi, t in segmentos]

def calcular_wer(referencia, hipotesis):
    """Word Error Rate entre ground truth SRT y predicción del modelo"""
    ref = referencia.lower().split()
    hyp = hipotesis.lower().split()
    d = np.zeros((len(ref)+1, len(hyp)+1))
    for i in range(len(ref)+1): d[i][0] = i
    for j in range(len(hyp)+1): d[0][j] = j
    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return d[len(ref)][len(hyp)] / max(len(ref), 1)

ground_truth = {}
for srt_path in sorted(glob.glob(os.path.join(SRT_DIR, "**/*.srt"), recursive=True)):
    nombre = os.path.splitext(os.path.basename(srt_path))[0]
    segs = parsear_srt(srt_path)
    ground_truth[nombre] = {
        "transcripcion": " ".join(s["texto"] for s in segs),
        "segmentos":     segs,
    }
print(f"SRT cargados: {len(ground_truth)}")
```

---

## `<BUILD_COMBINED_DATASET>` — Pipeline de construcción del NPZ

```python
"""
build_combined_dataset.py
Combina las 3 fuentes PKL → data/dataset_lstm.npz + data/lstm_label2idx.json
Ejecutar UNA VEZ antes de entrenar. Seed: 42.
"""
import pickle, json, re, numpy as np, pathlib, warnings
from collections import Counter
warnings.filterwarnings("ignore")

ROOT = pathlib.Path(".")
N_FRAMES, N_DIMS = 30, 150
MIN_SAMPLES_PER_CLASS = 2  # filtrar clases con < 2 muestras

PKL_SOURCES = [
    ROOT / "data" / "Keypoints" / "pkl",
    ROOT / "data" / "Keypoints" / "glosas_pkl",
    ROOT / "data" / "Keypoints" / "abecedario_pkl",
]

all_X, all_y = [], []

for fuente in PKL_SOURCES:
    pkls = sorted(fuente.glob("**/*.pkl"))
    print(f"\n{fuente.name}: {len(pkls)} PKL")
    loaded = 0
    for p in pkls:
        try:
            with open(p, "rb") as f:
                frames = pickle.load(f)
            if not frames: continue
            seq = []
            for fr in frames:
                row = []
                for key, n in [("pose", 33), ("left_hand", 21), ("right_hand", 21)]:
                    d = fr.get(key, {})
                    x = d.get("x", [0.0]*n)[:n]
                    y_lm = d.get("y", [0.0]*n)[:n]
                    if len(x) != n: x, y_lm = [0.0]*n, [0.0]*n
                    row.extend(x); row.extend(y_lm)
                seq.append(row)
            if len(seq) > N_FRAMES:
                idx = np.linspace(0, len(seq)-1, N_FRAMES, dtype=int)
                seq = [seq[i] for i in idx]
            while len(seq) < N_FRAMES: seq.append(seq[-1] if seq else [0.0]*N_DIMS)
            stem = p.stem
            m = re.match(r"^(.+)_(\d+)$", stem)
            label = m.group(1).upper() if m else stem.upper()
            all_X.append(np.array(seq[:N_FRAMES], dtype=np.float32))
            all_y.append(label)
            loaded += 1
        except Exception as e:
            pass
    print(f"  → {loaded} cargados")

# Filtrar clases con < MIN_SAMPLES_PER_CLASS
counter = Counter(all_y)
valid_labels = {lbl for lbl, cnt in counter.items() if cnt >= MIN_SAMPLES_PER_CLASS}
mask = [lbl in valid_labels for lbl in all_y]
all_X = [x for x, m in zip(all_X, mask) if m]
all_y = [lbl for lbl, m in zip(all_y, mask) if m]

# Codificar labels
label2idx = {lbl: i for i, lbl in enumerate(sorted(set(all_y)))}
y_int = np.array([label2idx[lbl] for lbl in all_y])
X_np = np.array(all_X, dtype=np.float32)

print(f"\n✅ Dataset final: {X_np.shape} | {len(label2idx)} clases")

np.savez_compressed("data/dataset_lstm.npz", X=X_np, y=y_int)
with open("data/lstm_label2idx.json", "w") as f:
    json.dump(label2idx, f, ensure_ascii=False)
print("Guardado: data/dataset_lstm.npz + data/lstm_label2idx.json")
```

---

## `<ARQUITECTURA_DL>` — Módulos del Sistema

### MÓDULO 1 — CAPTURA Y DETECCIÓN
- **OpenCV:** `cv2.VideoCapture(0)` para webcam o lectura de video
- **MediaPipe Holistic:** pose(33 pts), left_hand(21 pts), right_hand(21 pts) → **75 kp × 2 coords = 150 dims/frame**
- **Normalización:** coordenadas ya normalizadas por MediaPipe al espacio [0,1] de la imagen — invarianza posicional incluida
- **Nota:** YOLOv8-pose es opcional; MediaPipe Holistic cubre detección + extracción en un solo paso

### MÓDULO 2 — CLASIFICACIÓN DE SEÑAS (LSPLSTMBidir)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attn = nn.Linear(hidden * 2, 1)

    def forward(self, lstm_out):  # lstm_out: [B, T, hidden*2]
        scores = self.attn(lstm_out)              # [B, T, 1]
        weights = F.softmax(scores, dim=1)        # [B, T, 1]
        return (lstm_out * weights).sum(dim=1)    # [B, hidden*2]

class LSPLSTMBidir(nn.Module):
    """
    Arquitectura Sprint 9. ~10M parámetros.
    Input:  [B, T=30, 150]
    Output: [B, n_classes] — logits
    """
    def __init__(self, n_dims=150, n_classes=482, hidden=256, n_layers=2, dropout=0.35):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_dims, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.lstm = nn.LSTM(
            128, hidden, n_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.attn = TemporalAttention(hidden)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):              # x: [B, 30, 150]
        x = self.proj(x)               # [B, 30, 128]
        out, _ = self.lstm(x)          # [B, 30, 512]
        ctx = self.attn(out)           # [B, 512]
        return self.head(ctx)          # [B, n_classes]

# ── Alternativas de arquitectura a comparar ───────────────────────────────────
# 2B-1. LSTM Bidir+Attn (ACTUAL — Sprint 9) — ~10M params — F1-val=0.0365
# 2B-2. MLP sobre media temporal (baseline tabular, Sprints 5-8) — <1M params
# 2B-3. ST-GCN sobre grafo anatómico (opcional S10+) — ~3M params
# 2B-4. VideoMAE fine-tuneado (requiere GPU A100, S11+) — ~86M params
```

**Modelos a comparar (tabla de referencia):**

| Modelo | Input | Params | F1-val actual | Latencia |
|--------|-------|--------|:-------------:|:--------:|
| LogReg (S5 baseline) | 108 dims (media) | <1K | 0.0068 | 0.1 ms |
| RF Bayesian HPO (S8) | 108 dims (media) | — | 0.0045 | 0.02 ms |
| **LSTM Bidir+Attn (S9)** | **150×30 dims** | **~10M** | **0.0365** | **<50 ms** |
| ST-GCN (S10 propuesta) | grafo 75 kp | ~3M | pendiente | ~30 ms |

### MÓDULO 3 — POST-PROCESAMIENTO LINGÜÍSTICO

```python
from collections import deque
import numpy as np

class PostProcesadorLSP:
    """
    Buffer de confianza deslizante para evitar parpadeo de texto.
    Aplica suavizado temporal sobre las predicciones frame a frame.
    """
    def __init__(self, idx2label, window=5, conf_umbral=0.30):
        self.idx2label   = idx2label
        self.window      = window
        self.conf_umbral = conf_umbral
        self.buffer_preds = deque(maxlen=window)
        self.historial    = deque(maxlen=20)

    def update(self, logits):
        probs = softmax(logits)
        conf  = float(probs.max())
        pred_idx = int(probs.argmax())
        pred_lbl = self.idx2label.get(str(pred_idx), "???")

        if conf >= self.conf_umbral:
            self.buffer_preds.append(pred_lbl)

        # Votación mayoritaria sobre la ventana
        if self.buffer_preds:
            from collections import Counter
            seña = Counter(self.buffer_preds).most_common(1)[0][0]
            if not self.historial or self.historial[-1] != seña:
                self.historial.append(seña)
            return seña, conf
        return None, conf

    def texto_actual(self):
        return " ".join(list(self.historial)[-10:])

def softmax(x):
    x = np.array(x) - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

### MÓDULO 4 — INTERFAZ EN TIEMPO REAL (Gradio + React)

Dos niveles de implementación:

**Nivel 1 — Gradio (implementado Sprint 9):** demo funcional, sin React, solo Python
**Nivel 2 — React + FastAPI + WebSocket (Sprint 10+):** UI completa con overlay Canvas

---

## `<ENTRENAMIENTO>` — Configuración completa Sprint 9

### Hiperparámetros fijos (config.yaml + train_lstm_signs.py)

```yaml
# configs/config.yaml — Sprint 9
SEED:         42
N_FRAMES:     30
N_DIMS:       150   # pose(33×2) + left_hand(21×2) + right_hand(21×2)
HIDDEN:       256
N_LAYERS:     2
DROPOUT:      0.35
BATCH:        64
LR:           1.0e-3
WEIGHT_DECAY: 1.0e-4
N_EPOCHS:     80
PATIENCE:     12
DEVICE:       mps   # macOS; usar cuda en Colab/T4

# Optimizer: AdamW(lr=1e-3, weight_decay=1e-4)
# Scheduler: CosineAnnealingLR(T_max=80)
# Loss: CrossEntropyLoss(weight=class_weights_inversas)
# Sampler: WeightedRandomSampler (balanceo por clase en train)
```

### Split estratificado (Sprint 9)

```python
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
for train_idx, temp_idx in sss.split(X, y):
    pass
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
for val_idx, test_idx in sss2.split(X[temp_idx], y[temp_idx]):
    pass
# → 70% train / 15% val / 15% test
```

> **DEUDA S10:** Migrar a `GroupKFold(n_splits=5, groups=viñeta_id)` para las muestras PKL de viñetas, evitando data leakage entre frames del mismo video.

### Aumentación de datos (señas LSP)

```python
import numpy as np

def augment_sequence(seq, noise_sigma=0.008, flip_prob=0.5):
    """
    Aumentación para secuencias de keypoints LSP.
    seq: [T, 150] — pose(33×2) + left_hand(21×2) + right_hand(21×2)
    """
    seq = seq.copy()

    # 1. Ruido gaussiano sobre coordenadas
    seq += np.random.normal(0, noise_sigma, seq.shape).astype(np.float32)

    # 2. Flip horizontal (intercambia mano izquierda ↔ derecha, invierte x)
    if np.random.random() < flip_prob:
        # Coordenadas x están en índices [0:33, 33:54, 54:75] (x de pose, lhand, rhand)
        # y en [75:108, 108:129, 129:150]
        pose_x  = seq[:, :33].copy()
        lhand_x = seq[:, 33:54].copy()
        rhand_x = seq[:, 54:75].copy()
        pose_y  = seq[:, 75:108].copy()
        lhand_y = seq[:, 108:129].copy()
        rhand_y = seq[:, 129:150].copy()

        seq[:, :33]    = 1.0 - pose_x   # invertir x pose
        seq[:, 33:54]  = 1.0 - rhand_x  # mano derecha → posición izquierda
        seq[:, 54:75]  = 1.0 - lhand_x  # mano izquierda → posición derecha
        seq[:, 108:129] = rhand_y        # intercambiar y también
        seq[:, 129:150] = lhand_y

    return seq

# Otras aumentaciones para Sprint 10:
# - Variación de velocidad: ×0.75 y ×1.25 (temporal resampling)
# - Mixup de secuencias (α=0.2, entre pares de la misma o diferente clase)
# - Label smoothing: CrossEntropyLoss(label_smoothing=0.1)
```

### Loop de entrenamiento completo

```python
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score
import numpy as np

def entrenar_lstm(X_train, y_train, X_val, y_val, n_classes, config):
    device = config["device"]

    # WeightedRandomSampler para balanceo de clases
    counts = np.bincount(y_train)
    weights = 1.0 / counts[y_train]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    train_dl = DataLoader(train_ds, batch_size=config["batch"], sampler=sampler)
    val_dl   = DataLoader(val_ds,   batch_size=256, shuffle=False)

    model = LSPLSTMBidir(
        n_dims=config["n_dims"],
        n_classes=n_classes,
        hidden=config["hidden"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
    ).to(device)

    class_weights = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=config["lr"],
                                   weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["n_epochs"])

    best_f1, best_state, patience_cnt = 0.0, None, 0
    history = {"f1_tr": [], "f1_val": [], "loss_tr": [], "loss_val": []}

    for epoch in range(1, config["n_epochs"] + 1):
        model.train()
        losses = []
        for Xb, yb in train_dl:
            Xb = augment_sequence_batch(Xb.numpy())  # aumentación
            Xb = torch.tensor(Xb).to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        # Evaluación
        model.eval()
        with torch.no_grad():
            y_pred_tr = model(torch.tensor(X_train).to(device)).argmax(dim=1).cpu()
            y_pred_val = model(torch.tensor(X_val).to(device)).argmax(dim=1).cpu()

        f1_tr  = f1_score(y_train, y_pred_tr,  average="macro", zero_division=0)
        f1_val = f1_score(y_val,   y_pred_val, average="macro", zero_division=0)
        history["f1_tr"].append(f1_tr)
        history["f1_val"].append(f1_val)
        history["loss_tr"].append(np.mean(losses))

        print(f"Época {epoch:3d} | loss={np.mean(losses):.4f} | F1-tr={f1_tr:.4f} | F1-val={f1_val:.4f}")

        if f1_val > best_f1:
            best_f1 = f1_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
            print(f"  ✅ Nuevo mejor F1-val = {best_f1:.4f}")
        else:
            patience_cnt += 1
            if patience_cnt >= config["patience"]:
                print(f"  🛑 Early stopping en época {epoch} (patience={config['patience']})")
                break

    # Guardar checkpoint
    model.load_state_dict(best_state)
    torch.save({
        "model_state": best_state,
        "label2idx":   config["label2idx"],
        "idx2label":   {v: k for k, v in config["label2idx"].items()},
        "n_classes":   n_classes,
        "n_dims":      config["n_dims"],
        "n_frames":    config["n_frames"],
        "hidden":      config["hidden"],
        "n_layers":    config["n_layers"],
        "f1_val":      best_f1,
        "best_epoch":  epoch - patience_cnt,
        "history":     history,
    }, "checkpoints/lstm_signs.pt")
    print(f"\nGuardado: checkpoints/lstm_signs.pt | F1-val={best_f1:.4f}")
    return model, history
```

---

## `<VALIDACION_ESTRATEGIA>` — GroupKFold vs StratifiedShuffleSplit

```python
# Sprint 5-8 (RF, LogReg): GroupKFold(5) — correcto para evitar leakage de video
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=viñeta_id)):
    # viñeta_id: identificador del video del que proviene cada sample
    # Evita que frames del mismo video estén en train Y val
    pass

# Sprint 9 (LSTM): StratifiedShuffleSplit — más simple, pero con riesgo de leakage
# para muestras de viñetas (múltiples frames del mismo video)
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)

# SPRINT 10 — PLAN DE MIGRACIÓN:
# Usar GroupKFold también para LSTM con grupos por viñeta_id
# Las muestras del Abecedario (fotos independientes) pueden usar StratifiedSplit
# Las muestras de PKL viñetas y Glosas deben usar GroupKFold
viñeta_groups = np.array([p.stem.rsplit("_", 1)[0] for p in pkl_paths])
gkf_lstm = GroupKFold(n_splits=5)
for fold, (tr, val) in enumerate(gkf_lstm.split(X, y, groups=viñeta_groups)):
    pass
```

---

## `<INFERENCIA_TIEMPO_REAL>` — ONNX + MediaPipe

```python
"""
Inferencia completa: frame → landmarks → modelo → predicción
Latencia objetivo: <50ms total (MediaPipe + ONNX)
"""
import cv2, json, numpy as np
import mediapipe as mp
import onnxruntime as ort
from collections import deque

N_FRAMES = 30
N_DIMS   = 150
CONF_UMBRAL = 0.30

# ── Cargar modelo ONNX ────────────────────────────────────────────────────────
sess = ort.InferenceSession(
    "checkpoints/lstm_signs.onnx",
    providers=["CoreMLExecutionProvider", "CPUExecutionProvider"]
)
with open("data/lstm_label2idx.json") as f:
    label2idx = json.load(f)
idx2label = {v: k for k, v in label2idx.items()}

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4,
)

kp_buffer = deque(maxlen=N_FRAMES)

def extraer_kp_frame(frame):
    """Frame BGR → vector [150] de keypoints LSP"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = holistic.process(rgb)
    row = []
    for lm_list, n in [
        (res.pose_landmarks,       33),
        (res.left_hand_landmarks,  21),
        (res.right_hand_landmarks, 21),
    ]:
        if lm_list:
            xs = [l.x for l in lm_list.landmark[:n]]
            ys = [l.y for l in lm_list.landmark[:n]]
        else:
            xs, ys = [0.0]*n, [0.0]*n
        row.extend(xs); row.extend(ys)
    return np.array(row, dtype=np.float32)  # [150]

def predecir_seña(frame):
    """
    Procesa UN frame y retorna (seña, confianza) cuando el buffer está lleno.
    Llamar en cada frame del loop de cámara.
    """
    kp = extraer_kp_frame(frame)
    kp_buffer.append(kp)

    if len(kp_buffer) < N_FRAMES:
        return None, 0.0

    seq = np.array(list(kp_buffer), dtype=np.float32)[None]  # [1, 30, 150]
    logits = sess.run(None, {"input": seq})[0][0]              # [n_classes]
    probs  = np.exp(logits - logits.max())
    probs /= probs.sum()
    conf = float(probs.max())
    pred = idx2label.get(int(probs.argmax()), "???")

    if conf >= CONF_UMBRAL:
        return pred, conf
    return None, conf

# ── Loop de cámara ────────────────────────────────────────────────────────────
def loop_camara():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break

        seña, conf = predecir_seña(frame)

        # Overlay de landmarks + texto
        if seña:
            cv2.putText(frame, f"{seña} ({conf:.0%})",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 2)

        cv2.imshow("LSP → Castellano", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
```

---

## `<DEPLOY_GRADIO>` — Demo funcional Sprint 9

```python
"""
demo/app_gradio.py — Interfaz Gradio para traducción LSP en tiempo real.
Lanzar: python demo/app_gradio.py --share
URL pública: https://XXXX.gradio.live (activa 7 días)
Deploy permanente: huggingface-cli login + python scripts/deploy_huggingface.py
"""
import gradio as gr
import numpy as np

def traducir_video(frame_bgr):
    """Función principal de Gradio: frame → texto + frame anotado"""
    seña, conf = predecir_seña(frame_bgr)
    # Dibujar landmarks con MediaPipe Drawing
    # Retornar frame anotado + texto traducido
    return frame_anotado, texto

with gr.Blocks(title="Traductor LSP — Señas Peruanas") as demo:
    gr.Markdown("# 🤟 Traductor LSP → Castellano")
    with gr.Row():
        cam    = gr.Image(source="webcam", streaming=True, label="Cámara")
        output = gr.Image(label="Señas detectadas")
    texto  = gr.Textbox(label="Traducción en castellano", lines=3)
    historial_box = gr.Textbox(label="Historial (últimas 10 señas)", lines=5)
    cam.stream(traducir_video, inputs=[cam], outputs=[output, texto])

demo.launch(share=True)  # --share genera URL pública

# ── Deploy en HuggingFace Spaces ─────────────────────────────────────────────
# spaces/
# ├── app.py              (copia de app_gradio.py adaptada)
# ├── lstm_signs.onnx     (modelo principal)
# ├── lstm_label2idx.json (mapeo clases)
# ├── requirements.txt    (gradio>=4.0, onnxruntime, mediapipe, opencv-python-headless)
# └── README.md           (metadata HF: title, emoji, sdk=gradio)
#
# Subir:
# huggingface-cli login
# huggingface-cli repo create traductor-lsp --type space --sdk gradio
# cd spaces && git push
```

---

## `<MLOPS_LIGERO>` — Trazabilidad y Reproducibilidad

### Estandarización de experimentos

```
[✅] 1. Semilla fija:  SEED=42 en np/torch/random/sklearn en todos los scripts
[✅] 2. Validación coherente: GroupKFold(5) para RF | StratifiedSplit para LSTM (→ GroupKFold en S10)
[✅] 3. Pipeline cerrado: fit SOLO en train (LayerNorm interna al modelo, no StandardScaler externo)
[✅] 4. config.yaml centralizado: datos, features, HP, métrica, split
[✅] 5. Naming convention: exp_{fecha}_{modelo}_{features}_{split}
```

### Tablero de corridas — `logs/runs.csv`

| exp_id | Sprint | Modelo | F1-val (mean±std) | Tiempo (s) | Lat (ms) | Nota |
|--------|--------|--------|:-----------------:|:----------:|:--------:|------|
| exp_20260410_lr_108dims_gkfold | S5 | LogReg | 0.0068±0.0012 | 18 | 0.1 | Mejor baseline lineal |
| exp_20260515_rf_bay50_108dims | S8 | RF-Bay50 | 0.0045±0.0000 | 520 | 0.02 | Mejor árbol (std=0) |
| exp_20260601_lstm_150dims_strat | S9 | LSTM-Bidir | **0.0365** | 7200 | 48 | **BEST — deploy ONNX** |

### MLflow tracking (listo para Sprint 10)

```python
import mlflow

mlflow.set_experiment("TRADUCTOR_LSP")

with mlflow.start_run(run_name="exp_20260601_lstm_150dims_strat"):
    mlflow.log_params({
        "modelo": "LSPLSTMBidir", "hidden": 256, "n_layers": 2,
        "dropout": 0.35, "lr": 1e-3, "seed": 42,
        "split": "StratifiedShuffleSplit(70/15/15)",
        "features": "150dims_pose+rhand+lhand_30frames",
        "n_classes": 482,
    })
    mlflow.log_metrics({
        "F1_val": 0.0365, "F1_test": 0.0109, "Acc_test": 0.0262,
        "gap_overfitting": 0.95, "ECE": 0.427,
        "best_epoch": 27, "latencia_ms": 48.0,
    })
    mlflow.log_artifacts("figs/")
    mlflow.log_artifact("logs/runs.csv")
    mlflow.log_artifact("checkpoints/lstm_signs.onnx")
# Lanzar UI: mlflow ui → http://localhost:5000
```

---

## `<INTERFAZ_UI>` — Especificación completa

### Nivel 1 — Gradio (Sprint 9, IMPLEMENTADO)

```
┌─────────────────────────────────────────────────────────┐
│  🤟 Traductor LSP → Castellano (Gradio)                 │
├──────────────────────┬──────────────────────────────────┤
│  📷 CÁMARA / VIDEO   │  📝 TRADUCCIÓN                   │
│                      │                                  │
│  [frame con skeleton │  "HOLA BUENOS DIAS"              │
│   MediaPipe]         │                                  │
│                      │  Confianza: 87%                  │
│  Seña: HOLA          │  Latencia: 43ms                  │
├──────────────────────┴──────────────────────────────────┤
│  Historial: HOLA | BUENOS DIAS | GRACIAS | ...          │
└─────────────────────────────────────────────────────────┘
```

### Nivel 2 — React + FastAPI + WebSocket (Sprint 10+)

```
┌─────────────────────────────────────────────────────────────┐
│  🤟 Sistema LSP → Castellano | EN VIVO ● [CÁMARA] [VIDEO]  │
├────────────────────────────┬────────────────────────────────┤
│  📷 CÁMARA — SEÑAS LSP     │  📝 TRADUCCIÓN EN CASTELLANO   │
│                            │                                │
│  [Canvas overlay con       │  "Hola, buenos días.           │
│   skeleton de manos        │   ¿Cómo estás hoy?"           │
│   en tiempo real]          │                                │
│                            │  Confianza: ████████░░ 87%     │
│  🟢 Manos detectadas       │  ● 24 fps  ● 43ms latencia    │
│  Seña: BUENOS_DIAS         │                                │
├────────────────────────────┴────────────────────────────────┤
│  📜 HISTORIAL: "Hola" | "Me llamo..." | "Necesito ayuda"   │
│  [🔊 Leer en voz alta] [📋 Copiar texto] [💾 Exportar TXT] │
└─────────────────────────────────────────────────────────────┘
```

**Componentes obligatorios:**

**Panel izquierdo — Cámara:**
- Video en vivo con overlay Canvas de skeleton de manos (75 keypoints)
- Bounding box alrededor de cada mano + nombre de seña actual
- Indicador: 🟢 Manos detectadas / 🔴 Sin detección / 🟡 Procesando

**Panel derecho — Texto:**
- Texto traducido en fuente ≥18px, actualización suave (buffer deslizante)
- Barra de confianza: verde >75% | amarillo 50-75% | rojo <50%
- Historial de últimas 10 frases, scrolleable
- TTS: Web Speech API nativa del navegador (sin costo, funciona offline)

**Controles:**
- `[ Cámara en vivo ]` → webcam activa, traducción continua ≥24 fps
- `[ Subir video ]` → procesar .mp4, transcripción con timestamps
- `[ Modo Educativo ]` → nombre de seña + imagen de referencia
- `[ Pausar ]` | `[ Limpiar historial ]`

**Implementación técnica:**
```python
# api/main.py — FastAPI + WebSocket
from fastapi import FastAPI, WebSocket
import asyncio, cv2, numpy as np, json

app = FastAPI()

@app.websocket("/ws/traducir")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_bytes()
        frame = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        seña, conf = predecir_seña(frame)
        await websocket.send_json({"seña": seña, "confianza": conf})
```

---

## `<METRICAS_EVALUACION>` — Objetivos y resultados

| Métrica | Objetivo | S8 RF (mejor árbol) | S9 LSTM (actual) | S10 (pendiente) |
|---------|---------|:-------------------:|:----------------:|:---------------:|
| F1-macro val | > 0.70 | 0.0045 | **0.0365** | > 0.10 |
| F1-macro test | > 0.65 | 0.0040 | 0.0109 | > 0.08 |
| Accuracy test | > 70% | ~4.9% | 2.62% | > 10% |
| WER (frases SRT) | < 0.30 | — | — | pendiente |
| BLEU Score | > 0.40 | — | — | pendiente |
| Latencia inferencia | < 200ms | 0.02ms | **<50ms** | <50ms |
| FPS interfaz | ≥ 24fps | N/A | ~15fps Gradio | ≥ 24fps React |
| ECE calibración | < 0.10 | 0.312 | 0.427 (alto) | < 0.15 |

**Métricas adicionales (implementar S10):**
- Matriz de confusión normalizada por grupos de señas similares
- Top-5 accuracy (relevante con 482+ clases)
- Tasa de detección de manos (% frames con manos correctamente detectadas)

---

## `<ROADMAP_SPRINTS>` — Plan de desarrollo

| Sprint | Objetivo | Estado | Métrica clave |
|--------|----------|--------|:-------------:|
| S5 | Baseline ML: LogReg + SVC sobre keypoints temporales | ✅ | F1=0.0068 |
| S6 | RF/ET default: exploración de árboles | ✅ | F1=0.0040 |
| S7 | HPO 30 trials Optuna TPE: RF+ET | ✅ | F1=0.0041 |
| S8 | HPO Bayesian 50 trials + RF final | ✅ | F1=0.0045 |
| **S9** | **LSTM Bidir+Attn + dataset combinado + ONNX deploy** | **✅** | **F1=0.0365** |
| S10 | GroupKFold LSTM + HPO LSTM + label smoothing + más datos | 🔄 | F1>0.10 |
| S11 | ST-GCN sobre grafo anatómico | ⏳ | F1>0.20 |
| S12 | WER/BLEU sobre SRT.tar + integración NLP | ⏳ | WER<0.50 |
| S13 | UI React + FastAPI WebSocket + deploy permanente | ⏳ | 24fps real |

---

## `<STACK_TECNOLOGICO>`

| Categoría | Herramientas | Estado |
|-----------|-------------|--------|
| Captura de video | OpenCV (cv2) | ✅ Implementado |
| Extracción landmarks | MediaPipe Holistic (75 kp × 2 = 150 dims) | ✅ |
| Deep Learning | PyTorch 2.2.2 | ✅ |
| Modelo principal | LSPLSTMBidir (~10M params, BiLSTM×2 + Attention) | ✅ S9 |
| Export deploy | ONNX opset 17, onnxruntime 1.x | ✅ |
| Modelos árbol | sklearn RF/ET + Optuna TPE | ✅ S5–S8 |
| NLP post-proceso | Buffer deslizante + votación mayoritaria | ✅ básico |
| NLP avanzado | BERT español (HuggingFace) corrección SOV→SVO | ⏳ S12 |
| Backend API | FastAPI + WebSocket | ⏳ S13 |
| Frontend | Gradio (actual) → React + TailwindCSS + Canvas | 🔄 |
| Experiment tracking | logs/runs.csv (ligero) → MLflow | 🔄 |
| Deploy | HuggingFace Spaces + Gradio | ✅ preparado |
| MLOps | config.yaml + Git LFS + logs/runs.csv | ✅ |
| Entorno | `.venv310` (PyTorch), `.venv-1` (análisis/figuras) | ✅ |
| **Latencia actual** | **<50ms ONNX** | **✅** |
| **Latencia objetivo** | **<200ms end-to-end** | **✅** |

---

## `<ENTREGABLES_CURSO>`

| # | Entregable | Descripción | Estado |
|---|------------|-------------|--------|
| 1 | `README.md` | Descripción problema, dataset, instalación, resultados, estructura | ✅ |
| 2 | `notebooks/01_eda_lsp.ipynb` | EDA: 3 TAR/PKL, distribución clases, leakage, desbalance | ✅ |
| 3 | `scripts/build_combined_dataset.py` | Pipeline unificado 3 fuentes PKL → NPZ | ✅ |
| 4 | `notebooks/05_Semana5_*.ipynb` | Baseline: LogReg + SVC sobre keypoints, F1-macro | ✅ |
| 5 | `notebooks/07_Semana7.ipynb` + `08_Semana8.ipynb` | RF HPO Optuna 30/50 trials, tabla comparativa | ✅ |
| 6 | `scripts/train_lstm_signs.py` | LSPLSTMBidir: training + early stopping + ONNX export | ✅ S9 |
| 7 | `demo/app_gradio.py` | Demo cámara + overlay MediaPipe + traducción ONNX | ✅ S9 |
| 8 | `checkpoints/lstm_signs.onnx` | Modelo ONNX <50ms, opset 17, para producción | ✅ S9 |
| 9 | `logs/runs.csv` | Tablero ligero: 8 corridas S5–S9, exp_id/métricas/HP | ✅ S9 |
| 10 | `PARCIAL_SEMANA9.md` + `.html` + `.docx` | Informe parcial completo con MLOps avanzado | ✅ S9 |
| 11 | `api/main.py` | FastAPI + WebSocket, inferencia tiempo real | ⏳ S13 |
| 12 | `frontend/` | React + Canvas overlay + TTS + exportar | ⏳ S13 |

---

## `<INSTRUCCION_FINAL>` — Orden de ejecución desde cero

### Si continúas desde Sprint 9 (estado actual):

```bash
# 1. Verificar entorno
python --version                         # 3.10+
pip install torch torchvision mediapipe onnxruntime gradio opencv-python

# 2. Construir dataset combinado (si no existe data/dataset_lstm.npz)
python scripts/build_combined_dataset.py

# 3. Entrenar LSTM (si no existe checkpoints/lstm_signs.pt)
python scripts/train_lstm_signs.py

# 4. Lanzar demo
python demo/app_gradio.py --share

# 5. Ver tablero MLOps
cat logs/runs.csv
```

### Si empiezas desde cero (orden completo):

1. **Montar Drive y extraer los 3 TAR** → `Videos.tar`, `Keypoints.tar`, `SRT.tar`
2. **EDA:** explorar Keypoints PKL (estructura, clases, distribución), verificar MD5
3. **Build dataset:** `build_combined_dataset.py` → `data/dataset_lstm.npz` [N, 30, 150]
4. **Entrenar baseline** (LogReg sobre media temporal, F1-macro)
5. **Entrenar LSTM Bidir** → `checkpoints/lstm_signs.pt` + `lstm_signs.onnx`
6. **Lanzar demo Gradio** → `demo/app_gradio.py --share`
7. **Registrar corrida** → `logs/runs.csv` con exp_id/métricas/HP
8. **Deploy HF Spaces** → `huggingface-cli login` + push `spaces/`

### Sprint 10 — Próximos pasos prioritarios:

```python
# PRIORIDAD 1: Corregir sesgo de validación
# Migrar a GroupKFold(5) para muestras de viñetas PKL
gkf = GroupKFold(n_splits=5)
viñeta_groups = [p.stem.rsplit("_", 1)[0] for p in pkl_paths]

# PRIORIDAD 2: Reducir overfitting (gap actual 95%)
# Opción A: Reducir capacidad
model = LSPLSTMBidir(hidden=128, n_layers=1, dropout=0.50)
# Opción B: Label smoothing
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

# PRIORIDAD 3: HPO formal LSTM con Optuna
import optuna
def objective(trial):
    hidden  = trial.suggest_categorical("hidden",  [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.2, 0.6)
    lr      = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    # ... entrenar y retornar F1-val

study = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=30)
```

---

## `<EDA_COMPLETO>` — Análisis Exploratorio de Datos

### EDA de los 3 archivos TAR + PKL

```python
"""
01_eda_lsp.ipynb — Análisis exploratorio completo
Celdas independientes, ejecutar en orden.
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib
import pickle, json, os, glob, re
from pathlib import Path
from collections import Counter

matplotlib.rcParams.update({"figure.dpi": 120, "font.size": 11})

# ── 1. EXPLORAR ESTRUCTURA PKL ────────────────────────────────────────────────
KEYPOINTS_DIR = Path("data/Keypoints")

fuentes_info = {}
for subdir in ["pkl", "glosas_pkl", "abecedario_pkl"]:
    pkls = sorted((KEYPOINTS_DIR / subdir).glob("**/*.pkl"))
    labels = []
    longitudes = []
    for p in pkls[:50]:  # muestra rápida de 50
        try:
            with open(p, "rb") as f:
                frames = pickle.load(f)
            longitudes.append(len(frames))
            stem = p.stem
            m = re.match(r"^(.+)_(\d+)$", stem)
            labels.append(m.group(1).upper() if m else stem.upper())
        except:
            pass
    fuentes_info[subdir] = {
        "n_archivos": len(pkls),
        "n_clases":   len(set(labels)),
        "long_media": np.mean(longitudes) if longitudes else 0,
        "long_min":   np.min(longitudes)  if longitudes else 0,
        "long_max":   np.max(longitudes)  if longitudes else 0,
    }

print("─" * 60)
print(f"{'Fuente':<20} {'Muestras':>8} {'Clases':>7} {'Long.med':>9} {'min':>5} {'max':>5}")
print("─" * 60)
for nombre, info in fuentes_info.items():
    print(f"{nombre:<20} {info['n_archivos']:>8} {info['n_clases']:>7} "
          f"{info['long_media']:>9.1f} {info['long_min']:>5} {info['long_max']:>5}")
print("─" * 60)

# ── 2. DISTRIBUCIÓN DE CLASES ─────────────────────────────────────────────────
all_labels = []
for subdir in ["pkl", "glosas_pkl", "abecedario_pkl"]:
    for p in (KEYPOINTS_DIR / subdir).glob("**/*.pkl"):
        stem = p.stem
        m = re.match(r"^(.+)_(\d+)$", stem)
        all_labels.append(m.group(1).upper() if m else stem.upper())

counter = Counter(all_labels)
counts  = np.array(sorted(counter.values(), reverse=True))

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# 2a. Distribución de frecuencias (log scale)
axes[0].hist(counts, bins=30, color="#2563eb", alpha=0.7, edgecolor="white")
axes[0].set_yscale("log")
axes[0].set_xlabel("Muestras por clase")
axes[0].set_ylabel("Número de clases (log)")
axes[0].set_title("Distribución de clases (todas las fuentes)")
axes[0].axvline(x=20, color="red", linestyle="--", label="min. recomendado (20)")
axes[0].legend()

# 2b. Top-20 clases más frecuentes
top20 = counter.most_common(20)
axes[1].barh([t[0] for t in top20[::-1]], [t[1] for t in top20[::-1]],
             color="#16a34a", alpha=0.8)
axes[1].set_xlabel("Número de muestras")
axes[1].set_title("Top-20 clases más representadas")

# 2c. Curva de Pareto: % muestras acumuladas
sorted_counts = np.array(sorted(counter.values(), reverse=True))
cumsum = np.cumsum(sorted_counts) / sorted_counts.sum() * 100
axes[2].plot(range(1, len(cumsum)+1), cumsum, color="#dc2626", linewidth=2)
axes[2].axhline(y=80, color="gray", linestyle="--", alpha=0.5, label="80% muestras")
axes[2].set_xlabel("Número de clases (ordenadas por frecuencia)")
axes[2].set_ylabel("% muestras acumuladas")
axes[2].set_title("Curva de Pareto: concentración de datos")
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figs/eda_01_distribucion_clases.png", bbox_inches="tight")
plt.show()

# ── 3. ANÁLISIS DE KEYPOINTS (detectar manos faltantes) ──────────────────────
manos_detectadas = {"pose": 0, "left_hand": 0, "right_hand": 0}
manos_ausentes   = {"pose": 0, "left_hand": 0, "right_hand": 0}
sample_pkls = sorted((KEYPOINTS_DIR / "pkl").glob("**/*.pkl"))[:200]

for p in sample_pkls:
    try:
        with open(p, "rb") as f:
            frames = pickle.load(f)
        for fr in frames:
            for key in ["pose", "left_hand", "right_hand"]:
                kp = fr.get(key, {})
                xs = kp.get("x", [])
                if xs and any(v != 0.0 for v in xs):
                    manos_detectadas[key] += 1
                else:
                    manos_ausentes[key] += 1
    except:
        pass

print("\n─ Detección de keypoints (muestra 200 PKL) ─")
for key in ["pose", "left_hand", "right_hand"]:
    total = manos_detectadas[key] + manos_ausentes[key]
    if total > 0:
        tasa = manos_detectadas[key] / total * 100
        print(f"  {key:15}: {tasa:5.1f}% detectados ({manos_ausentes[key]} frames con ceros)")

# ── 4. RIESGOS IDENTIFICADOS ──────────────────────────────────────────────────
print("\n─ CHECKLIST DE RIESGOS ─")

# Desbalance extremo
clases_bajo_umbral = sum(1 for c in counter.values() if c < 5)
print(f"  Desbalance: {clases_bajo_umbral}/{len(counter)} clases con < 5 muestras "
      f"({'CRÍTICO' if clases_bajo_umbral/len(counter) > 0.5 else 'MODERADO'})")

# Clases solitarias
clases_singleton = sum(1 for c in counter.values() if c == 1)
print(f"  Singletons: {clases_singleton} clases con exactamente 1 muestra → filtrar")

# Leakage potencial
print(f"  Leakage: PKL viñetas tienen múltiples frames del mismo video → usar GroupKFold")

# Distribución de longitudes
pkls_muy_cortos = sum(1 for p in sample_pkls
                      if len(pickle.load(open(p,"rb"))) < 5)  # type: ignore
print(f"  PKL cortos: {pkls_muy_cortos}/200 tienen < 5 frames → revisar extracción")
```

### EDA de Videos.tar (EDA visual)

```python
import cv2, mediapipe as mp
from pathlib import Path

VIDEO_DIR = Path("/content/lsp_dataset/Videos/")

def analizar_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur   = total / fps if fps > 0 else 0
    cap.release()
    return {"fps": fps, "frames": total, "width": w, "height": h, "duracion_s": dur}

videos = list(VIDEO_DIR.glob("**/*.mp4"))[:100]
stats  = [analizar_video(v) for v in videos]
df     = pd.DataFrame(stats)

print(df.describe().round(2))
print(f"\nDuración media: {df['duracion_s'].mean():.2f}s | "
      f"Max: {df['duracion_s'].max():.2f}s | Min: {df['duracion_s'].min():.2f}s")
print(f"FPS: {df['fps'].mode()[0]:.0f}fps más común")
print(f"Resolución dominante: {df['width'].mode()[0]}×{df['height'].mode()[0]}")

# Riesgo: videos muy cortos
videos_cortos = (df["duracion_s"] < 0.5).sum()
print(f"⚠️  Videos < 0.5s: {videos_cortos} ({videos_cortos/len(df)*100:.1f}%)")
```

### EDA de SRT.tar (análisis de ground truth)

```python
import glob, re

SRT_DIR = "/content/lsp_dataset/SRT/"
srt_files = glob.glob(os.path.join(SRT_DIR, "**/*.srt"), recursive=True)

vocabulario = Counter()
longitudes_frases = []

for srt_path in srt_files:
    with open(srt_path, "r", encoding="utf-8", errors="replace") as f:
        contenido = f.read()
    # Extraer solo texto (sin timestamps y numeración)
    textos = re.findall(r"\d{2}:\d{2}:\d{2}.*?-->.+?\n([\s\S]*?)(?=\n\n|\Z)", contenido)
    for t in textos:
        palabras = t.strip().lower().split()
        vocabulario.update(palabras)
        longitudes_frases.append(len(palabras))

print(f"Total SRT: {len(srt_files)}")
print(f"Vocabulario único: {len(vocabulario)} palabras")
print(f"Top-20 palabras: {vocabulario.most_common(20)}")
print(f"Longitud media frase: {np.mean(longitudes_frases):.1f} palabras")
print(f"Frases de 1 palabra: {sum(1 for l in longitudes_frases if l==1)} "
      f"(señas aisladas a clasificar directamente)")
```

---

## `<FUENTES_COMPLEMENTARIAS>` — Datos adicionales opcionales

> Solo si el dataset TAR no cubre ciertas señas o clases tienen < 5 muestras.

### Imágenes estáticas (dactilología + números)

```python
import cv2, mediapipe as mp, numpy as np
from pathlib import Path

# Estructura esperada: dataset/imagenes/[SEÑA]/img_001.jpg
IMG_DIR = Path("data/imagenes_lsp")

mp_hands = mp.solutions.hands

def imagen_a_landmarks(img_path):
    """JPG de seña estática → vector [63,] (21 kp × 3 coords: x,y,z)"""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2) as hands:
        result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        return None
    lm = result.multi_hand_landmarks[0]
    row = []
    for p in lm.landmark:
        row.extend([p.x, p.y, p.z])
    return np.array(row, dtype=np.float32)  # [63]

X_img, y_img = [], []
for clase_dir in sorted(IMG_DIR.iterdir()):
    if not clase_dir.is_dir(): continue
    for img_path in clase_dir.glob("*.jpg"):
        lm = imagen_a_landmarks(img_path)
        if lm is not None:
            # Para usar en LSTM: replicar el vector [63] → [30, 150] con padding cero
            # Nota: imágenes son señas estáticas → T=1, pero LSTM necesita T=30
            seq = np.zeros((30, 150), dtype=np.float32)
            # Mapear 63 dims → últimas 42 dims (right_hand: índices 108:150)
            seq[:, 108:150] = np.tile(lm[:42], (30, 1))
            X_img.append(seq)
            y_img.append(clase_dir.name.upper())

print(f"Imágenes procesadas: {len(X_img)} | Clases: {len(set(y_img))}")
```

### PDFs LSP — Diccionario de señas (fuente MINEDU/CPAL/FENAL)

```python
import pdfplumber, json, re
from pathlib import Path

PDF_DIR = Path("data/pdfs_lsp")

def extraer_glosario_pdf(pdf_path):
    """
    Extrae pares {seña: descripción} de PDFs de diccionarios LSP.
    Asume formato: SEÑA en mayúsculas seguida de descripción en párrafo.
    """
    glosario = {}
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            texto = page.extract_text() or ""
            for linea in texto.split("\n"):
                linea = linea.strip()
                if re.match(r"^[A-ZÁÉÍÓÚÑ\s]{3,}$", linea) and len(linea) > 2:
                    seña_actual = linea
                    glosario[seña_actual] = {"descripcion": "", "pagina": page.page_number}
    return glosario

glosario_lsp = {}
for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
    g = extraer_glosario_pdf(pdf_path)
    glosario_lsp.update(g)
    print(f"  {pdf_path.name}: {len(g)} señas extraídas")

# Guardar para usar como fallback semántico
with open("data/glosario_lsp.json", "w", encoding="utf-8") as f:
    json.dump(glosario_lsp, f, ensure_ascii=False, indent=2)
print(f"\nGlosario total: {len(glosario_lsp)} señas → data/glosario_lsp.json")

# Uso como fallback en la inferencia:
# if confianza < 0.30:
#     seña = buscar_en_glosario(prediccion_alternativa, glosario_lsp)
```

---

## `<SPRINT10_IMPLEMENTACION>` — Plan concreto y código

> Sprint 10 corrige las 3 deudas técnicas principales del Sprint 9.

### TAREA 1 — GroupKFold para LSTM (corregir sesgo val/test)

```python
"""
train_lstm_s10_groupkfold.py
Reemplaza StratifiedShuffleSplit por GroupKFold para muestras de viñetas.
Resultado esperado: reducir brecha F1-val/F1-test (actualmente 3.4×).
"""
import numpy as np, torch, json
from pathlib import Path
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
from sklearn.metrics import f1_score

# Cargar dataset
data = np.load("data/dataset_lstm.npz")
X, y = data["X"], data["y"]
with open("data/lstm_label2idx.json") as f:
    label2idx = json.load(f)
with open("data/lstm_sources.json") as f:    # ← nuevo: metadatos de origen
    sources = json.load(f)   # {filename: "pkl"|"glosas"|"abecedario"}

# Construir grupos (viñeta_id para muestras PKL y glosas; None para abecedario)
groups = []
viñeta_ids = []
for src_name in sources["filenames"]:
    src_type = sources["types"][src_name]
    if src_type in ("pkl", "glosas"):
        stem = Path(src_name).stem
        m = re.match(r"^(.+)_(\d+)$", stem)
        viñeta_ids.append(m.group(1) if m else stem)
    else:
        viñeta_ids.append(f"abc_{src_name}")   # abecedario: cada imagen es independiente
groups = np.array(viñeta_ids)

# Máscaras por tipo de fuente
mask_seq = np.array([t in ("pkl","glosas") for t in sources["types"].values()])
mask_abc = ~mask_seq

# ── Split Híbrido ─────────────────────────────────────────────────────────────
# 1. GroupKFold(5) para PKL + Glosas (para 5-fold CV)
gkf = GroupKFold(n_splits=5)
cv_results = []

for fold, (tr, val) in enumerate(gkf.split(X[mask_seq], y[mask_seq],
                                            groups=groups[mask_seq])):
    # Combinar con Abecedario en cada fold
    X_tr  = np.concatenate([X[mask_seq][tr],  X[mask_abc]])
    y_tr  = np.concatenate([y[mask_seq][tr],  y[mask_abc]])
    X_val = X[mask_seq][val]
    y_val = y[mask_seq][val]

    model, hist = entrenar_lstm(X_tr, y_tr, X_val, y_val, len(label2idx), CONFIG_S10)
    cv_results.append({"fold": fold, "f1_val": hist["f1_val_best"]})
    print(f"  Fold {fold}: F1-val = {hist['f1_val_best']:.4f}")

f1_media = np.mean([r["f1_val"] for r in cv_results])
f1_std   = np.std( [r["f1_val"] for r in cv_results])
print(f"\n✅ GroupKFold CV: F1 = {f1_media:.4f} ± {f1_std:.4f}")
```

### TAREA 2 — HPO LSTM con Optuna (30 trials TPE)

```python
"""
hpo_lstm_s10.py
Búsqueda de HP para LSPLSTMBidir. Objetivo: F1-val > 0.05 en 30 trials.
Naming: exp_20260615_lstm_hpo30_150dims_gkfold
"""
import optuna
from optuna.samplers import TPESampler

N_TRIALS  = 30
N_FOLD    = 3   # CV más rápido para HPO (3-fold)
LOG_FILE  = "logs/runs.csv"

def objective(trial):
    hidden   = trial.suggest_categorical("hidden",   [64, 128, 256])
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout  = trial.suggest_float("dropout",  0.20, 0.60, step=0.05)
    lr       = trial.suggest_float("lr",       1e-4, 5e-3, log=True)
    wd       = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    smooth   = trial.suggest_float("label_smoothing", 0.0, 0.15, step=0.05)

    config = {**CONFIG_BASE, "hidden": hidden, "n_layers": n_layers,
              "dropout": dropout, "lr": lr, "weight_decay": wd,
              "label_smoothing": smooth, "n_epochs": 40, "patience": 8}

    f1_vals = []
    for fold, (tr, val) in enumerate(gkf_3fold.split(X[mask_seq], y[mask_seq],
                                                       groups=groups[mask_seq])):
        if fold >= N_FOLD: break
        X_tr  = np.concatenate([X[mask_seq][tr], X[mask_abc]])
        y_tr  = np.concatenate([y[mask_seq][tr], y[mask_abc]])
        X_val = X[mask_seq][val]
        y_val = y[mask_seq][val]
        _, hist = entrenar_lstm(X_tr, y_tr, X_val, y_val, len(label2idx), config)
        f1_vals.append(hist["f1_val_best"])

        # Pruning de trials malos (Hyperband-style)
        trial.report(np.mean(f1_vals), fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return float(np.mean(f1_vals))

study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    study_name="LSTM_S10_HPO",
)
study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)

# Mostrar mejores resultados
best = study.best_trial
print(f"\n✅ Mejor trial: F1 = {best.value:.4f}")
print(f"   Params: {best.params}")

# Guardar en runs.csv
import csv, datetime
exp_id = f"exp_{datetime.date.today().strftime('%Y%m%d')}_lstm_hpo30_150dims_gkfold"
with open(LOG_FILE, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        exp_id, "S10", "LSTM-HPO30", "150dims×30fr",
        str(best.params), f"{best.value:.4f}", "",
        "", "", "", "", "GroupKFold(5)", "42", str(len(label2idx)),
        f"HPO Optuna 30 trials TPE | best: {best.params}"
    ])
print(f"Registrado en {LOG_FILE}: {exp_id}")
```

### TAREA 3 — Mitigación de overfitting (label smoothing + reducción capacidad)

```python
"""
Configuración S10 con mitigaciones anti-overfitting.
Aplica 3 técnicas simultáneas: label smoothing + dropout↑ + hidden↓
"""
CONFIG_S10 = {
    # Mismo seed y features
    "seed":       42,
    "n_frames":   30,
    "n_dims":     150,

    # Capacidad reducida (S9: hidden=256, n_layers=2)
    "hidden":     128,    # ↓ de 256 → reduce params de 10M a ~2.5M
    "n_layers":   2,
    "dropout":    0.50,   # ↑ de 0.35

    # Label smoothing (reduce sobreconfianza, ECE↓)
    "label_smoothing": 0.10,

    # Mismos optimizador y scheduler
    "lr":         1e-3,
    "weight_decay": 5e-4,   # ↑ ligeramente
    "batch":      64,
    "n_epochs":   80,
    "patience":   12,
    "device":     "mps",

    # Augmentación reforzada
    "noise_sigma": 0.015,   # ↑ de 0.008
    "flip_prob":   0.50,

    # Nuevo: mixup de secuencias
    "mixup_alpha": 0.20,
}

def mixup_batch(Xb, yb, alpha=0.2):
    """Mixup para secuencias de keypoints LSP"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    idx = torch.randperm(Xb.size(0))
    mixed_x = lam * Xb + (1 - lam) * Xb[idx]
    y_a, y_b = yb, yb[idx]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Ratio params/muestras esperado S10:
# ~2.5M params / 4,900 muestras_train ≈ 510:1  (vs 2,040:1 en S9)
```

### TAREA 4 — Temperature Scaling (calibración post-entrenamiento)

```python
"""
Calibrar el LSTM S10 con Temperature Scaling.
Reduce ECE de 0.427 (S9) al objetivo < 0.15.
"""
import torch, numpy as np
from scipy.optimize import minimize_scalar

def find_temperature(logits_val, y_val):
    """Encuentra la temperatura T óptima que minimiza NLL en val set."""
    def nll(T):
        scaled = torch.tensor(logits_val) / T
        log_probs = torch.nn.functional.log_softmax(scaled, dim=1)
        nll_val = -log_probs[np.arange(len(y_val)), y_val].mean().item()
        return nll_val

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return result.x

# Obtener logits del val set con el modelo entrenado
model.eval()
with torch.no_grad():
    logits_val = model(torch.tensor(X_val).to(DEVICE)).cpu().numpy()

T_opt = find_temperature(logits_val, y_val)
print(f"Temperatura óptima: T = {T_opt:.3f}")
# T > 1 → modelo era sobreconfiado (expected para LSTM S9)
# T < 1 → modelo era subconfiado

# Aplicar en inferencia ONNX:
def predecir_calibrado(logits, T):
    logits_scaled = logits / T
    probs = np.exp(logits_scaled - logits_scaled.max())
    return probs / probs.sum()
```

---

## `<API_FASTAPI_COMPLETA>` — Backend de producción

```python
"""
api/main.py — FastAPI + WebSocket + ONNX
Requisitos: pip install fastapi uvicorn websockets onnxruntime mediapipe opencv-python
Lanzar: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio, cv2, json, numpy as np, time
from pathlib import Path
import onnxruntime as ort
import mediapipe as mp

app = FastAPI(title="Traductor LSP API", version="2.0")

# CORS para desarrollo React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://*.gradio.live"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Cargar modelos al inicio ───────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent

lstm_sess = ort.InferenceSession(
    str(ROOT / "checkpoints" / "lstm_signs.onnx"),
    providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
)
with open(ROOT / "data" / "lstm_label2idx.json") as f:
    idx2label = {v: k for k, v in json.load(f).items()}

mp_holistic   = mp.solutions.holistic
holistic_inst = mp_holistic.Holistic(
    static_image_mode=False, model_complexity=1,
    min_detection_confidence=0.4, min_tracking_confidence=0.4,
)

from collections import deque
buffers = {}  # por conexión WebSocket

# ── Endpoints REST ────────────────────────────────────────────────────────────
@app.get("/")
async def raiz():
    return {"status": "ok", "version": "2.0", "modelo": "LSPLSTMBidir-ONNX",
            "n_clases": len(idx2label), "latencia_objetivo_ms": 50}

@app.get("/clases")
async def listar_clases():
    return {"total": len(idx2label), "clases": list(idx2label.values())[:50]}

@app.post("/predecir/imagen")
async def predecir_imagen(imagen_base64: str):
    """Predecir desde imagen base64 (para pruebas de integración)"""
    import base64
    img_bytes = base64.b64decode(imagen_base64)
    arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Imagen inválida")
    seña, conf = _predecir_frame(frame, "http_single")
    return {"seña": seña, "confianza": round(conf, 4)}

# ── WebSocket — Streaming en tiempo real ─────────────────────────────────────
@app.websocket("/ws/traducir/{client_id}")
async def ws_traducir(websocket: WebSocket, client_id: str):
    await websocket.accept()
    buffers[client_id] = deque(maxlen=30)
    print(f"Cliente conectado: {client_id}")
    try:
        while True:
            data = await websocket.receive_bytes()
            t0 = time.perf_counter()

            # Decodificar frame JPEG enviado por el frontend
            arr   = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            seña, conf = _predecir_frame(frame, client_id)
            latencia   = (time.perf_counter() - t0) * 1000

            await websocket.send_json({
                "seña":      seña,
                "confianza": round(conf, 4),
                "latencia_ms": round(latencia, 1),
                "buffer_len":  len(buffers[client_id]),
            })
    except WebSocketDisconnect:
        buffers.pop(client_id, None)
        print(f"Cliente desconectado: {client_id}")

def _predecir_frame(frame, client_id):
    """Extrae keypoints del frame y predice seña LSP"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = holistic_inst.process(rgb)

    row = []
    for lm_list, n in [
        (res.pose_landmarks,       33),
        (res.left_hand_landmarks,  21),
        (res.right_hand_landmarks, 21),
    ]:
        if lm_list:
            row.extend([l.x for l in lm_list.landmark[:n]])
            row.extend([l.y for l in lm_list.landmark[:n]])
        else:
            row.extend([0.0] * (n * 2))

    buffers[client_id].append(np.array(row, dtype=np.float32))

    if len(buffers[client_id]) < 30:
        return None, 0.0

    seq    = np.array(list(buffers[client_id]), dtype=np.float32)[None]  # [1,30,150]
    logits = lstm_sess.run(None, {"input": seq})[0][0]
    probs  = np.exp(logits - logits.max())
    probs /= probs.sum()
    conf   = float(probs.max())
    pred   = idx2label.get(int(probs.argmax()), "???") if conf >= 0.30 else None
    return pred, conf

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
```

### Frontend React — estructura de archivos

```
frontend/
├── package.json           (react, tailwindcss, @shadcn/ui)
├── src/
│   ├── App.tsx            (layout: panel izquierdo + panel derecho)
│   ├── components/
│   │   ├── CameraPanel.tsx     (Canvas overlay de skeleton MediaPipe)
│   │   ├── TranslationPanel.tsx (texto, barra confianza, historial)
│   │   ├── ControlBar.tsx       (modo cámara/video, pausar, limpiar)
│   │   └── MetricsBadge.tsx     (FPS, latencia, confianza)
│   ├── hooks/
│   │   ├── useWebSocket.ts      (ws://localhost:8000/ws/traducir/{id})
│   │   ├── useCamera.ts         (getUserMedia + canvas frame capture)
│   │   └── useTTS.ts            (Web Speech API)
│   └── utils/
│       ├── drawSkeleton.ts      (dibujar landmarks en Canvas)
│       └── postprocess.ts       (buffer deslizante, suavizado texto)
```

```typescript
// src/hooks/useWebSocket.ts
import { useRef, useCallback, useState } from "react";

export function useWebSocket(clientId: string) {
  const ws = useRef<WebSocket | null>(null);
  const [prediction, setPrediction] = useState<{
    seña: string | null; confianza: number; latencia_ms: number;
  }>({ seña: null, confianza: 0, latencia_ms: 0 });

  const connect = useCallback(() => {
    ws.current = new WebSocket(`ws://localhost:8000/ws/traducir/${clientId}`);
    ws.current.onmessage = (e) => {
      const data = JSON.parse(e.data);
      setPrediction(data);
    };
  }, [clientId]);

  const sendFrame = useCallback((frameBlob: Blob) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      frameBlob.arrayBuffer().then((buf) => ws.current!.send(buf));
    }
  }, []);

  return { connect, sendFrame, prediction };
}
```

---

## `<PIPELINE_DIAGRAMA>` — Integración completa del sistema

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    PIPELINE COMPLETO — TRADUCTOR LSP v2.0                       │
└─────────────────────────────────────────────────────────────────────────────────┘

DATOS
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│  Videos.tar    │  │ Keypoints.tar  │  │    SRT.tar     │
│   1.0 GB       │  │   690.2 MB     │  │   191.5 KB     │
│ MP4 por clase  │  │ pkl/glosas/abc │  │ timestamps+txt │
└───────┬────────┘  └───────┬────────┘  └───────┬────────┘
        │                   │                    │
        ▼                   ▼                    ▼
┌───────────────┐  ┌────────────────┐  ┌────────────────┐
│ MediaPipe     │  │ pkl_to_sequence│  │ parsear_srt()  │
│ extraer_kp()  │  │ [30,150] dims  │  │ ground_truth{} │
│ (solo si falta│  │  (principal)   │  │ WER / BLEU     │
│  en Keypoints)│  └───────┬────────┘  └───────┬────────┘
└───────────────┘          │                    │
                           ▼                    │
                  ┌────────────────────┐         │
                  │ build_combined_    │         │
                  │ dataset.py         │         │
                  │ 3 fuentes PKL →    │         │
                  │ dataset_lstm.npz   │         │
                  │ [~7000, 30, 150]   │         │
                  └──────────┬─────────┘         │
                             │                   │
ENTRENAMIENTO                ▼
                  ┌────────────────────┐
                  │ GroupKFold(5) S10  │
                  │ +StratifiedSplit   │
                  │ 70/15/15           │
                  └──────────┬─────────┘
                             │
             ┌───────────────┼──────────────────┐
             ▼               ▼                  ▼
    ┌─────────────┐  ┌──────────────┐  ┌───────────────┐
    │ LogReg/SVC  │  │  RF/ET+HPO   │  │ LSPLSTMBidir  │
    │  Baseline   │  │  Bayesian    │  │  +Attention   │
    │  F1=0.0068  │  │  F1=0.0045   │  │  F1=0.0365    │
    └─────────────┘  └──────────────┘  └───────┬───────┘
                                                │
                                       ┌────────▼────────┐
                                       │  ONNX export    │
                                       │  opset 17       │
                                       │  <50ms / 10MB   │
                                       └────────┬────────┘
                                                │
DEPLOY ─────────────────────────────────────────┼──────────────
                            ┌───────────────────┼──────────────┐
                            ▼                   ▼              ▼
                   ┌──────────────┐   ┌──────────────┐  ┌──────────────┐
                   │ demo/        │   │ api/main.py  │  │ spaces/      │
                   │ app_gradio   │   │ FastAPI WS   │  │ HuggingFace  │
                   │ (actual S9)  │   │ (S13)        │  │ Spaces       │
                   │ ✅ funcional │   │ ⏳ pendiente │  │ ⏳ pendiente │
                   └──────────────┘   └──────────────┘  └──────────────┘
                            │               │
                            └───────┬───────┘
                                    ▼
INTERFAZ ─────────────────────────────────────────────────────────────
                   ┌────────────────────────────────────────────────┐
                   │         USUARIO FINAL (navegador)              │
                   │                                                │
                   │  [Cámara + skeleton MediaPipe] ← Canvas       │
                   │  [Texto LSP → Castellano]      ← WebSocket    │
                   │  [Historial + TTS]              ← Web API     │
                   │  FPS ≥ 24 | Latencia < 200ms                  │
                   └────────────────────────────────────────────────┘

MLOPS ──────────────────────────────────────────────────────────────
  logs/runs.csv   ← 8 corridas S5–S9 registradas
  configs/config.yaml  ← HP centralizados
  checkpoints/         ← Git LFS (.pt, .onnx, .pkl)
  data/ MD5 hash       ← 473828a489f4797b839218c8169a05e1
  figs/ (13 figuras)   ← evidencia visual del informe
  mlflow ui            ← pendiente S10 (código listo en logs/)
```

### Diagrama de flujo de inferencia en tiempo real

```
Webcam frame  →  MediaPipe Holistic
     │               │
     │          pose(33kp) + lhand(21kp) + rhand(21kp)
     │               │
     │          → [150 dims/frame]
     │               │
     │          kp_buffer.append(kp)    # deque(maxlen=30)
     │               │
     │          ¿buffer lleno?
     │          ├─ NO → mostrar "acumulando..." → siguiente frame
     │          └─ SÍ → seq [1, 30, 150]
     │                       │
     │               ONNX Runtime (lstm_signs.onnx)
     │                       │
     │               logits [1, 482]
     │                       │
     │               softmax + temperatura T
     │                       │
     │               conf = max(probs)
     │               ├─ conf < 0.30 → None (no mostrar)
     │               └─ conf ≥ 0.30 → pred = idx2label[argmax]
     │                                       │
     │                              buffer_predicciones.append(pred)
     │                                       │
     │                              votación mayoritaria (ventana=5)
     │                                       │
     └──────────────────────────────→  texto en pantalla (castellano)
                                              │
                                      historial.append(seña)
                                              │
                                      WebSocket → React frontend
```

---

## `<GLOSARIO_TECNICO>` — Términos clave del proyecto

| Término | Definición en contexto LSP |
|---------|---------------------------|
| **MediaPipe Holistic** | Framework de Google para extracción de landmarks corporales: 33 pose + 21 por mano + 468 cara. El proyecto usa solo pose+manos (75 kp, 150 dims). |
| **PKL** | Formato Python pickle. Cada archivo es una lista de dicts `{pose: {x,y}, left_hand: {x,y}, right_hand: {x,y}}` — un dict por frame del video. |
| **Viñeta** | Segmento de video con una sola seña ejecutada, identificado por `{label}_{número}.pkl`. Múltiples viñetas del mismo label provienen del mismo actor/video. |
| **GroupKFold** | Validación cruzada que garantiza que las viñetas del mismo video no mezclan train/val. Evita el data leakage por correlación temporal. |
| **StratifiedShuffleSplit** | Split simple con estratificación por clase. Más rápido que GroupKFold pero con riesgo de leakage si hay múltiples frames del mismo video. |
| **LSPLSTMBidir** | Arquitectura del proyecto: Projection(150→128) + BiLSTM×2(hidden=256) + TemporalAttention + Head(512→256→n_classes). |
| **WeightedRandomSampler** | Sampler de PyTorch que oversamples clases poco frecuentes durante entrenamiento. Equivalente a `class_weight="balanced"` en sklearn. |
| **ONNX** | Open Neural Network Exchange. Formato de modelo independiente de framework. Permite inferencia con `onnxruntime` sin PyTorch, en <50ms. |
| **WER** | Word Error Rate — métrica de ASR/traducción. `WER = (S+D+I)/N` donde S=sustituciones, D=eliminaciones, I=inserciones, N=palabras en referencia. |
| **BLEU Score** | Bilingual Evaluation Understudy — métrica de traducción automática. Compara n-gramas predichos vs referencia. Objetivo del sistema: BLEU > 0.40. |
| **Temperature Scaling** | Post-procesamiento de calibración: dividir logits por temperatura T > 1 reduce sobreconfianza (ECE↓). T se optimiza con el val set. |
| **ECE** | Expected Calibration Error. Mide el gap entre confianza predicha y accuracy observada. LSTM S9 tiene ECE=0.427 (sobreconfiado severo). |
| **Label Smoothing** | Técnica anti-sobreconfianza: CrossEntropyLoss(label_smoothing=0.1) distribuye 10% del peso entre clases incorrectas → calibración más suave. |
| **Mixup** | Aumentación: interpolar pares de secuencias (`x̃ = λ·xa + (1-λ)·xb`) con λ~Beta(α,α). Aumenta diversidad del train sin datos nuevos. |
| **F1-macro** | Promedio no ponderado de F1 por clase. Igual peso a clases frecuentes e infrecuentes. Métrica principal del proyecto por el desbalance severo. |

---

**El sistema final debe ser funcional, inclusivo, reproducible y desplegable en instituciones educativas y de salud del Perú.**

**La interfaz es el corazón del sistema:** el usuario debe ver la cámara con las señas detectadas y el texto traducido en castellano simultáneamente, en tiempo real, sin necesidad de conocimiento técnico previo.

**Métrica de éxito final:** F1-macro > 0.70 en vocabulario LSP de uso diario + latencia <200ms en dispositivo de gama media + accesible desde navegador sin instalación.
