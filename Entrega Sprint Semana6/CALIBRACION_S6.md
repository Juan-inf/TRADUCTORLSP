# CALIBRACIÓN — Sprint Semana 6
## Sistema de Traducción LSP (Lengua de Señas Peruana)

**Proyecto:** Traductor LSP — Sistema de comunicación inclusiva con Deep Learning  
**Fecha:** 2026-05-22  
**Dataset:** 26 videos continuos ("Historias viñetas"), 109 068 frames, 60.7 min, 26 clases  
**Pipeline actual:** MediaPipe Holistic → 75 keypoints (33 pose + 21 mano-der + 21 mano-izq) → Sliding window 30f/15f → CNN-LSTM / ST-GCN  
**Resultados Sprint 6 (baseline ML clásico):** LogReg F1-macro 0.859, Accuracy 0.912, Latencia 0.02 ms

---

## 1. Definición de Features a Probar

### Supuestos (para datasets sin metadatos publicados)
- Cada keypoint tiene coordenadas `(x, y, z)` normalizadas → **225 features por frame** (75 × 3).
- Frecuencia: 30 fps → ventana de 30 frames = 1 segundo de señas.
- Segmentos totales estimados: ~7 271 (stride 15, 70/15/15 a nivel de segmento).
- Signers: múltiples por viñeta, sin ID de signer disponible en las carpetas actuales.

---

### Baseline — Features mínimas (225 valores/frame)

**¿Qué incluye?** Las coordenadas crudas normalizadas de los 75 keypoints MediaPipe Holistic, aplanadas por frame.

| Grupo | Keypoints | Features |
|---|---|---|
| Pose (hombros, codos, muñecas, caderas) | 33 | 99 (×3) |
| Mano derecha | 21 | 63 (×3) |
| Mano izquierda | 21 | 63 (×3) |
| **Total por frame** | **75** | **225** |

**Normalización obligatoria:** Restar posición de muñeca derecha (`pose[16]`) para hacerlo invariante a posición del signer en cámara.

**Justificación:** En LSP (como en ASL y LSA), las configuraciones de mano (*handshape*) y la posición relativa al cuerpo son los dos parámetros fonológicos primarios. Con solo coordenadas normalizadas se captura ambos.

**Representación por segmento:**
```
X_baseline.shape = (n_segmentos, 30, 225)   # (N, T, F)
```

---

### Variante 1 — Features dinámicas derivadas (+90 features/frame)

Agrega información de **movimiento**, el tercer parámetro fonológico del lenguaje de señas.

| Feature derivada | Cálculo | Features añadidas | Justificación |
|---|---|---|---|
| **Velocidad** (Δposición) | `kp[t] - kp[t-1]` sobre 75 kp | 225 | El movimiento de manos es fonémico: "agua" vs "beber" difieren principalmente en velocidad/dirección |
| **Aceleración** (Δ²posición) | `vel[t] - vel[t-1]` | 225 | Captura inicio/fin brusco de seña; relevante para señas de un tiempo vs dos tiempos |
| **Distancia entre manos** | `‖muñeca_der - muñeca_izq‖` (euclidiana) | 1 escalar | Señas bimanuales vs unimanuales |

> En frames `t=0` la velocidad es cero (padding). En frames con keypoints perdidos (NaN), interpolar linealmente antes de calcular derivadas.

**Representación:**
```
X_var1.shape = (n_segmentos, 30, 225 + 225 + 225 + 1)  # 676 features/frame
```

---

### Variante 2 — Features lingüísticas avanzadas (+features por segmento)

Incorpora rasgos del nivel **morfológico y prosódico** del LSP.

| Feature avanzada | Cálculo | Tipo | Justificación lingüística |
|---|---|---|---|
| **Orientación de palma** | Ángulo entre vectores palma→índice y palma→meñique | Escalar/frame | La orientación (*orientation*) es parámetro fonológico: "yo" vs "tú" difieren solo en orientación |
| **Simetría bilateral** | Distancia euclidiana entre keypoints espejados izq/der: `‖pose[15]-pose[16]‖, ‖pose[13]-pose[14]‖` | Escalar/frame | Señas simétricas son la mayoría en LSP; asimetría discrimina significado |
| **Repetición interna** | Cross-correlación de la trayectoria de muñeca consigo misma con lag=T/2 | Escalar/segmento | Muchas señas LSP se articulan con movimiento repetido (ej. "muchos", "siempre") |
| **Apertura de mano** | Media de `‖punta_dedo - metacarpo‖` sobre 5 dedos | Escalar/frame | Handshape abierta vs cerrada es el rasgo visual más discriminativo |
| **Velocidad angular** de muñeca | Δ(ángulo muñeca) / Δt, calculado sobre 3 keypoints del carpo | Escalar/frame | Movimientos de rotación (pronación/supinación) diferencian variantes de señas |

**Código de extracción de features avanzadas (Variante 2):**
```python
import numpy as np

# landmarks.shape = (T, 75, 3)  — segmento completo
def extract_v2_features(landmarks: np.ndarray) -> dict:
    T = landmarks.shape[0]

    # Orientación de palma derecha (keypoints 0=muñeca, 5=índice_mcp, 17=meñique_mcp)
    rh_offset = 33  # mano derecha empieza en índice 33
    wrist  = landmarks[:, rh_offset + 0, :2]   # (T, 2)
    index  = landmarks[:, rh_offset + 5, :2]
    pinky  = landmarks[:, rh_offset + 17, :2]
    v1 = index - wrist
    v2 = pinky - wrist
    cross = v1[:, 0]*v2[:, 1] - v1[:, 1]*v2[:, 0]
    palm_orientation = np.arctan2(cross, np.einsum('ti,ti->t', v1, v2))  # (T,)

    # Simetría bilateral (muñecas)
    rw = landmarks[:, 16, :2]   # pose[16] = muñeca derecha
    lw = landmarks[:, 15, :2]   # pose[15] = muñeca izquierda
    symmetry = np.linalg.norm(rw - lw, axis=1)  # (T,)

    # Repetición interna (correlación de muñeca derecha consigo misma)
    wrist_traj = rw[:, 0]  # solo x
    half = T // 2
    if half > 0:
        repetition = np.correlate(wrist_traj[:half], wrist_traj[half:], mode='full').max()
    else:
        repetition = 0.0

    # Apertura media de mano derecha (distancia punta↔mcp para 5 dedos)
    tips   = [rh_offset+4, rh_offset+8, rh_offset+12, rh_offset+16, rh_offset+20]
    mcps   = [rh_offset+2, rh_offset+5, rh_offset+9,  rh_offset+13, rh_offset+17]
    aperture = np.mean([
        np.linalg.norm(landmarks[:, t, :2] - landmarks[:, m, :2], axis=1)
        for t, m in zip(tips, mcps)
    ], axis=0)  # (T,)

    return {
        "palm_orientation": palm_orientation,   # (T,)
        "symmetry":         symmetry,           # (T,)
        "repetition":       repetition,         # scalar
        "aperture":         aperture,           # (T,)
    }
```

**Representación final Variante 2:**
```
X_var2.shape = (n_segmentos, 30, 225+225+225+1+1+1+1+1)  # ≈683 features/frame + 1 escalar/segmento
```

---

## 2. Diseño del Split y Validación

### Estrategia: Split por Vídeo (Video-Level Split)

**Problema clave:** Los frames de un mismo video son altamente correlacionados. Si un segmento de la viñeta 12 aparece en train y otro segmento de la misma viñeta en test, el modelo aprende la "firma" del signer, no la seña en sí → **data leakage**.

**Split propuesto:**

```
Dataset: 26 videos (viñetas 2–43, excluyendo vineta 9)

TRAIN  (18 videos, ~70%): vinetas 2,3,4,5,6,8,11,12,13,14,15,17,18,19,20,21,22,23
VAL    ( 4 videos, ~15%): vinetas 24,25,27,29
TEST   ( 4 videos, ~15%): vinetas 30,37,42,43
```

**Criterios de selección:**
- Los 4 videos de test son los de mayor duración (viñetas 37, 42, 43) y uno de resolución FullHD (viñeta 30) → representan diversidad de condiciones.
- Ningún signer se repite entre splits (dado que cada viñeta tiene intérprete diferente).
- **NO** mezclar segmentos de la misma viñeta entre splits.

**¿Por qué NO split por frames ni temporal global?**

| Estrategia | Riesgo | Veredicto |
|---|---|---|
| Split por frame (aleatorio) | Frames del mismo video en train y test → leakage severo | ✗ Prohibido |
| Split temporal global (primeros 70% del tiempo) | Puede mezclar viñetas cortas completas en test | ✗ Inestable |
| **Split por video** | Separa signers, evita autocorrelación temporal | ✓ Correcto |
| Split por signer (si hubiera ID) | Óptimo para generalización | ✓ Ideal (no disponible) |

**Cross-validation (k=5 folds):**
```python
from sklearn.model_selection import GroupKFold

# group = número de viñeta del segmento
gkf = GroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=video_ids)):
    # garantiza que ninguna viñeta aparezca en train y val a la vez
    ...
```

### Manejo de Keypoints Nulos (frames sin detección)

MediaPipe falla en oclusiones parciales, manos fuera de cuadro, o iluminación deficiente.

**Protocolo en orden de preferencia:**

```python
def impute_keypoints(segment: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    segment: (T, 75, 3) — NaN donde MediaPipe no detectó
    threshold: fracción máxima de frames nulos aceptable
    """
    T = segment.shape[0]
    nan_mask = np.isnan(segment).any(axis=(1, 2))  # (T,) booleano
    nan_frac  = nan_mask.mean()

    if nan_frac > threshold:
        return None  # descartar segmento

    # Interpolación lineal por keypoint
    for kp in range(75):
        for coord in range(3):
            series = segment[:, kp, coord]
            nans = np.isnan(series)
            if nans.any() and not nans.all():
                x = np.arange(T)
                segment[:, kp, coord] = np.interp(x, x[~nans], series[~nans])

    return segment
```

**Reglas:**
- Si >50% de frames del segmento son NaN → **descartar**.
- Si <50% → **interpolación lineal** por cada coordenada de cada keypoint.
- La imputación se **ajusta solo en TRAIN**. Los parámetros (media, desvío para normalización Z-score) se aprenden en train y se aplican a val/test.

---

## 3. Plan de Ablaciones

### Tabla Comparativa (5-Fold Cross-Validation)

| Configuración | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | **Media** | **Std** |
|---|---|---|---|---|---|---|---|
| **Baseline (225 f/frame)** | | | | | | | |
| — Accuracy (%) | — | — | — | — | — | — | — |
| — F1-macro | — | — | — | — | — | — | — |
| — WER (%) | — | — | — | — | — | — | — |
| — Latencia (ms/frame) | — | — | — | — | — | — | — |
| **Variante 1 (+velocidad, aceleración, dist. manos)** | | | | | | | |
| — Accuracy (%) | — | — | — | — | — | — | — |
| — F1-macro | — | — | — | — | — | — | — |
| — WER (%) | — | — | — | — | — | — | — |
| — Latencia (ms/frame) | — | — | — | — | — | — | — |
| **Variante 2 (+orientación, simetría, repetición, apertura)** | | | | | | | |
| — Accuracy (%) | — | — | — | — | — | — | — |
| — F1-macro | — | — | — | — | — | — | — |
| — WER (%) | — | — | — | — | — | — | — |
| — Latencia (ms/frame) | — | — | — | — | — | — | — |

> Las celdas con `—` se rellenan al ejecutar el experimento. El script de ablación está en la sección de código más abajo.

### Cómo calcular WER para señas
```python
# WER adaptado a glosas (cada ventana es una "palabra")
def wer_gloss(y_true: list, y_pred: list) -> float:
    from jiwer import wer
    ref = " ".join([str(g) for g in y_true])
    hyp = " ".join([str(g) for g in y_pred])
    return wer(ref, hyp)
```

### Script de ablación
```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score
import numpy as np, time

def run_ablation(X, y, groups, config_name, n_folds=5):
    gkf = GroupKFold(n_splits=n_folds)
    results = []
    for fold, (tr, val) in enumerate(gkf.split(X, y, groups)):
        scaler = StandardScaler().fit(X[tr].reshape(len(tr), -1))
        Xtr = scaler.transform(X[tr].reshape(len(tr), -1))
        Xvl = scaler.transform(X[val].reshape(len(val), -1))

        model = LogisticRegression(C=1, max_iter=1000, random_state=42)
        model.fit(Xtr, y[tr])

        t0 = time.perf_counter()
        pred = model.predict(Xvl)
        latency_ms = (time.perf_counter() - t0) / len(val) * 1000

        results.append({
            "config": config_name, "fold": fold+1,
            "accuracy": accuracy_score(y[val], pred),
            "f1_macro": f1_score(y[val], pred, average='macro'),
            "latency_ms": latency_ms,
        })
    return results
```

### Hiperparámetros a variar por modelo (Deep Learning)

| Modelo | Param 1 | Param 2 | Param 3 |
|---|---|---|---|
| CNN-LSTM | hidden=128/256/512 | dropout=0.3/0.5 | lr=1e-3/1e-4 |
| ST-GCN | num_layers=4/6 | edge_strategy=spatial/temporal | dropout=0.3/0.5 |
| Fusión | fusion=concat/attention/weighted | alpha=0.3/0.5/0.7 | freeze_backbone=T/F |

---

## 4. Importancia de Variables / SHAP

### Desafío para modelos secuenciales

Los modelos LSTM/Transformer/GCN reciben `(B, T, F)`. SHAP Kernel no asume estructura, por lo que funciona pero requiere colapsar la dimensión temporal primero (aplanar o resumir).

### Estrategia recomendada: SHAP sobre features aplanadas (para benchmark rápido)

```python
import shap
import numpy as np

# X_flat.shape = (n_samples, T*F) — e.g., (7271, 30*225)
background = X_flat[train_idx][:100]   # subsample para KernelSHAP
explainer  = shap.KernelExplainer(model.predict_proba, background)

# Explicar 50 muestras de test
X_explain  = X_flat[test_idx][:50]
shap_values = explainer.shap_values(X_explain, nsamples=200)
# shap_values: lista de 26 arrays (uno por clase), cada uno (50, 6750)
```

### Reconstruir importancia por keypoint y por timestep

```python
def shap_by_keypoint(shap_vals_class: np.ndarray, T: int, n_kp: int, coord: int = 3) -> np.ndarray:
    """
    shap_vals_class: (n_samples, T*n_kp*coord)
    Retorna: (n_kp,) — importancia media por keypoint
    """
    reshaped = shap_vals_class.reshape(-1, T, n_kp, coord)
    return np.abs(reshaped).mean(axis=(0, 1, 3))  # promedio sobre muestras, tiempo, coord

kp_importance = shap_by_keypoint(shap_values[best_class], T=30, n_kp=75, coord=3)
top5_kp_idx   = np.argsort(kp_importance)[::-1][:5]

KP_NAMES = (
    [f"pose_{i}"    for i in range(33)] +
    [f"rhand_{i}"   for i in range(21)] +
    [f"lhand_{i}"   for i in range(21)]
)
print("Top-5 keypoints más influyentes:")
for i, idx in enumerate(top5_kp_idx):
    print(f"  {i+1}. {KP_NAMES[idx]} (SHAP={kp_importance[idx]:.4f})")
```

### SHAP para ST-GCN (modelo de grafos, approach alternativo)

Para ST-GCN ya entrenado, usar Gradient-based SHAP (DeepSHAP):
```python
import shap, torch

model_eval = stgcn_model.eval()
background  = torch.tensor(X_train[:50]).float()
explainer   = shap.GradientExplainer(model_eval, background)

X_test_t    = torch.tensor(X_test[:20]).float()
shap_values = explainer.shap_values(X_test_t)
# shap_values[class_idx].shape = (20, T, n_kp, coord)
```

### Gráficos para mostrar estabilidad entre folds

**1. SHAP summary plot (beeswarm) por fold:**
```python
shap.summary_plot(
    shap_values[target_class].reshape(n_samples, -1),
    feature_names=[f"{kp}_{t}" for t in range(30) for kp in KP_NAMES],
    max_display=20
)
```

**2. Ranking de top-10 keypoints por fold (para detectar inestabilidad):**
```python
import matplotlib.pyplot as plt
import pandas as pd

fold_rankings = {}  # dict: fold → array (n_kp,) de importancias
for fold_id, shap_fold in enumerate(all_fold_shap):
    fold_rankings[f"Fold {fold_id+1}"] = shap_by_keypoint(shap_fold, 30, 75, 3)

df_ranks = pd.DataFrame(fold_rankings, index=KP_NAMES)
df_ranks.nlargest(10, "Fold 1").plot(kind="bar", figsize=(12, 5))
plt.title("Top-10 keypoints por importancia SHAP — estabilidad entre folds")
plt.ylabel("SHAP medio |valor|")
plt.tight_layout()
plt.savefig("shap_stability_folds.png", dpi=150)
```

**Interpretación esperada para LSP:**
- Los 5 keypoints más importantes deberían incluir: muñecas (pose 15/16), puntas de dedos (rhand 8, 12), y articulaciones del hombro (pose 11/12).
- Si SHAP identifica keypoints de cara (pose 0–10) como los más importantes → señal de que el modelo está aprendiendo la identidad del signer, no la seña.

---

## 5. Curvas de Aprendizaje y Calibración

### 5.1 Descripción de curvas esperadas

**Caso A: Alto bias (underfitting)**
```
Accuracy
  1.0 |
  0.9 |
  0.8 |____________________  ← train y val convergen, ambas bajas (~0.6)
  0.7 |
  0.6 |── train
      |── val
  0.5 +------------------------→ Tamaño de training set
```
- Train y val tienen accuracy similar pero baja.
- Agregar más datos NO ayuda.
- **Acción:** Aumentar capacidad del modelo (más capas LSTM, añadir attention) o agregar features (Variante 1/2).

**Caso B: Alta varianza (overfitting)**
```
Accuracy
  1.0 |        _______________  ← train sube a ~0.99
  0.9 |_______/
  0.8 |
  0.7 |  _____________________ ← val se estanca ~0.70
  0.6 |_/
  0.5 +------------------------→ Tamaño de training set
```
- Brecha grande entre train y val.
- **Acción:** Dropout, weight decay, data augmentation, reducir features con baja importancia SHAP.

**Caso C: Datos insuficientes**
```
Accuracy
  1.0 |                   /─── train (alta varianza de fold a fold)
  0.8 |  /\/\/\/\/\/\/\/\/
  0.6 |  /\/\/\/\/\/\/\/\/     ← val con alta varianza entre folds
  0.4 |
  0.2 +------------------------→ Tamaño de training set
```
- Alta varianza entre folds del CV.
- Las curvas no convergen aunque aumente el tamaño.
- **Acción:** Más datos (data augmentation sobre keypoints), transfer learning.

### 5.2 Acción prioritaria para el dataset LSP actual

| Factor | Situación actual | Prioridad |
|---|---|---|
| Cantidad de datos | ~7 271 segmentos, 26 clases → media ~280/clase | ⚠️ Baja |
| Calidad de features | Solo coordenadas brutas (Baseline) | ⚠️ Mejorable |
| Regularización | Sin dropout ni augmentation en Sprint 6 | ⚠️ Pendiente |

**Recomendación priorizada:**

```
1° → Data augmentation sobre keypoints (bajo costo, alto impacto)
2° → Agregar features dinámicas (Variante 1: velocidad/aceleración)
3° → Regularización (dropout=0.3 en LSTM, weight_decay=1e-4)
4° → Más datos reales (grabar nuevos videos — alto costo)
```

**Data augmentation para keypoints LSP:**
```python
import numpy as np

def augment_keypoints(segment: np.ndarray, p: float = 0.5) -> np.ndarray:
    """segment: (T, 75, 3)"""
    seg = segment.copy()

    # Flip horizontal (espejado) — válido para señas simétricas
    if np.random.rand() < p:
        seg[:, :, 0] = -seg[:, :, 0]  # invertir eje x

    # Jitter gaussiano leve (simula error de detección MediaPipe)
    if np.random.rand() < p:
        seg += np.random.normal(0, 0.01, seg.shape)

    # Escala (simula distintas distancias a cámara)
    if np.random.rand() < p:
        scale = np.random.uniform(0.9, 1.1)
        seg *= scale

    # Desplazamiento temporal (speed perturbation): re-muestrear a velocidad ±10%
    if np.random.rand() < p:
        T = seg.shape[0]
        speed = np.random.uniform(0.9, 1.1)
        old_t = np.linspace(0, T-1, T)
        new_t = np.linspace(0, T-1, int(T * speed))
        from scipy.interpolate import interp1d
        for kp in range(75):
            for c in range(3):
                f = interp1d(old_t, seg[:, kp, c], kind='linear', fill_value='extrapolate')
                seg[:, kp, c] = f(new_t)[:T]

    return seg
```

> **IMPORTANTE:** Flip horizontal solo es semánticamente válido para señas simétricas. Para señas asimétricas (que distinguen mano dominante) hay que intercambiar también los índices de mano derecha/izquierda.

### 5.3 Calibración de probabilidades por seña

Si el modelo devuelve `softmax` sobre 26 clases, evaluar si las probabilidades están calibradas:

```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Para cada clase (one-vs-rest):
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for cls_idx, ax in enumerate(axes.flat[:8]):
    y_bin    = (y_test == cls_idx).astype(int)
    prob_cls = y_prob[:, cls_idx]

    frac_pos, mean_pred = calibration_curve(y_bin, prob_cls, n_bins=10)
    ax.plot(mean_pred, frac_pos, "s-", label="Modelo")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Calibración perfecta")
    ax.set_title(f"Clase {cls_idx} (viñeta)")
    ax.set_xlabel("Probabilidad predicha")
    ax.set_ylabel("Fracción positivos")

plt.suptitle("Curvas de calibración por glosa — LSP")
plt.tight_layout()
plt.savefig("calibration_curves_lsp.png", dpi=150)
```

**Expected Calibration Error (ECE):**
```python
def ece_score(y_true_bin, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true_bin[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return ece

# ECE < 0.05 → bien calibrado; ECE > 0.15 → necesita temperatura scaling o Platt scaling
```

**Corrección de calibración si ECE > 0.10:**
```python
from sklearn.calibration import CalibratedClassifierCV
model_cal = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
model_cal.fit(X_val_flat, y_val)
```

---

## 6. Checklist de Validación — Sprint 2 (LSP + Keypoints)

### 6.1 Split y datos

- [ ] **Split a nivel de viñeta** (no de frame ni de segmento aleatoriamente): ninguna viñeta aparece en más de un split.
- [ ] **Sin leakage temporal**: todos los segmentos de la viñeta N están en el mismo split.
- [ ] El split se define una sola vez y se guarda como `data/splits/split_v1.json` con la lista de viñetas por split.
- [ ] Los índices de segmento para train/val/test están fijos y reproducibles.
- [ ] GroupKFold usa `groups = video_ids` (número de viñeta), nunca `random_state` en el split.

### 6.2 Preprocesamiento (fit solo en TRAIN)

- [ ] `StandardScaler` / `MinMaxScaler` es `.fit()` solo sobre segmentos de **train**.
- [ ] Los parámetros del scaler (mean_, scale_) se guardan en `models/scaler_v1.pkl` para inference.
- [ ] La imputación de NaN (interpolación lineal) se aplica **antes** de normalizar.
- [ ] El threshold de descarte de segmentos nulos (>50% NaN) se aplica **antes** del split.
- [ ] Data augmentation se aplica **solo en train**, nunca en val/test.

### 6.3 Reproducibilidad

- [ ] Semilla global fijada en todo el pipeline:
  ```python
  import random, numpy as np, torch
  SEED = 42
  random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
  torch.backends.cudnn.deterministic = True
  ```
- [ ] El notebook/script comienza con la celda de semillas antes de cualquier operación aleatoria.
- [ ] Los argumentos del modelo (hidden_size, dropout, lr) están en `configs/semana6_config.yaml`, no hardcodeados.
- [ ] La versión del split, del scaler y del modelo están versionadas (v1, v2, …) en sus nombres de archivo.

### 6.4 Métricas y logs

- [ ] Para cada fold se loguea: accuracy, F1-macro, WER, latencia ms/frame.
- [ ] La matriz de confusión se guarda como imagen para identificar pares de glosas confundidas.
- [ ] Se reporta F1 **por clase** (no solo macro) para detectar clases con bajo rendimiento.
- [ ] El log incluye: número de segmentos descartados, fracción NaN por split, shape de X_train/val/test.
- [ ] Se valida que `len(set(train_vids) & set(val_vids)) == 0` (sin traslape de viñetas).

### 6.5 Modelo e inferencia

- [ ] El modelo no ha visto datos de val/test durante el entrenamiento (verificar con early stopping sobre val).
- [ ] La normalización en inferencia usa los parámetros del scaler entrenado en train.
- [ ] El pipeline de inferencia acepta un nuevo video MP4 y produce predicción en <200 ms por seña (objetivo del proyecto).
- [ ] Se prueba el pipeline con al menos 1 video completamente nuevo (no de las 26 viñetas).

### 6.6 SHAP (si aplica)

- [ ] SHAP se calcula solo sobre muestras de test (nunca de train).
- [ ] El background para KernelSHAP es un subsample de train (100–200 muestras).
- [ ] Los top-5 keypoints son interpretables lingüísticamente (muñecas, puntas de dedos esperadas).
- [ ] Se genera el plot de estabilidad SHAP entre folds.

---

## 7. Entregables para la Demo del Sprint 2

### 7.1 Video original vs keypoints superpuestos

**Objetivo:** Mostrar que el pipeline de extracción funciona correctamente.

```python
import cv2, mediapipe as mp

mp_holistic  = mp.solutions.holistic
mp_drawing   = mp.solutions.drawing_utils

def draw_landmarks_on_video(input_path: str, output_path: str):
    cap = cv2.VideoCapture(input_path)
    fps, w, h = cap.get(5), int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    with mp_holistic.Holistic(min_detection_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)
            if res.pose_landmarks:
                mp_drawing.draw_landmarks(frame, res.pose_landmarks,    mp_holistic.POSE_CONNECTIONS)
            if res.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if res.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, res.left_hand_landmarks,  mp_holistic.HAND_CONNECTIONS)
            out.write(frame)
    cap.release(); out.release()

# Generar para 30 segundos de la viñeta 3 (video de demostración)
draw_landmarks_on_video(
    "data/videos/original/Historias vinetas (3).mp4",
    "data/demo_keypoints_vineta3.mp4"
)
```

**Presentar:** Side-by-side (original | keypoints) — clips de 10–15 segundos, 2–3 señas distintas.

---

### 7.2 Comparativa de métricas (tabla + gráfico)

**Tabla resumen (rellenar con resultados reales):**

| Modelo | Accuracy | F1-macro | WER | Latencia |
|---|---|---|---|---|
| Dummy Classifier | 0.052 | 0.037 | ~0.98 | 0.02 ms |
| KNN (k=5) | 0.864 | 0.791 | — | 1.23 ms |
| **LogReg (Baseline)** | **0.912** | **0.859** | — | **0.02 ms** |
| LSTM (Variante 1) | — | — | — | — |
| ST-GCN (Variante 2) | — | — | — | — |

**Gráfico de barras comparativo:**
```python
import matplotlib.pyplot as plt
import numpy as np

modelos = ["Dummy", "KNN", "LogReg", "LSTM V1", "ST-GCN V2"]
f1s     = [0.037, 0.791, 0.859, None, None]  # rellenar

plt.figure(figsize=(8, 5))
bars = plt.bar(modelos, f1s, color=["#d9534f","#f0ad4e","#5bc0de","#5cb85c","#337ab7"])
plt.axhline(0.859, linestyle='--', color='gray', label='Baseline LogReg')
plt.ylabel("F1-macro")
plt.title("Comparativa de modelos — Traductor LSP Sprint 2")
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig("comparativa_modelos_sprint2.png", dpi=150)
```

---

### 7.3 Curva de aprendizaje

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

train_sizes, train_scores, val_scores = learning_curve(
    estimator=model,
    X=X_train_flat, y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=GroupKFold(n_splits=5),
    groups=train_groups,
    scoring='f1_macro',
    n_jobs=-1
)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Train F1")
plt.fill_between(train_sizes,
                 train_scores.mean(1) - train_scores.std(1),
                 train_scores.mean(1) + train_scores.std(1), alpha=0.2)
plt.plot(train_sizes, val_scores.mean(axis=1), "s--", label="Val F1")
plt.fill_between(train_sizes,
                 val_scores.mean(1) - val_scores.std(1),
                 val_scores.mean(1) + val_scores.std(1), alpha=0.2)
plt.xlabel("Segmentos de entrenamiento")
plt.ylabel("F1-macro")
plt.title("Curva de Aprendizaje — Traductor LSP")
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig("learning_curve_sprint2.png", dpi=150)
```

---

### 7.4 Top Features por SHAP

**Gráfico de barras SHAP (top-5 keypoints):**
```python
import shap, matplotlib.pyplot as plt

# Después de calcular kp_importance (sección 4)
top10_idx   = np.argsort(kp_importance)[::-1][:10]
top10_names = [KP_NAMES[i] for i in top10_idx]
top10_vals  = kp_importance[top10_idx]

plt.figure(figsize=(8, 5))
plt.barh(top10_names[::-1], top10_vals[::-1], color="#5bc0de")
plt.xlabel("SHAP medio |valor|")
plt.title("Top-10 keypoints más influyentes — Predicción LSP")
plt.tight_layout()
plt.savefig("shap_top10_keypoints.png", dpi=150)
```

**Interpretación esperada:**
- `rhand_8` (punta índice der.), `rhand_12` (punta medio der.) → configuración de mano.
- `pose_16` (muñeca der.) → posición en espacio de articulación.
- `pose_11`, `pose_12` (hombros) → referencia de espacio de señas.
- Si aparece `pose_0` (nariz) o `pose_1` (ojo izq.) → el modelo está usando identidad del signer → alerta de bias.

---

### 7.5 Conclusión: ¿Qué features me quedo?

**Árbol de decisión para selección de features:**

```
¿Las features de Variante 1 mejoran F1 > 0.02 sobre Baseline?
├── SÍ → Incluir velocidad y aceleración en el pipeline final.
│         ¿Las features de Variante 2 mejoran adicionalmente > 0.02?
│         ├── SÍ → Incluir orientación de palma y apertura (las de mayor SHAP).
│         │         Descartar repetición y simetría si SHAP < 0.001.
│         └── NO → Quedarse con Variante 1 + regularización.
└── NO → Revisar implementación de velocidad (¿normalización correcta?).
          Comprobar que el baseline no tiene data leakage.
```

**Ejemplo de conclusión (rellenar con resultados reales):**
> "Las features de velocidad de muñecas (Variante 1) aportaron un incremento de F1-macro de 0.859 → 0.XXX. Las features de Variante 2 no mostraron mejora significativa (p>0.05 en test de Wilcoxon entre folds), por lo que se descartaron. La orientación de palma tuvo SHAP promedio de 0.0023 (top-3 en 4/5 folds), sugiriendo que debe incluirse aunque con menor prioridad. **Configuración final: Baseline + velocidad + orientación de palma.**"

---

## Recomendación Priorizada de Implementación

### Orden sugerido para el Sprint 2

| Prioridad | Acción | Esfuerzo | Impacto esperado |
|---|---|---|---|
| **1** | Implementar split por video con GroupKFold (evitar leakage) | 1–2 h | Alto: valida resultados actuales |
| **2** | Agregar features de velocidad y aceleración (Variante 1) | 2–3 h | Alto: captura movimiento de señas |
| **3** | Data augmentation: flip, jitter, speed perturbation | 3–4 h | Alto: duplica dataset efectivo |
| **4** | Curvas de aprendizaje con GroupKFold | 1–2 h | Medio: diagnóstica bias/varianza |
| **5** | SHAP sobre modelo LogReg o LSTM flatten | 2–3 h | Medio: interpretabilidad y detección de bias de signer |
| **6** | Calibración de probabilidades (ECE) | 1 h | Bajo-Medio: mejora confiabilidad de scores |
| **7** | Variante 2 (orientación, apertura) | 4–6 h | Bajo: solo si Var1 ya converge bien |

> **Primera sesión de trabajo (4h):** Completar prioridades 1 y 2. Con un split limpio y las features de velocidad, se puede obtener una comparativa válida y honesta antes de agregar complejidad.

---

*Documento generado para el Proyecto TRADUCTOR LSP — Sprint Semana 6 — 2026-05-22*
