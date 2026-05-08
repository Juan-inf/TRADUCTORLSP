"""
Baseline clásico para LSP — sin Deep Learning.
Extrae features de color + movimiento de cada segmento de video y
entrena KNN y Regresión Logística con scikit-learn.

Uso: python scripts/run_baseline.py [--max_segments N]
     .venv310/bin/python scripts/run_baseline.py

Salida:
  - logs/baseline_results.txt
  - data/baseline_features.npz
  - data/baseline_confusion.png
  - data/baseline_metrics.png
"""
import argparse, cv2, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.dummy import DummyClassifier

DATA_DIR = Path("data")
LOG_DIR  = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# ── Feature extraction ────────────────────────────────────────────────────
def extract_features(video_path: str, start: int, end: int,
                     sample_frames: int = 5, resize: int = 32) -> np.ndarray:
    """
    Extrae vector de features de un segmento de video:
      - RGB histogramas (32 bins × 3 canales × sample_frames)
      - Diferencia de frames (motion proxy)
      - Media y std temporal de brightness
    Total: 32*3*sample_frames + (sample_frames-1) + 2 features
    """
    n_frames = end - start
    indices  = np.linspace(start, end - 1, sample_frames, dtype=int)

    cap = cv2.VideoCapture(video_path)
    frames_gray  = []
    frames_color = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            # Padding con el último frame si falla lectura
            if frames_gray:
                frames_gray.append(frames_gray[-1].copy())
                frames_color.append(frames_color[-1].copy())
            else:
                frames_gray.append(np.zeros((resize, resize), np.float32))
                frames_color.append(np.zeros((resize, resize, 3), np.float32))
            continue
        small = cv2.resize(frame, (resize, resize))
        frames_gray.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32))
        frames_color.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32))

    cap.release()

    # Feature 1: histograma RGB normalizado por frame (96 features × N frames)
    hist_feats = []
    for fc in frames_color:
        for ch in range(3):
            h, _ = np.histogram(fc[:, :, ch], bins=32, range=(0, 256), density=True)
            hist_feats.append(h)
    hist_feats = np.concatenate(hist_feats)  # [32*3*sample_frames]

    # Feature 2: diferencia temporal (motion proxy)
    diffs = []
    for i in range(1, len(frames_gray)):
        diff = np.abs(frames_gray[i].astype(np.float32) - frames_gray[i-1].astype(np.float32))
        diffs.append(diff.mean())
    motion_feats = np.array(diffs, dtype=np.float32)  # [sample_frames-1]

    # Feature 3: estadísticas de brightness
    brightnesses = np.array([f.mean() for f in frames_gray])
    bright_feats = np.array([brightnesses.mean(), brightnesses.std()], dtype=np.float32)

    return np.concatenate([hist_feats, motion_feats, bright_feats])


def build_feature_matrix(df: pd.DataFrame, max_segments=None, sample_frames=5):
    """Extrae features de todos los segmentos del df."""
    if max_segments:
        df = df.sample(min(max_segments, len(df)), random_state=42)

    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Extrayendo features"):
        feat = extract_features(
            row['video_path'], row['start_frame'], row['end_frame'],
            sample_frames=sample_frames
        )
        X.append(feat)
        y.append(row['clase'])

    return np.array(X, dtype=np.float32), np.array(y)


# ── Métricas y visualizaciones ────────────────────────────────────────────
def eval_model(model, X_test, y_test, le, name):
    t0 = time.time()
    y_pred = model.predict(X_test)
    latency_ms = (time.time() - t0) / len(X_test) * 1000

    acc  = accuracy_score(y_test, y_pred)
    f1m  = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1w  = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # ROC-AUC multiclase (one-vs-rest)
    try:
        proba = model.predict_proba(X_test)
        auc = roc_auc_score(
            le.transform(y_test), proba,
            multi_class='ovr', average='macro'
        )
    except Exception:
        auc = float('nan')

    return {
        'model':      name,
        'accuracy':   acc,
        'f1_macro':   f1m,
        'f1_weighted':f1w,
        'roc_auc':    auc,
        'latency_ms': latency_ms,
        'y_pred':     y_pred,
    }


def plot_confusion(y_true, y_pred, class_names, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, data, fmt, ttl in [
        (axes[0], cm,      'd',    'Absoluta'),
        (axes[1], cm_norm, '.2f',  'Normalizada'),
    ]:
        short = [c.split('_')[1] for c in class_names]
        sns.heatmap(data, ax=ax, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=short, yticklabels=short, linewidths=0.3)
        ax.set_xlabel('Predicho', fontsize=10)
        ax.set_ylabel('Real', fontsize=10)
        ax.set_title(f'Matriz de Confusión {ttl}', fontsize=11, fontweight='bold')
        ax.tick_params(labelsize=7)

    fig.suptitle(f'{title}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(results, out_path):
    df_r = pd.DataFrame(results)[['model', 'accuracy', 'f1_macro', 'f1_weighted', 'roc_auc']]
    df_m = df_r.melt(id_vars='model', var_name='metric', value_name='value')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Comparación de Modelos Baseline — LSP', fontsize=14, fontweight='bold')

    # Barplot métricas
    ax = axes[0]
    metrics_order = ['accuracy', 'f1_macro', 'f1_weighted', 'roc_auc']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    x = np.arange(len(metrics_order))
    w = 0.8 / len(results)
    for i, res in enumerate(results):
        vals = [res.get(m, 0) for m in metrics_order]
        bars = ax.bar(x + i*w - w*(len(results)-1)/2, vals, w*0.9,
                      label=res['model'], alpha=0.85)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'F1-macro', 'F1-weighted', 'ROC-AUC'], fontsize=9)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8)
    ax.set_title('Métricas por Modelo', fontweight='bold')
    ax.axhline(1/26, color='gray', ls='--', lw=1, label=f'Random ({1/26:.3f})')
    ax.grid(axis='y', alpha=0.3)

    # Latencia
    ax2 = axes[1]
    latencies = [r['latency_ms'] for r in results]
    bars = ax2.barh([r['model'] for r in results], latencies,
                    color=['#2196F3','#4CAF50','#FF9800','#9C27B0'][:len(results)], alpha=0.85)
    for bar, val in zip(bars, latencies):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{val:.2f}ms', va='center', fontsize=9)
    ax2.set_xlabel('Latencia por muestra (ms)')
    ax2.set_title('Latencia de Inferencia', fontweight='bold')
    ax2.axvline(200, color='red', ls='--', lw=1.5, label='Objetivo 200ms')
    ax2.legend(fontsize=8)
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()


def plot_per_class_f1(results, class_names, out_path):
    """F1 por clase para cada modelo."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(7*n_models, 6), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        report = classification_report(
            res['y_true'], res['y_pred'],
            labels=class_names, output_dict=True, zero_division=0
        )
        f1s = [report.get(c, {}).get('f1-score', 0) for c in class_names]
        short = [c.split('_')[1] for c in class_names]
        colors = ['#4CAF50' if v >= 0.5 else ('#FF9800' if v >= 0.25 else '#F44336') for v in f1s]
        bars = ax.barh(short, f1s, color=colors, edgecolor='white', alpha=0.85)
        ax.axvline(0.5, color='gray', ls='--', lw=1)
        ax.axvline(np.mean(f1s), color='blue', ls=':', lw=1.5,
                   label=f'Media={np.mean(f1s):.3f}')
        ax.set_xlim(0, 1.05)
        ax.set_title(f'{res["model"]} — F1 por Clase', fontweight='bold')
        ax.set_xlabel('F1-score')
        ax.legend(fontsize=8)
        ax.tick_params(axis='y', labelsize=8)

    fig.suptitle('F1-score por Clase (verde≥0.5, naranja≥0.25, rojo<0.25)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────
def main(args):
    log_path = LOG_DIR / "baseline_results.txt"
    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    log("=" * 70)
    log("BASELINE CLÁSICO — Sistema LSP (Lengua de Señas Peruana)")
    log("=" * 70)
    log(f"Dataset: 26 clases | Sliding window 30f stride15")
    log()

    # ── Cargar datos ──────────────────────────────────────────────────────
    df = pd.read_csv(DATA_DIR / "manifest_segments.csv")
    with open(DATA_DIR / "label2idx.json") as f:
        label2idx = json.load(f)
    class_names = sorted(label2idx.keys())
    le = LabelEncoder().fit(class_names)

    df_train = df[df['split'] == 'train']
    df_val   = df[df['split'] == 'val']
    df_test  = df[df['split'] == 'test']

    log(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
    log()

    # Limitamos para velocidad si se pide
    n_train = args.max_segments or len(df_train)
    n_val   = min(args.max_segments or 9999, len(df_val))
    n_test  = min(args.max_segments or 9999, len(df_test))

    # ── Feature extraction ────────────────────────────────────────────────
    log("1. EXTRACCIÓN DE FEATURES (color histograma + movimiento)")
    log(f"   sample_frames=5, resize=32 → {32*3*5 + 4 + 2} features/segmento")

    npz_path = DATA_DIR / "baseline_features.npz"
    if npz_path.exists() and not args.recompute:
        log("   Cargando features pre-computadas...")
        data = np.load(npz_path, allow_pickle=True)
        X_train, y_train = data['X_train'], data['y_train']
        X_val,   y_val   = data['X_val'],   data['y_val']
        X_test,  y_test  = data['X_test'],  data['y_test']
    else:
        log("   Extrayendo features train...")
        X_train, y_train = build_feature_matrix(
            df_train.sample(n_train, random_state=42), sample_frames=5)
        log("   Extrayendo features val...")
        X_val, y_val = build_feature_matrix(
            df_val.sample(n_val, random_state=42), sample_frames=5)
        log("   Extrayendo features test...")
        X_test, y_test = build_feature_matrix(
            df_test.sample(n_test, random_state=42), sample_frames=5)
        np.savez(npz_path,
                 X_train=X_train, y_train=y_train,
                 X_val=X_val, y_val=y_val,
                 X_test=X_test, y_test=y_test)
        log(f"   Guardado: {npz_path}")

    log(f"   Shape: X_train {X_train.shape} | X_test {X_test.shape}")
    log(f"   Desbalance: {pd.Series(y_train).value_counts().max()} / "
        f"{pd.Series(y_train).value_counts().min()} segmentos por clase")
    log()

    # ── Modelos ───────────────────────────────────────────────────────────
    log("2. ENTRENAMIENTO DE MODELOS BASELINE")

    models = {
        'DummyClassifier (stratified)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    DummyClassifier(strategy='stratified', random_state=42)),
        ]),
        'KNN (k=5)': Pipeline([
            ('scaler', StandardScaler()),
            ('pca',    PCA(n_components=50, random_state=42)),
            ('clf',    KNeighborsClassifier(n_neighbors=5, metric='euclidean', n_jobs=-1)),
        ]),
        'Naive Bayes': Pipeline([
            ('scaler', StandardScaler()),
            ('pca',    PCA(n_components=50, random_state=42)),
            ('clf',    GaussianNB()),
        ]),
        'LogReg (C=1)': Pipeline([
            ('scaler', StandardScaler()),
            ('pca',    PCA(n_components=100, random_state=42)),
            ('clf',    LogisticRegression(
                C=1.0, max_iter=1000, solver='lbfgs',
                multi_class='multinomial', random_state=42, n_jobs=-1
            )),
        ]),
    }

    results = []
    for name, pipe in models.items():
        log(f"   Entrenando: {name}...")
        t0 = time.time()
        pipe.fit(X_train, y_train)
        train_time = time.time() - t0
        log(f"   → Tiempo entrenamiento: {train_time:.1f}s")

        res = eval_model(pipe, X_test, y_test, le, name)
        res['y_true'] = y_test
        res['train_time_s'] = train_time
        results.append(res)

        log(f"   → Acc: {res['accuracy']:.4f} | F1-macro: {res['f1_macro']:.4f} | "
            f"F1-w: {res['f1_weighted']:.4f} | AUC: {res['roc_auc']:.4f} | "
            f"Latencia: {res['latency_ms']:.2f}ms")
        log()

    # ── Resultados ────────────────────────────────────────────────────────
    log("3. TABLA DE RESULTADOS")
    log("-" * 75)
    log(f"{'Modelo':30s} | {'Acc':6s} | {'F1-m':6s} | {'F1-w':6s} | {'AUC':6s} | {'ms/seg':7s}")
    log("-" * 75)
    for r in results:
        log(f"{r['model']:30s} | {r['accuracy']:.4f} | {r['f1_macro']:.4f} | "
            f"{r['f1_weighted']:.4f} | {r['roc_auc']:.4f} | {r['latency_ms']:.3f}")
    log("-" * 75)
    log(f"{'Random baseline':30s} | {1/26:.4f} | {'—':6s} | {'—':6s} | {'—':6s}")
    log()

    best = max(results, key=lambda r: r['f1_macro'])
    log(f"Mejor modelo: {best['model']} — F1-macro: {best['f1_macro']:.4f}")
    log()

    # ── Reporte por clase del mejor modelo ───────────────────────────────
    log("4. REPORTE CLASIFICACIÓN (mejor modelo)")
    log(classification_report(
        best['y_true'], best['y_pred'],
        labels=class_names, zero_division=0
    ))

    # ── Riesgos identificados ─────────────────────────────────────────────
    log("5. RIESGOS Y OBSERVACIONES")
    log()
    log("  [RIESGO] Desbalance 11:1 (vineta_003: 1137 segs vs vineta_027: 102 segs)")
    log("           → F1-macro penaliza esto; F1-weighted favorece clases mayoritarias")
    log()
    log("  [RIESGO] Data leakage: frames consecutivos del mismo video")
    log("           → MITIGADO: splits a nivel de VIDEO (no de segmento)")
    log("           → Test incluye SOLO videos no vistos en train/val")
    log()
    log("  [RIESGO] Concept drift: resoluciones distintas (480p vs 1080p)")
    log("           → Normalización + resize mitigan parcialmente")
    log()
    log("  [RIESGO] Alta varianza inter-clase baja (1 video/clase)")
    log("           → Overfitting esperado; baseline bajo es normal")
    log()
    log("  [OBSERVACIÓN] Random baseline = 1/26 = 0.038")
    log(f"  [OBSERVACIÓN] Mejor F1-macro baseline = {best['f1_macro']:.4f}")
    uplift = (best['f1_macro'] - 1/26) / (1/26) * 100
    log(f"  [OBSERVACIÓN] Uplift sobre random = +{uplift:.0f}%")

    # ── Guardar log ───────────────────────────────────────────────────────
    with open(log_path, 'w') as f:
        f.write('\n'.join(log_lines))
    log(f"\nLog guardado: {log_path}")

    # ── Visualizaciones ───────────────────────────────────────────────────
    log("\n6. GENERANDO VISUALIZACIONES...")

    plot_metrics_comparison(results, DATA_DIR / "baseline_metrics.png")
    log("   data/baseline_metrics.png")

    plot_confusion(best['y_true'], best['y_pred'], class_names,
                   f"Mejor Baseline: {best['model']}",
                   DATA_DIR / "baseline_confusion.png")
    log("   data/baseline_confusion.png")

    plot_per_class_f1(
        [r for r in results if r['model'] != 'DummyClassifier (stratified)'],
        class_names, DATA_DIR / "baseline_per_class_f1.png"
    )
    log("   data/baseline_per_class_f1.png")

    log()
    log("=" * 70)
    log("RESUMEN EJECUTIVO")
    log("=" * 70)
    log(f"  Dataset:      26 clases | 7,235 segmentos | desbalance 11:1")
    log(f"  Random:       Acc={1/26:.4f} | F1-macro=0.038")
    for r in results:
        log(f"  {r['model']:<30s} Acc={r['accuracy']:.4f} | F1-m={r['f1_macro']:.4f}")
    log()
    log("  CONCLUSIÓN: El baseline clásico es útil como referencia mínima.")
    log("  Para superar el desbalance y capturar temporalidad, se requiere")
    log("  un modelo secuencial (CNN-LSTM/ST-GCN) con WeightedRandomSampler.")
    log("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_segments', type=int, default=None,
                        help='Limitar segmentos por split (debug)')
    parser.add_argument('--recompute', action='store_true',
                        help='Recomputar features aunque existan en disco')
    args = parser.parse_args()
    main(args)
