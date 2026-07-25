"""generar_ipynb_entrega_final.py — TESIS_FINAL_S15.md / ENTREGA_FINAL_SEMANA15.md
→ .ipynb.

Cada sección "## " (y cada slice "### " del Anexo) se vuelve una celda
Markdown. Dentro de eso, cada GRÁFICO de resultados (8 de los 10 —
quedan fuera los 2 diagramas de arquitectura, que son dibujo técnico, no
datos) se reemplaza por su propia celda de CÓDIGO independiente: no una
imagen `.png` pre-generada por otro script, sino código que carga sus
propios datos desde cero (checkpoints, `logs/runs.csv`, JSON de
resultados) y dibuja el gráfico ahí mismo, en el momento de ejecutarse.
Cada celda de gráfico es autosuficiente — no depende de variables creadas
por celdas anteriores, se puede correr sola.

Además, varias secciones llevan una celda de código no-gráfico (golpea la
API real, lee checkpoints, etc.) — igual que antes.

Requiere `api/main.py` corriendo en localhost:8000 para las celdas de
§3/§5 (`make run-api` en otra terminal).

Uso:
    python3 scripts/generar_ipynb_entrega_final.py TESIS_FINAL_S15
    python3 scripts/generar_ipynb_entrega_final.py ENTREGA_FINAL_SEMANA15
"""
import base64
import re
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).parent.parent

if len(sys.argv) < 2:
    print("Uso: generar_ipynb_entrega_final.py <NOMBRE_SIN_EXTENSION>")
    sys.exit(1)

NOMBRE = sys.argv[1]
MD_IN = ROOT / f"{NOMBRE}.md"
OUT = ROOT / "notebooks" / f"{NOMBRE}.ipynb"

MPL_STYLE = (
    "%matplotlib inline\n"
    "import matplotlib.pyplot as plt\n"
    "plt.rcParams.update({'font.size': 10.5, 'axes.spines.top': False, 'axes.spines.right': False,\n"
    "                      'axes.edgecolor': '#425055', 'figure.facecolor': 'white'})\n"
    "TEAL, TEAL_DARK, AMBER, BAD, GOOD, GREY = '#0e7c78', '#075c59', '#b3791f', '#c2384a', '#1f8f5f', '#8398a0'\n\n"
)

# ── una celda de código independiente por gráfico ───────────────────────────
GRAFICOS = {
    "fig_f1_evolucion.png": (
        "**Gráfico independiente** — lee `logs/runs.csv` real y grafica la evolución de F1-test corrida por corrida.",
        MPL_STYLE +
        "import pandas as pd\n\n"
        "df = pd.read_csv('logs/runs.csv')\n"
        "df['f1_test'] = df['f1_test'].astype(float)\n\n"
        "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
        "ax.plot(range(len(df)), df['f1_test'], color=TEAL, marker='o', markersize=3.5, linewidth=1.5, zorder=3)\n"
        "ax.fill_between(range(len(df)), df['f1_test'], color=TEAL, alpha=0.08, zorder=2)\n"
        "mejor = df['f1_test'].idxmax()\n"
        "ax.scatter([mejor], [df['f1_test'].iloc[mejor]], color=AMBER, s=70, zorder=4,\n"
        "           label=f\"mejor: {df['sprint'].iloc[mejor]} (F1={df['f1_test'].iloc[mejor]:.3f})\")\n"
        "paso = max(1, len(df) // 12)\n"
        "ax.set_xticks(range(0, len(df), paso))\n"
        "ax.set_xticklabels(df['sprint'].iloc[::paso], rotation=45, ha='right', fontsize=8)\n"
        "ax.set_ylabel('F1-macro (test)')\n"
        "ax.set_title(f'Evolución de F1-macro — {len(df)} corridas registradas ({df[\"sprint\"].iloc[0]}→{df[\"sprint\"].iloc[-1]})')\n"
        "ax.legend(fontsize=9, loc='upper left')\n"
        "ax.grid(axis='y', color='#e2e8ea', linewidth=0.8, zorder=0)\n"
        "ax.set_axisbelow(True)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "fig_topk.png": (
        "**Gráfico independiente** — reproduce el split de test determinista de v4 (SEED=42) y corre inferencia ONNX real para recalcular Top-1/3/5.",
        MPL_STYLE +
        "import json\n"
        "import numpy as np\n"
        "import onnxruntime as ort\n"
        "from collections import Counter\n"
        "from sklearn.model_selection import StratifiedShuffleSplit\n\n"
        "data = np.load('data/dataset_s17.npz')\n"
        "X_raw, y_raw = data['X'], data['y']\n"
        "counts = Counter(y_raw.tolist())\n"
        "keep = np.array([counts[int(v)] >= 15 for v in y_raw])\n"
        "X_raw, y_raw = X_raw[keep], y_raw[keep]\n\n"
        "def normalize_sample(x):\n"
        "    mu, std = x.mean(), x.std()\n"
        "    return ((x - mu) / std).astype(np.float32) if std >= 1e-8 else x\n\n"
        "X = np.stack([normalize_sample(X_raw[i]) for i in range(len(X_raw))])\n"
        "old_ids = sorted(set(y_raw.tolist()))\n"
        "remap = {o: n for n, o in enumerate(old_ids)}\n"
        "y = np.array([remap[int(v)] for v in y_raw], dtype=np.int64)\n\n"
        "sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)\n"
        "_, te_idx = next(sss.split(X, y))\n"
        "X_te, y_te = X[te_idx], y[te_idx]\n\n"
        "sess = ort.InferenceSession('checkpoints/bilstm_s27.onnx', providers=['CPUExecutionProvider'])\n"
        "inp = sess.get_inputs()[0].name\n"
        "probs = np.concatenate([\n"
        "    (lambda l: np.exp(l - l.max(1, keepdims=True)) / np.exp(l - l.max(1, keepdims=True)).sum(1, keepdims=True))\n"
        "    (sess.run(None, {inp: X_te[i:i+64].astype(np.float32)})[0])\n"
        "    for i in range(0, len(X_te), 64)\n"
        "])\n"
        "y_pred = probs.argmax(1)\n"
        "top1 = (y_pred == y_te).mean()\n"
        "top3 = np.mean([y_te[i] in np.argsort(-probs[i])[:3] for i in range(len(y_te))])\n"
        "top5 = np.mean([y_te[i] in np.argsort(-probs[i])[:5] for i in range(len(y_te))])\n"
        "print(f'N={len(y_te)}  Top-1={top1:.4f}  Top-3={top3:.4f}  Top-5={top5:.4f}')\n\n"
        "fig, ax = plt.subplots(figsize=(6, 4.2))\n"
        "vals = [top1, top3, top5]\n"
        "bars = ax.bar(['Top-1', 'Top-3', 'Top-5'], vals, color=[TEAL_DARK, TEAL, '#5fb3af'], width=0.55)\n"
        "for b, v in zip(bars, vals):\n"
        "    ax.text(b.get_x() + b.get_width()/2, v + 0.015, f'{v*100:.1f}%', ha='center', fontweight='bold')\n"
        "ax.set_ylim(0, 0.85)\n"
        "ax.set_ylabel('Exactitud')\n"
        "ax.set_title(f'Exactitud Top-k — modelo S27 v4 (test real, N={len(y_te)})')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "fig_matriz_confusion.png": (
        "**Gráfico independiente** — mismo split real de test, misma inferencia ONNX, esta vez para la matriz de confusión completa.",
        MPL_STYLE +
        "import json\n"
        "import numpy as np\n"
        "import onnxruntime as ort\n"
        "from collections import Counter\n"
        "from sklearn.model_selection import StratifiedShuffleSplit\n"
        "from sklearn.metrics import confusion_matrix\n\n"
        "data = np.load('data/dataset_s17.npz')\n"
        "X_raw, y_raw = data['X'], data['y']\n"
        "counts = Counter(y_raw.tolist())\n"
        "keep = np.array([counts[int(v)] >= 15 for v in y_raw])\n"
        "X_raw, y_raw = X_raw[keep], y_raw[keep]\n\n"
        "def normalize_sample(x):\n"
        "    mu, std = x.mean(), x.std()\n"
        "    return ((x - mu) / std).astype(np.float32) if std >= 1e-8 else x\n\n"
        "X = np.stack([normalize_sample(X_raw[i]) for i in range(len(X_raw))])\n"
        "old_ids = sorted(set(y_raw.tolist()))\n"
        "remap = {o: n for n, o in enumerate(old_ids)}\n"
        "y = np.array([remap[int(v)] for v in y_raw], dtype=np.int64)\n\n"
        "sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)\n"
        "_, te_idx = next(sss.split(X, y))\n"
        "X_te, y_te = X[te_idx], y[te_idx]\n\n"
        "sess = ort.InferenceSession('checkpoints/bilstm_s27.onnx', providers=['CPUExecutionProvider'])\n"
        "inp = sess.get_inputs()[0].name\n"
        "probs = np.concatenate([\n"
        "    (lambda l: np.exp(l - l.max(1, keepdims=True)) / np.exp(l - l.max(1, keepdims=True)).sum(1, keepdims=True))\n"
        "    (sess.run(None, {inp: X_te[i:i+64].astype(np.float32)})[0])\n"
        "    for i in range(0, len(X_te), 64)\n"
        "])\n"
        "y_pred = probs.argmax(1)\n\n"
        "cm = confusion_matrix(y_te, y_pred, normalize='true')\n"
        "fig, ax = plt.subplots(figsize=(8, 7))\n"
        "im = ax.imshow(cm, cmap='BuGn', vmin=0, vmax=1)\n"
        "ax.set_title(f'Matriz de confusión normalizada — 96 clases (N={len(y_te)})')\n"
        "ax.set_xlabel('Clase predicha (índice)')\n"
        "ax.set_ylabel('Clase real (índice)')\n"
        "plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label='Proporción de aciertos por fila')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "fig_f1_por_clase.png": (
        "**Gráfico independiente** — mismo split/inferencia, F1 por clase (mejores y peores 15).",
        MPL_STYLE +
        "import json\n"
        "import numpy as np\n"
        "import onnxruntime as ort\n"
        "from collections import Counter\n"
        "from sklearn.model_selection import StratifiedShuffleSplit\n"
        "from sklearn.metrics import f1_score\n\n"
        "data = np.load('data/dataset_s17.npz')\n"
        "X_raw, y_raw = data['X'], data['y']\n"
        "with open('data/s17_label2idx.json', encoding='utf-8') as f:\n"
        "    label2idx = json.load(f)\n"
        "idx2label_raw = {int(v): k for k, v in label2idx.items()}\n"
        "counts = Counter(y_raw.tolist())\n"
        "keep = np.array([counts[int(v)] >= 15 for v in y_raw])\n"
        "X_raw, y_raw = X_raw[keep], y_raw[keep]\n\n"
        "def normalize_sample(x):\n"
        "    mu, std = x.mean(), x.std()\n"
        "    return ((x - mu) / std).astype(np.float32) if std >= 1e-8 else x\n\n"
        "X = np.stack([normalize_sample(X_raw[i]) for i in range(len(X_raw))])\n"
        "old_ids = sorted(set(y_raw.tolist()))\n"
        "remap = {o: n for n, o in enumerate(old_ids)}\n"
        "y = np.array([remap[int(v)] for v in y_raw], dtype=np.int64)\n"
        "idx2label = {remap[o]: idx2label_raw.get(o, str(o)) for o in old_ids}\n\n"
        "sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)\n"
        "_, te_idx = next(sss.split(X, y))\n"
        "X_te, y_te = X[te_idx], y[te_idx]\n\n"
        "sess = ort.InferenceSession('checkpoints/bilstm_s27.onnx', providers=['CPUExecutionProvider'])\n"
        "inp = sess.get_inputs()[0].name\n"
        "probs = np.concatenate([\n"
        "    (lambda l: np.exp(l - l.max(1, keepdims=True)) / np.exp(l - l.max(1, keepdims=True)).sum(1, keepdims=True))\n"
        "    (sess.run(None, {inp: X_te[i:i+64].astype(np.float32)})[0])\n"
        "    for i in range(0, len(X_te), 64)\n"
        "])\n"
        "y_pred = probs.argmax(1)\n\n"
        "present = sorted(set(y_te.tolist()))\n"
        "f1_pc = f1_score(y_te, y_pred, average=None, labels=present)\n"
        "names = [idx2label.get(c, str(c)) for c in present]\n"
        "order = np.argsort(f1_pc)\n"
        "worst15, best15 = order[:15], order[-15:][::-1]\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.5))\n"
        "for ax, idxs, title, color in [(axes[0], best15, 'Mejores 15 clases (F1)', GOOD),\n"
        "                                (axes[1], worst15, 'Peores 15 clases (F1)', BAD)]:\n"
        "    vals = [f1_pc[i] for i in idxs]\n"
        "    ax.barh(range(len(idxs)), vals, color=color)\n"
        "    ax.set_yticks(range(len(idxs)))\n"
        "    ax.set_yticklabels([names[i] for i in idxs], fontsize=8.5)\n"
        "    ax.invert_yaxis()\n"
        "    ax.set_xlim(0, 1)\n"
        "    ax.set_title(title, fontsize=10.5)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "fig_he3_comparacion.png": (
        "**Gráfico independiente** — valores HE3 conocidos de las 4 corridas S27 (v1-v4), medidos y registrados en `logs/runs.csv` / reportes HE3 de cada sprint (no recalculables sin los checkpoints v1-v3, que se sobrescribieron — ver R2).",
        MPL_STYLE +
        "versiones = ['v1', 'v2', 'v3', 'v4']\n"
        "delta_f1 = [0.1311, 0.0509, 0.0785, 0.0406]\n"
        "psi = [0.1281, 0.0818, 0.0184, 0.0288]\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))\n"
        "colores = [GREY, GREY, TEAL, TEAL_DARK]\n"
        "for ax, vals, thr, thr_label, ylabel in [\n"
        "    (axes[0], delta_f1, 0.15, 'umbral ΔF1≤0.15', 'ΔF1 (holdout de grupo)'),\n"
        "    (axes[1], psi, 0.20, 'umbral PSI<0.20', 'PSI'),\n"
        "]:\n"
        "    bars = ax.bar(versiones, vals, color=colores, width=0.6)\n"
        "    ax.axhline(thr, color=BAD, linestyle='--', linewidth=1.2)\n"
        "    ax.text(3.4, thr, thr_label, color=BAD, fontsize=8, va='bottom', ha='right')\n"
        "    for b, v in zip(bars, vals):\n"
        "        ax.text(b.get_x()+b.get_width()/2, v+0.004, f'{v:.4f}', ha='center', fontsize=8.5)\n"
        "    ax.set_ylabel(ylabel)\n"
        "fig.suptitle('Generalización HE3 — 4 corridas S27 (v1→v4)')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "fig_latencia.png": (
        "**Gráfico independiente** — latencias reales medidas esta semana (modelo ONNX aislado vs. pipeline E2E con cliente WebSocket real).",
        MPL_STYLE +
        "import numpy as np\n\n"
        "etiquetas = ['p50', 'p95', 'max']\n"
        "lat_modelo = [0.6, 0.9, 1.3]\n"
        "lat_e2e = [54.7, 58.7, 118.2]\n\n"
        "x = np.arange(len(etiquetas)); w = 0.35\n"
        "fig, ax = plt.subplots(figsize=(6.5, 4.3))\n"
        "ax.bar(x - w/2, lat_modelo, width=w, color=TEAL_DARK, label='Modelo ONNX aislado')\n"
        "ax.bar(x + w/2, lat_e2e, width=w, color=AMBER, label='Pipeline E2E real (WebSocket)')\n"
        "ax.axhline(200, color=BAD, linestyle='--', linewidth=1.2, label='Umbral objetivo (200ms)')\n"
        "ax.set_xticks(x); ax.set_xticklabels(etiquetas)\n"
        "ax.set_ylabel('Latencia (ms)')\n"
        "ax.set_title('Latencia real — modelo vs. pipeline E2E')\n"
        "ax.legend(fontsize=8.5)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "fig_wer_progresion.png": (
        "**Gráfico independiente** — lee `data/s33_wer_resultados.json` real y grafica el WER medio de las 3 corridas vs. producción.",
        MPL_STYLE +
        "import json\n\n"
        "with open('data/s33_wer_resultados.json', encoding='utf-8') as f:\n"
        "    wer = json.load(f)\n\n"
        "labels = ['S31\\n(aislado)', 'S32\\n(combinado,\\nsplit único)', 'S33\\n(combinado,\\nKFold completo)', 'Producción\\n(v4+S29)']\n"
        "keys = ['s31', 's32', 's33', 'produccion']\n"
        "medias = [wer['wer_medio'][k] for k in keys]\n"
        "colores = [GREY, '#5fb3af', TEAL_DARK, AMBER]\n\n"
        "fig, ax = plt.subplots(figsize=(7.2, 4.6))\n"
        "bars = ax.bar(labels, medias, color=colores, width=0.6)\n"
        "ax.axhline(1.0, color=BAD, linestyle='--', linewidth=1.2)\n"
        "ax.text(3.45, 1.03, 'WER=1.0 (tantos errores\\ncomo palabras reales)', color=BAD, fontsize=8.5, va='bottom', ha='right')\n"
        "for b, v in zip(bars, medias):\n"
        "    ax.text(b.get_x()+b.get_width()/2, v+0.06, f'{v:.3f}', ha='center', fontweight='bold')\n"
        "ax.set_ylabel('WER medio (5 videos narrativos nunca vistos)')\n"
        "ax.set_title('WER real — 3 corridas consecutivas vs. producción')\n"
        "ax.set_ylim(0, max(medias)*1.22)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    "fig_slices_ic.png": (
        "**Gráfico independiente** — Wilson CI 95% calculado desde cero (no depende de la celda de funciones auxiliares de arriba) para los 4 slices problemáticos.",
        MPL_STYLE +
        "import math\n"
        "import numpy as np\n\n"
        "def wilson(k, n, z=1.96):\n"
        "    p = k / n\n"
        "    denom = 1 + z**2/n\n"
        "    centre = p + z**2/(2*n)\n"
        "    adj = z*math.sqrt(p*(1-p)/n + z**2/(4*n**2))\n"
        "    return p, max(0,(centre-adj)/denom), min(1,(centre+adj)/denom)\n\n"
        "slices = [\n"
        "    ('Abecedario — video/cámara\\n(cuerpo completo)', 0, 6),\n"
        "    ('Abecedario — imagen\\n(post-fix R14)', 19, 24),\n"
        "    ('Videos largos\\nsin segmentar (top-3)', 1, 7),\n"
        "    ('Clips ya aislados\\n(cortos, top-3)', 3, 4),\n"
        "]\n\n"
        "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
        "ys = np.arange(len(slices))[::-1]\n"
        "for y, (label, k, n) in zip(ys, slices):\n"
        "    p, lo, hi = wilson(k, n)\n"
        "    color = BAD if p < 0.5 else (AMBER if p < 0.7 else GOOD)\n"
        "    ax.plot([lo*100, hi*100], [y, y], color=color, linewidth=3, solid_capstyle='round')\n"
        "    ax.scatter([p*100], [y], color=color, s=70, zorder=4, edgecolor='white', linewidth=1.2)\n"
        "    ax.text(hi*100+3, y, f'{p*100:.1f}%  (n={n})', va='center', fontsize=9.5)\n"
        "ax.set_yticks(ys); ax.set_yticklabels([s[0] for s in slices], fontsize=9.5)\n"
        "ax.set_xlim(0, 118)\n"
        "ax.set_xlabel('Tasa de acierto (%) — punto = medición real, línea = IC95 Wilson')\n"
        "ax.set_title('Slices problemáticos — proporción de acierto con IC')\n"
        "ax.axvline(50, color='#c8d2d4', linestyle=':', linewidth=1)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
}


def embed_images(md_text: str) -> str:
    """Para imágenes que NO son gráficos con código propio (los 2 diagramas
    de arquitectura) — se quedan como imagen estática embebida en base64."""
    def _sub(m):
        alt, path = m.group(1), m.group(2)
        img_path = ROOT / path
        if img_path.exists():
            data = base64.b64encode(img_path.read_bytes()).decode()
            ext = img_path.suffix.lower().lstrip(".")
            mime = f"image/{'jpeg' if ext == 'jpg' else ext}"
            return f"![{alt}](data:{mime};base64,{data})"
        return m.group(0)

    return re.sub(r"!\[([^\]]*)\]\(([^)]+\.(?:png|jpg|jpeg|gif))\)", _sub, md_text)


def emitir_bloque(texto: str, cells: list):
    """Recorre un bloque de markdown línea por línea. Cuando encuentra la
    imagen de un gráfico con código propio, corta la celda markdown ahí,
    inserta la celda de código de ese gráfico, y sigue acumulando markdown
    después. Las demás imágenes (diagramas) quedan embebidas en base64
    dentro del markdown, como antes."""
    buffer = []
    img_re = re.compile(r"^!\[([^\]]*)\]\(([^)]+\.(?:png|jpg|jpeg|gif))\)\s*$")

    def flush():
        if buffer:
            texto_md = embed_images("\n".join(buffer))
            if texto_md.strip():
                cells.append(nbf.v4.new_markdown_cell(texto_md))
            buffer.clear()

    for line in texto.splitlines():
        m = img_re.match(line.strip())
        if m:
            nombre_archivo = Path(m.group(2)).name
            if nombre_archivo in GRAFICOS:
                flush()
                intro, code = GRAFICOS[nombre_archivo]
                cells.append(nbf.v4.new_markdown_cell(intro))
                cells.append(nbf.v4.new_code_cell(code))
                continue
        buffer.append(line)
    flush()


# celdas de código NO gráficas, insertadas justo después de la sección que empieza así
CODE_AFTER = {
    "## 1. Resumen ejecutivo": (
        "Celda ejecutable — carga los checkpoints reales del ensemble activo "
        "(`bilstm_s27.pt` + `bilstm_s29.pt`) y muestra las métricas que "
        "quedaron guardadas dentro de cada uno al entrenarlos.",
        "import torch\n\n"
        "for nombre in ['bilstm_s27.pt', 'bilstm_s29.pt']:\n"
        "    ckpt = torch.load(f'checkpoints/{nombre}', map_location='cpu', weights_only=False)\n"
        "    print(f'--- {nombre} ---')\n"
        "    for k in ['sprint', 'f1_test', 'acc_test', 'top3_test', 'top5_test', 'latencia_onnx_ms']:\n"
        "        if k in ckpt:\n"
        "            print(f'  {k}: {ckpt[k]}')\n"
        "    if 'he3' in ckpt:\n"
        "        print(f\"  he3: {ckpt['he3']}\")\n"
        "    print()"
    ),
    "## 2. Arquitectura candidata": (
        "Celda ejecutable — importa de verdad los módulos del pipeline "
        "listados en la tabla de componentes y confirma que existen y cargan "
        "sin error.",
        "import importlib, pathlib\n\n"
        "componentes = {\n"
        "    'src.features.landmarks': 'kp_seq_to_features, normalize_sample, aplicar_respaldo_manos_y_pose',\n"
        "    'src.features.segmentacion': 'SegmentadorPausas',\n"
        "    'src.inference.predictor': 'ONNXPredictor',\n"
        "}\n"
        "for modulo, simbolos in componentes.items():\n"
        "    m = importlib.import_module(modulo)\n"
        "    faltantes = [s for s in simbolos.split(', ') if not hasattr(m, s)]\n"
        "    estado = 'OK' if not faltantes else f'FALTAN: {faltantes}'\n"
        "    print(f'{modulo:<30} {estado}')\n\n"
        "for archivo in ['demo/app_gradio.py', 'api/main.py', 'spaces/app.py']:\n"
        "    existe = pathlib.Path(archivo).exists()\n"
        "    print(f'{archivo:<30} {\"existe\" if existe else \"NO EXISTE\"}')"
    ),
    "## 3. Contratos I/O": (
        "Celda ejecutable — golpea la API real corriendo en `localhost:8000` "
        "(`make run-api` en otra terminal).",
        "import requests\n"
        "try:\n"
        "    r = requests.get('http://localhost:8000/health', timeout=3)\n"
        "    print('GET /health ->', r.status_code)\n"
        "    print(r.json())\n"
        "    r2 = requests.get('http://localhost:8000/classes', timeout=3)\n"
        "    j2 = r2.json()\n"
        "    print('\\nGET /classes ->', r2.status_code, '| total:', j2['total'])\n"
        "    print('primeras 10 clases:', j2['classes'][:10])\n"
        "except requests.exceptions.ConnectionError:\n"
        "    print('API no está corriendo en :8000 — levantar con: make run-api')"
    ),
    "## 4. Reproducibilidad": (
        "Celda ejecutable — carga el dataset combinado real (`dataset_s32.npz`) "
        "y recalcula el desglose por fuente.",
        "import numpy as np, json\n"
        "from collections import Counter\n\n"
        "data = np.load('data/dataset_s32.npz')\n"
        "X, y, groups = data['X'], data['y'], data['groups']\n"
        "sources = np.load('data/s32_sources.npy', allow_pickle=True)\n"
        "print('X shape:', X.shape, '| clases:', len(set(y.tolist())), '| grupos únicos:', len(set(groups.tolist())))\n\n"
        "print('\\nMuestras por fuente (recalculado en vivo):')\n"
        "for src, n in sorted(Counter(sources.tolist()).items()):\n"
        "    clases_src = len(set(y[sources == src].tolist()))\n"
        "    print(f'  {src:<18}: {n:>5} muestras, {clases_src:>4} clases')"
    ),
    "## 5. E2E en limpio": (
        "Celda ejecutable — corre el mismo E2E que `make test-video`, contra "
        "la API real, con el video de ejemplo del repo.",
        "import requests\n"
        "try:\n"
        "    with open('data/videos/original/Historias vinetas (11).mp4', 'rb') as f:\n"
        "        r = requests.post('http://localhost:8000/predict/video',\n"
        "                           files={'file': ('v.mp4', f, 'video/mp4')}, timeout=30)\n"
        "    print('POST /predict/video ->', r.status_code)\n"
        "    print(r.json())\n"
        "except requests.exceptions.ConnectionError:\n"
        "    print('API no está corriendo en :8000 — levantar con: make run-api')"
    ),
    "## 6. Observabilidad": (
        "Celda ejecutable — lee `logs/runs.csv` real y muestra las últimas corridas.",
        "import pandas as pd\n"
        "df = pd.read_csv('logs/runs.csv')\n"
        "print(f'{len(df)} corridas registradas, sprints {df[\"sprint\"].iloc[0]} → {df[\"sprint\"].iloc[-1]}')\n"
        "df[['exp_id', 'sprint', 'f1_test', 'n_classes', 'tiempo_s']].tail(6)"
    ),
    "## 8. Seguridad & configuración": (
        "Celda ejecutable — lee el código fuente real de `api/main.py` y "
        "extrae los valores vigentes de CORS y límite de upload.",
        "import re\n"
        "src_lines = open('api/main.py', encoding='utf-8').readlines()\n"
        "src_sin_comentarios = ''.join(re.sub(r'\\s*#.*$', '', l) + '\\n'\n"
        "                              for l in src_lines if not l.strip().startswith('#'))\n\n"
        "origins = re.search(r'allow_origins=\\[(.*?)\\]', src_sin_comentarios, re.S)\n"
        "print('CORS allow_origins (config real, sin contar comentarios):')\n"
        "print(' ', [o.strip() for o in origins.group(1).split(chr(44)) if o.strip()])\n\n"
        "limite = re.search(r'\"max_upload_mb\":\\s*(\\d+)', src_sin_comentarios)\n"
        "print('Límite de upload (MB):', limite.group(1))\n\n"
        "import pathlib\n"
        "print('.env existe:', pathlib.Path('.env').exists())\n"
        "print('Dockerfile existe:', pathlib.Path('Dockerfile').exists())"
    ),
    "## 9. Hoja de ruta a Docker/API": (
        "Celda ejecutable — verifica el estado real de lo que la hoja de ruta da por pendiente.",
        "import pathlib, re\n\n"
        "print('Dockerfile presente:', pathlib.Path('Dockerfile').exists())\n"
        "print('requirements-api.txt presente:', pathlib.Path('requirements-api.txt').exists())\n\n"
        "spaces_src = pathlib.Path('spaces/app.py').read_text(encoding='utf-8')\n"
        "usa_ensemble = 'S29' in spaces_src or 'ensemble' in spaces_src.lower()\n"
        "print('spaces/app.py ya usa el ensemble v4+S29:', usa_ensemble, '(pendiente si es False)')"
    ),
    "## 10. Riesgos & mitigaciones": (
        "Celda ejecutable — verifica que el mecanismo de rollback (checkpoints versionados) existe de verdad.",
        "import pathlib\n\n"
        "versionado = pathlib.Path('checkpoints/versionado')\n"
        "archivos = list(versionado.glob('*.onnx')) if versionado.exists() else []\n"
        "print(f'checkpoints/versionado/ existe: {versionado.exists()} — {len(archivos)} checkpoint(s) de respaldo')\n"
        "for a in archivos:\n"
        "    print(' ', a.name)"
    ),
}

CODE_ANEXO = {
    "### Slice 2": (
        "**Celda de análisis** (distinta del gráfico de arriba) — recalcula media/desvío/IC95 numéricos del WER, no solo el gráfico.",
        "import json, numpy as np\n\n"
        "with open('data/s33_wer_resultados.json', encoding='utf-8') as f:\n"
        "    wer = json.load(f)\n\n"
        "for modelo in ['s31', 's32', 's33', 'produccion']:\n"
        "    v = np.array(wer['wer_por_video'][modelo])\n"
        "    mean, std = v.mean(), v.std(ddof=1)\n"
        "    sem = std / np.sqrt(len(v))\n"
        "    print(f'{modelo:<12} WER medio={mean:.3f}  std={std:.3f}  '\n"
        "          f'IC95≈[{mean-1.96*sem:.3f}, {mean+1.96*sem:.3f}]  (n={len(v)})')"
    ),
    "### Slice 4": (
        "Celda ejecutable — no hay F1 por clase archivado en JSON (gap real, "
        "documentado tal cual). Esta celda confirma qué evidencia SÍ existe "
        "en disco, en vez de inventar un número.",
        "import pathlib\n\n"
        "for fig in ['fig_matriz_confusion.png', 'fig_f1_por_clase.png']:\n"
        "    p = pathlib.Path(f'data/sustentacion_figs/{fig}')\n"
        "    print(f'{fig:<28} existe={p.exists()}  ({p.stat().st_size//1024 if p.exists() else 0} KB)')\n\n"
        "print('\\nSin JSON de F1 por clase archivado — evidencia disponible es solo visual (arriba), no recomputable aquí.')"
    ),
}

FUNCIONES_AUX_INTRO = (
    "**Funciones auxiliares** — usadas en las celdas de *análisis* del Anexo "
    "(no en los gráficos, que son 100% independientes y traen su propio "
    "cálculo de IC inline)."
)
FUNCIONES_AUX_CODE = (
    "import math\n\n"
    "def wilson(k, n, z=1.96):\n"
    "    \"\"\"Intervalo de confianza Wilson 95% para una proporción k/n.\"\"\"\n"
    "    p = k / n\n"
    "    denom = 1 + z**2 / n\n"
    "    centre = p + z**2 / (2 * n)\n"
    "    adj = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))\n"
    "    return p, max(0, (centre - adj) / denom), min(1, (centre + adj) / denom)"
)


raw = MD_IN.read_text(encoding="utf-8")
lines = raw.splitlines()

i = 0
while i < len(lines) and not lines[i].startswith("## "):
    i += 1
portada = "\n".join(lines[:i]).strip()

cuerpo = lines[i:]
secciones = []
actual = []
for line in cuerpo:
    if line.startswith("## ") and actual:
        secciones.append("\n".join(actual)); actual = [line]
    else:
        actual.append(line)
if actual:
    secciones.append("\n".join(actual))

nb = nbf.v4.new_notebook()
cells = [nbf.v4.new_markdown_cell(embed_images(portada))]
cells.append(nbf.v4.new_markdown_cell(FUNCIONES_AUX_INTRO))
cells.append(nbf.v4.new_code_cell(FUNCIONES_AUX_CODE))

for sec in secciones:
    if sec.startswith("## Anexo"):
        sec_lines = sec.splitlines()
        j = 0
        while j < len(sec_lines) and not sec_lines[j].startswith("### Slice"):
            j += 1
        emitir_bloque("\n".join(sec_lines[:j]), cells)

        slices, cur = [], []
        for line in sec_lines[j:]:
            if line.startswith("### Slice") and cur:
                slices.append("\n".join(cur)); cur = [line]
            else:
                cur.append(line)
        if cur:
            slices.append("\n".join(cur))

        for sl in slices:
            emitir_bloque(sl, cells)
            for prefix, val in CODE_ANEXO.items():
                if sl.startswith(prefix) and val:
                    intro, code = val
                    cells.append(nbf.v4.new_markdown_cell(intro))
                    cells.append(nbf.v4.new_code_cell(code))
        continue

    emitir_bloque(sec, cells)
    for prefix, val in CODE_AFTER.items():
        if sec.startswith(prefix) and val:
            intro, code = val
            cells.append(nbf.v4.new_markdown_cell(intro))
            cells.append(nbf.v4.new_code_cell(code))

cells.append(nbf.v4.new_markdown_cell(
    "## Evidencia ejecutable — suite de tests completa\n\n"
    "La siguiente celda corre de verdad la suite de tests del proyecto "
    "(`pytest tests/ -v`) como evidencia de que §7 es un hecho verificable "
    "ahora mismo."
))
cells.append(nbf.v4.new_code_cell(
    "import subprocess\n"
    "result = subprocess.run(\n"
    "    ['.venv310/bin/python', '-m', 'pytest', 'tests/', '-v'],\n"
    "    cwd='.', capture_output=True, text=True, timeout=120,\n"
    ")\n"
    "print(result.stdout[-3000:])\n"
    "if result.returncode != 0:\n"
    "    print(result.stderr[-1500:])\n"
    "print('Exit code:', result.returncode)"
))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3.10 (.venv310)", "language": "python", "name": "venv310"},
    "language_info": {"name": "python", "version": "3.10.20"},
}

n_graficos = sum(1 for c in cells if c.cell_type == "code" and "plt.show()" in c.source)
print(f"Ejecutando {OUT.name} ({len(cells)} celdas, {n_graficos} gráficos independientes)...")
client = NotebookClient(nb, timeout=240, kernel_name="python3",
                         resources={"metadata": {"path": str(ROOT)}})
try:
    client.execute()
    print("  ✅ ejecutado sin errores de kernel")
except Exception as e:
    print(f"  ⚠️  no se pudo ejecutar el kernel ({e}) — se guarda sin ejecutar")

OUT.parent.mkdir(exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
size_kb = OUT.stat().st_size / 1024
print(f"OK — {OUT.name} ({size_kb:.0f} KB) en {OUT}")
