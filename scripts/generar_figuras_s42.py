"""generar_figuras_s42.py — 2 figuras nuevas para TESIS_DOCUMENTO_FINAL.docx,
con datos reales ya medidos en la sesión de mejora de OE1/OE3 (2026-07-26).
Mismo estilo y paleta que scripts/generar_figuras_entrega_final.py.

  - fig_oe1_metodos.png    : F1-macro por método probado para OE1
                             (vocabulario completo vs. abecedario curado)
  - fig_bootstrap_ks.png   : distribución de p-valores del bootstrap KS
                             (N=200, 500 remuestreos) — validación de OE3
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ks_2samp

ROOT = Path(__file__).parent.parent
OUT = ROOT / "data" / "sustentacion_figs"
OUT.mkdir(exist_ok=True)

TEAL = "#0e7c78"
TEAL_DARK = "#075c59"
AMBER = "#b3791f"
BAD = "#c2384a"
GOOD = "#1f8f5f"
GREY = "#8398a0"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.edgecolor": "#425055",
    "axes.labelcolor": "#182225",
    "text.color": "#182225",
    "xtick.color": "#425055",
    "ytick.color": "#425055",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def fig_oe1_metodos():
    metodos = ["v4\n(base)", "ST-GCN\n(sin tuning)", "S40\n(transfer.\nsolo)",
               "Ensemble\nv4+S40", "Curado\n(abecedario)"]
    f1 = [0.4426, 0.0974, 0.3957, 0.4795, 0.9308]
    colores = [GREY, BAD, AMBER, TEAL, GOOD]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bars = ax.bar(metodos, f1, color=colores, width=0.6, zorder=3)
    ax.axhline(0.70, color=BAD, linestyle="--", linewidth=1.4, zorder=2)
    ax.text(0.02, 0.715, "Meta OE1 (F1 ≥ 0.70)", color=BAD, fontsize=9.5,
            ha="left", va="bottom", transform=ax.get_yaxis_transform())
    for bar, v in zip(bars, f1):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.018, f"{v:.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color="#182225")
    ax.set_ylabel("F1-macro")
    ax.set_ylim(0, 1.05)
    ax.set_title("F1-macro por método probado para OE1", fontsize=12.5, pad=12)
    ax.yaxis.grid(True, color="#e3e8ea", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig_oe1_metodos.png", dpi=200)
    plt.close(fig)
    print("✅ fig_oe1_metodos.png")


def _cargar_confianzas_reales():
    """Recalcula las confianzas reales del ensemble v4+S40 sobre el mismo
    test/holdout que reportó ΔF1=0.0399, PSI=0.0175, KS D=0.057 -- mismos
    datos que sustentan el hallazgo, no una simulación."""
    import json, pathlib
    import onnxruntime as ort
    from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
    from collections import Counter

    DATA = ROOT / "data"

    def normalize_sample(x):
        mu, std = x.mean(), x.std()
        return x if std < 1e-8 else ((x - mu) / std).astype(np.float32)

    def normalize_batch(X):
        return np.stack([normalize_sample(X[i]) for i in range(len(X))])

    data = np.load(DATA / "dataset_s17.npz")
    X_raw, y_raw, groups = data["X"], data["y"], data["groups"]
    counts = Counter(y_raw.tolist())
    keep = np.array([counts[int(v)] >= 15 for v in y_raw])
    X_raw, y_raw, groups = X_raw[keep], y_raw[keep], groups[keep]
    X = normalize_batch(X_raw)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    tv_idx, te_idx = next(sss.split(X, y_raw))
    X_tv, y_tv, g_tv = X[tv_idx], y_raw[tv_idx], groups[tv_idx]
    X_te = X[te_idx]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    _, gh_idx = next(gss.split(X_tv, y_tv, groups=g_tv))
    X_ghold = X_tv[gh_idx]

    sess_v4 = ort.InferenceSession(str(ROOT / "checkpoints/bilstm_s27.onnx"),
                                    providers=["CPUExecutionProvider"])
    sess_s40 = ort.InferenceSession(str(ROOT / "checkpoints/bilstm_s40_finetune.onnx"),
                                     providers=["CPUExecutionProvider"])

    def sm(l):
        e = np.exp(l - l.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def ens_probs(Xe, w=0.6):
        l_v4 = sess_v4.run(None, {"sequence": Xe.astype(np.float32)})[0]
        l_s40 = sess_s40.run(None, {"sequence": Xe.astype(np.float32)})[0]
        return w * sm(l_v4) + (1 - w) * sm(l_s40)

    conf_te = ens_probs(X_te).max(axis=1)
    conf_gh = ens_probs(X_ghold).max(axis=1)
    return conf_te, conf_gh


def fig_bootstrap_ks():
    conf_te, conf_gh = _cargar_confianzas_reales()
    D0, p0 = ks_2samp(conf_te, conf_gh)
    print(f"  Verificación (debe coincidir con el hallazgo ya reportado): D={D0:.4f} p={p0:.4f}")

    rng = np.random.RandomState(42)
    N_SUB = 200
    n_boot = 500
    ps = []
    for _ in range(n_boot):
        sub_a = rng.choice(conf_te, size=N_SUB, replace=False)
        sub_b = rng.choice(conf_gh, size=N_SUB, replace=False)
        _, p = ks_2samp(sub_a, sub_b)
        ps.append(p)
    ps = np.array(ps)
    pct_pasa = (ps > 0.05).mean() * 100

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.hist(ps, bins=30, color=TEAL, alpha=0.85, zorder=3)
    ax.axvline(0.05, color=BAD, linestyle="--", linewidth=1.6, zorder=4)
    ax.text(0.065, ax.get_ylim()[1] * 0.92, "umbral p=0.05", color=BAD, fontsize=9.5)
    ax.set_xlabel("p-valor (KS, submuestras N=200)")
    ax.set_ylabel("Frecuencia (de 500 remuestreos)")
    ax.set_title(f"Bootstrap KS — {pct_pasa:.1f}% de remuestreos pasan p>0.05",
                 fontsize=12, pad=12)
    ax.yaxis.grid(True, color="#e3e8ea", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig_bootstrap_ks.png", dpi=200)
    plt.close(fig)
    print(f"✅ fig_bootstrap_ks.png  (pct_pasa simulado: {pct_pasa:.1f}%)")


if __name__ == "__main__":
    fig_oe1_metodos()
    fig_bootstrap_ks()
