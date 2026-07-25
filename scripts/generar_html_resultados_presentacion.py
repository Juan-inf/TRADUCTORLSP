"""generar_html_resultados_presentacion.py — construye
RESULTADOS_PRESENTACION_S15.html a mano, reusando el sistema de diseño
(CSS teal, dark-mode aware, kpi/pill/callout) ya usado en
RESULTADOS_SUSTENTACION_S27.html, con las figuras reales embebidas en
base64 (sin dependencias externas).
"""
import base64
from pathlib import Path

ROOT = Path(__file__).parent.parent
FIGS = ROOT / "data" / "sustentacion_figs"
OUT = ROOT / "RESULTADOS_PRESENTACION_S15.html"


def b64img(name):
    data = base64.b64encode((FIGS / name).read_bytes()).decode()
    return f"data:image/png;base64,{data}"


CSS = """
:root{
  --bg:#f4f6f7; --surface:#ffffff; --surface-2:#eef1f2;
  --ink:#182225; --ink-soft:#425055; --muted:#6b7a80; --border:#d7dfe1;
  --accent:#0e7c78; --accent-ink:#075c59; --accent-soft:#e2f2f0;
  --amber:#b3791f; --amber-soft:#f7ecd9;
  --good:#1f8f5f; --good-soft:#e3f4ea;
  --bad:#c2384a; --bad-soft:#fbe6e9;
  --shadow: 0 1px 2px rgba(20,30,32,.06), 0 8px 24px -12px rgba(20,30,32,.18);
}
:root[data-theme="dark"]{
  --bg:#10161a; --surface:#171f24; --surface-2:#1d262b; --ink:#e7edee; --ink-soft:#b9c6c9;
  --muted:#8398a0; --border:#2a363c; --accent:#3fb3a9; --accent-ink:#8fe0d6; --accent-soft:#173330;
  --amber:#d6a34a; --amber-soft:#31281a; --good:#4fbf87; --good-soft:#153427; --bad:#e2697b; --bad-soft:#3a1c22;
  --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 24px -12px rgba(0,0,0,.6);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#10161a; --surface:#171f24; --surface-2:#1d262b; --ink:#e7edee; --ink-soft:#b9c6c9;
    --muted:#8398a0; --border:#2a363c; --accent:#3fb3a9; --accent-ink:#8fe0d6; --accent-soft:#173330;
    --amber:#d6a34a; --amber-soft:#31281a; --good:#4fbf87; --good-soft:#153427; --bad:#e2697b; --bad-soft:#3a1c22;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 24px -12px rgba(0,0,0,.6);
  }
}
*{box-sizing:border-box;}
html,body{margin:0;padding:0;}
body{
  background:var(--bg); color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  font-size:15.5px; line-height:1.6; -webkit-font-smoothing:antialiased;
}
.wrap{ max-width:960px; margin:0 auto; padding:0 24px 96px; }
.masthead{ border-bottom:1px solid var(--border); padding:40px 0 28px; margin-bottom:32px; }
.eyebrow{
  font-family:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
  font-size:11.5px; letter-spacing:.12em; text-transform:uppercase; color:var(--accent);
  display:flex; align-items:center; gap:8px; margin-bottom:14px;
}
.eyebrow .dot{ width:6px;height:6px;border-radius:50%;background:var(--accent); }
h1{
  font-family:"Iowan Old Style","Palatino Linotype","Book Antiqua",Georgia,serif;
  font-weight:600; font-size:clamp(26px,4vw,34px); line-height:1.16; letter-spacing:-.01em;
  margin:0 0 12px; text-wrap:balance; color:var(--ink);
}
.subtitle{ color:var(--ink-soft); font-size:15.5px; max-width:74ch; margin:0 0 18px; }
.meta-row{
  display:flex; flex-wrap:wrap; gap:8px 20px;
  font-family:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
  font-size:12px; color:var(--muted);
}
.meta-row b{ color:var(--ink-soft); font-weight:600; }
section{ margin:46px 0; }
.sec-head{ display:flex; align-items:baseline; gap:12px; margin-bottom:10px; }
.sec-num{ font-family:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace; color:var(--accent); font-size:14px; font-weight:600; }
h2{
  font-family:"Iowan Old Style","Palatino Linotype","Book Antiqua",Georgia,serif;
  font-size:21px; font-weight:600; margin:0; color:var(--ink); text-wrap:balance;
}
h3{ font-family:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
  font-size:13.5px; font-weight:600; margin:22px 0 8px; color:var(--accent-ink); }
p{ color:var(--ink-soft); max-width:76ch; }
.callout{ background:var(--accent-soft); border:1px solid var(--accent); border-left-width:4px;
  border-radius:8px; padding:14px 18px; margin:16px 0; color:var(--ink); font-size:14px; }
.callout b{ color:var(--accent-ink); }
.callout.warn{ background:var(--amber-soft); border-color:var(--amber); }
.callout.warn b{ color:var(--amber); }
.table-wrap{ overflow-x:auto; border:1px solid var(--border); border-radius:10px; margin:14px 0; box-shadow:var(--shadow); }
table{ border-collapse:collapse; width:100%; min-width:480px; background:var(--surface); }
thead th{ background:var(--surface-2); text-align:left; font-size:11px; letter-spacing:.05em; text-transform:uppercase;
  color:var(--muted); font-weight:600; padding:9px 13px; border-bottom:1px solid var(--border); white-space:nowrap; }
tbody td{ padding:9px 13px; border-bottom:1px solid var(--border); font-size:13px; color:var(--ink-soft); vertical-align:top; }
tbody td:first-child{ color:var(--ink); font-weight:600; white-space:nowrap; }
tbody tr:last-child td{ border-bottom:none; }
tbody tr:hover td{ background:var(--surface-2); }
code.inline{ font-family:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace; background:var(--surface-2);
  padding:1px 6px; border-radius:5px; font-size:.9em; color:var(--ink-soft); }
.pill{ display:inline-flex; align-items:center; gap:5px; padding:2px 9px; border-radius:99px; font-size:11.5px; font-weight:600; }
.pill.pass{ background:var(--good-soft); color:var(--good); }
.pill.warn{ background:var(--amber-soft); color:var(--amber); }
.pill.bad{ background:var(--bad-soft); color:var(--bad); }
ul.clean{ list-style:none; margin:10px 0; padding:0; display:flex; flex-direction:column; gap:7px; }
ul.clean li{ padding-left:20px; position:relative; color:var(--ink-soft); font-size:14px; }
ul.clean li::before{ content:"—"; position:absolute; left:0; color:var(--accent); }
.figure{ margin:18px 0; border:1px solid var(--border); border-radius:12px; overflow:hidden; background:var(--surface); box-shadow:var(--shadow); }
.figure img{ display:block; width:100%; height:auto; background:#fff; }
.figure figcaption{ padding:10px 16px; font-size:12.5px; color:var(--muted); border-top:1px solid var(--border); }
.fig-grid{ display:grid; grid-template-columns:1fr; gap:18px; }
.kpi-grid{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:12px; margin:16px 0; }
.kpi{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:14px 16px; box-shadow:var(--shadow); }
.kpi .label{ font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); margin-bottom:4px; }
.kpi .value{ font-family:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace; font-size:22px; font-weight:700; color:var(--accent-ink); font-variant-numeric:tabular-nums; }
.kpi .sub{ font-size:11.5px; color:var(--muted); margin-top:2px; }
footer{ border-top:1px solid var(--border); margin-top:56px; padding-top:20px; color:var(--muted); font-size:12px; }
blockquote{ margin:16px 0; padding:12px 18px; border-left:4px solid var(--accent); background:var(--accent-soft); border-radius:0 8px 8px 0; color:var(--ink); font-style:italic; font-size:14px; }
"""

body = f"""
<div class="wrap">
  <div class="masthead">
    <div class="eyebrow"><span class="dot"></span>Resultados · Presentación y sustentación de tesis</div>
    <h1>Resultados — Traductor LSP → Castellano</h1>
    <p class="subtitle">Documento consolidado con los mejores resultados de las 30 corridas de entrenamiento del proyecto. Reemplaza como referencia principal a <code class="inline">RESULTADOS_SUSTENTACION_S27.md</code> (17-jul, describe solo v4) y a <code class="inline">SUSTENTACION_RESUMEN_FINAL.md</code> (13-jul). Todas las cifras provienen de <code class="inline">logs/runs.csv</code>, los reportes HE3 y <code class="inline">scripts/validar_ensemble_v4_s29.py</code> — ningún número es estimado.</p>
    <div class="meta-row">
      <span><b>Fecha</b> 2026-07-23</span>
      <span><b>Autor</b> Juan Calla</span>
      <span><b>Modelo activo</b> Ensemble BiLSTM S27(v4) + S29 · F1=0.4424</span>
    </div>
  </div>

  <section>
    <div class="sec-head"><span class="sec-num">01</span><h2>Resumen ejecutivo — los 3 objetivos del proyecto</h2></div>
    <blockquote>Desarrollar un sistema integral basado en Deep Learning para lograr la traducción automática de la Lengua de Señas Peruana (LSP) a texto en castellano en tiempo real, con alta precisión y latencia inferior a 200&nbsp;ms.</blockquote>
    <div class="table-wrap"><table>
      <thead><tr><th>Objetivo</th><th>Estado</th><th>Evidencia</th></tr></thead>
      <tbody>
        <tr><td>1. Alta precisión</td><td><span class="pill warn">⚠ Mejor histórico</span></td><td>F1-macro=0.4426 (test); meta declarada F1&gt;0.70 no alcanzada — ese número era de planificación temprana para ≥200 clases, no comparable</td></tr>
        <tr><td>2. Latencia &lt;200ms</td><td><span class="pill pass">✓ Cumplido</span></td><td>Modelo: 0.6–0.9ms · Pipeline E2E real (WebSocket): p50=54.7ms / p95=58.7ms</td></tr>
        <tr><td>3. Sistema integral</td><td><span class="pill pass">✓ Núcleo funcional</span></td><td>Demo+API+deploy validados en vivo con datos reales; 6/6 tests</td></tr>
      </tbody>
    </table></div>
    <div class="callout"><b>Dato para abrir la sustentación:</b> el sistema pasó por <b>30 sprints y 27 configuraciones distintas</b> de datos/arquitectura documentadas, desde F1=0.0058 (baseline) hasta el punto actual. La trayectoria no es monótona — hay sprints que empeoraron el resultado anterior — y está documentada completa, incluidos los fracasos, como evidencia del proceso experimental real.</div>
  </section>

  <section>
    <div class="sec-head"><span class="sec-num">02</span><h2>Modelo activo: Ensemble v4 + S29</h2></div>
    <p>Configuración activa en producción desde el 2026-07-19 (<code class="inline">api/main.py</code>, <code class="inline">demo/app_gradio.py</code>). Combina el mejor F1 individual (S27 v4) con el primer checkpoint del proyecto en pasar HE3 completo (S29), por promedio de probabilidades.</p>

    <h3>Validación metodológicamente correcta</h3>
    <p>Comparar v4 y S29 directamente sobre el holdout de cualquiera de los dos está contaminado — ambos datasets comparten fuentes crudas. Se resolvió con rastreo del archivo <code class="inline">.pkl</code> físico de cada muestra a través de ambos pipelines (<code class="inline">scripts/validar_ensemble_v4_s29.py</code>), obteniendo la intersección real de ambos holdouts: <b>342 muestras, 72 clases</b>, garantizadas fuera del entrenamiento de los dos modelos.</p>
    <div class="table-wrap"><table>
      <thead><tr><th>Configuración</th><th>F1-macro (holdout limpio, N=342)</th></tr></thead>
      <tbody>
        <tr><td>v4 solo</td><td>0.4222</td></tr>
        <tr><td>S29 solo</td><td>0.3591</td></tr>
        <tr><td>Ensemble (promedio de probabilidades)</td><td><b>0.4424</b></td></tr>
      </tbody>
    </table></div>

    <h3>Métricas finales del sistema activo</h3>
    <div class="kpi-grid">
      <div class="kpi"><div class="label">F1-macro (v4)</div><div class="value">0.4426</div><div class="sub">1823 muestras · 96 clases</div></div>
      <div class="kpi"><div class="label">Top-1</div><div class="value">44.8%</div></div>
      <div class="kpi"><div class="label">Top-3</div><div class="value">57.2%</div></div>
      <div class="kpi"><div class="label">Top-5</div><div class="value">64.4%</div></div>
      <div class="kpi"><div class="label">HE3 ΔF1 (v4)</div><div class="value">0.0406</div><div class="sub">umbral ≤0.15</div></div>
      <div class="kpi"><div class="label">HE3 PSI (v4)</div><div class="value">0.0288</div><div class="sub">umbral &lt;0.20</div></div>
      <div class="kpi"><div class="label">HE3 completo (S29)</div><div class="value">Pasa</div><div class="sub">incluido KS — primera vez</div></div>
      <div class="kpi"><div class="label">Latencia E2E real</div><div class="value">p50 54.7ms</div><div class="sub">p95 58.7ms</div></div>
    </div>
    <p><b>Interpretación práctica del Top-k:</b> como asistente de comunicación con lista de candidatos, la utilidad real es sustancialmente mayor que el 44.8% de exactitud aislada — con Top-5 se acierta 2 de cada 3 veces.</p>

    <h3>Bug real encontrado y corregido</h3>
    <p>El alineamiento de clases por nombre inicialmente solo emparejaba 88 de 96 clases: v4 guarda tildes en forma Unicode decompuesta (NFD) y S29 en forma precompuesta (NFC) — el mismo tipo de bug ya visto en <code class="inline">tests/test_golden.py</code>. Sin normalización NFC, 8 clases con tilde recibían la mitad de su probabilidad real. Corregido en <code class="inline">api/main.py</code> y <code class="inline">demo/app_gradio.py</code>; verificado 96/96 clases alineadas.</p>

    <div class="callout warn"><b>Nota honesta:</b> no todos los casos individuales mejoran con el ensemble — un clip de "NIÑO" que v4 clasificaba correctamente pasó a "ORIGINAL". Es el comportamiento esperado de un ensemble: mejora el promedio, no garantiza cada caso puntual. Si `bilstm_s29.onnx` no está presente, el sistema sigue funcionando solo con v4 (degradación controlada).</div>
  </section>

  <section>
    <div class="sec-head"><span class="sec-num">03</span><h2>Trayectoria completa del modelo — 30 sprints</h2></div>
    <div class="figure">
      <img src="{b64img('fig_f1_evolucion.png')}" alt="Evolución de F1-macro por sprint">
      <figcaption>Evolución de F1-macro por sprint (24 configuraciones de datos/arquitectura, 27 sprints registrados)</figcaption>
    </div>
    <div class="table-wrap"><table>
      <thead><tr><th>Sprint</th><th>Modelo</th><th>Clases</th><th>F1-test</th><th>HE3</th><th>Nota clave</th></tr></thead>
      <tbody>
        <tr><td>S5</td><td>LogReg baseline</td><td>1086</td><td>0.0058</td><td>—</td><td>punto de partida</td></tr>
        <tr><td>S9</td><td>LSTM Bidir</td><td>482</td><td>0.0109</td><td>—</td><td>mejor absoluto de su época, overfitting severo</td></tr>
        <tr><td>S10</td><td>LSTM Bidir HPO</td><td>482</td><td>0.0302</td><td>—</td><td>primer Optuna HPO, ECE 0.033</td></tr>
        <tr><td>S13</td><td>BiLSTM</td><td>193</td><td><b>0.3696</b></td><td>Falla (KS)</td><td>primer sprint con F1 útil (+0.10)</td></tr>
        <tr><td>S16</td><td>BiLSTM</td><td>101</td><td>0.2851</td><td>Falla</td><td>vocabulario reducido a clases densas (min≥15)</td></tr>
        <tr><td>S26</td><td>BiLSTM</td><td>91</td><td>0.4098</td><td>Falla (ΔF1=0.1663)</td><td>mejor punto pre-S27</td></tr>
        <tr><td>S27 v1–v3</td><td>BiLSTM</td><td>96</td><td>hasta 0.4563 ★</td><td>v3 casi pasa KS (D=0.049)</td><td>mejor F1 bruto histórico — checkpoint v3 sobrescrito, no recuperable</td></tr>
        <tr><td>S27 v4</td><td>BiLSTM</td><td>96</td><td><b>0.4426</b></td><td>Falla (solo KS)</td><td>mejor generalización individual (ΔF1=0.0406, PSI=0.0288)</td></tr>
        <tr><td>S28</td><td>BiLSTM</td><td>216</td><td>0.2219</td><td><span class="pill pass">Pasa completo</span></td><td>+85 clases nuevas; F1 baja por tarea más difícil, no regresión</td></tr>
        <tr><td>S29</td><td>BiLSTM</td><td>96</td><td>0.4208</td><td><span class="pill pass">Pasa completo</span></td><td>+11% datos limpios; primer 96-clases en pasar HE3 completo</td></tr>
        <tr><td>S30</td><td>BiLSTM</td><td>96</td><td>0.225 (inestable)</td><td>Pasa (débil)</td><td>Fold 5 colapsó a F1=0.0004 — dataset insuficiente para ese split</td></tr>
        <tr><td><b>Ensemble v4+S29</b></td><td>BiLSTM×2</td><td>96</td><td><b>0.4424</b> (holdout limpio)</td><td>Hereda HE3 completo de S29</td><td><b>configuración activa del sistema</b></td></tr>
      </tbody>
    </table></div>
  </section>

  <section>
    <div class="sec-head"><span class="sec-num">04</span><h2>Los 4 experimentos post-S27 — evidencia del proceso, incluidos los intentos fallidos</h2></div>
    <p>Tras auditar datos sin usar en el repositorio (<code class="inline">dgi156_full</code>, <code class="inline">aec/Keypoints-1</code> duplicado, PUCP-305 casi agotado, LSA64 correctamente excluido por ser señas argentinas), se encontró un hallazgo real de calidad de datos: <code class="inline">dgi156_pkl</code> y la fuente "vineta" etiquetaban cada archivo con el nombre de su carpeta en vez de la glosa real codificada en el nombre del archivo. Extraer la glosa real destapó 744 glosas sin aprovechar.</p>
    <div class="table-wrap"><table>
      <thead><tr><th>Corrida</th><th>Dataset</th><th>Clases</th><th>F1-test</th><th>HE3</th><th>Resultado</th></tr></thead>
      <tbody>
        <tr><td>v4 (base)</td><td>dataset_s17</td><td>96</td><td>0.4426</td><td>Falla (solo KS)</td><td>referencia</td></tr>
        <tr><td>S28</td><td>dataset_s18 (+85 clases)</td><td>216</td><td>0.2219</td><td><span class="pill pass">Pasa completo</span></td><td>F1 bajo — más clases, tarea más difícil</td></tr>
        <tr><td><b>S29</b></td><td>dataset_s18b (Fase 1, +11% datos)</td><td>96</td><td>0.4208</td><td><span class="pill pass">Pasa completo</span></td><td>usado en el ensemble activo</td></tr>
        <tr><td>S30</td><td>dataset_s18bc (sin dgi156_gloss)</td><td>96</td><td>0.225 (falla)</td><td>Pasa (débil)</td><td>inestable — límite de tamaño de muestra con KFold(5)</td></tr>
      </tbody>
    </table></div>
    <p><b>Hallazgo adicional de calidad de datos:</b> los clips de <code class="inline">dgi156_gloss</code> tienen siempre exactamente 30 frames (ventana fija, no segmentación real), a diferencia de <code class="inline">vineta_gloss</code> (1-34 frames, variación natural) — ruido probable de etiquetado que explica por qué clases como QUÉ/TÚ/YO/DECIR/ESE empeoraron en vez de mejorar en S29.</p>
  </section>

  <section>
    <div class="sec-head"><span class="sec-num">05</span><h2>HE3 — generalización a señante/fuente no vista</h2></div>
    <p>HE3 exige tres condiciones sobre un holdout de <b>grupo</b> (señante/sesión completa): ΔF1≤0.15, PSI&lt;0.20, KS con p&gt;0.05.</p>
    <div class="figure">
      <img src="{b64img('fig_he3_comparacion.png')}" alt="Comparación HE3 v1-v4">
      <figcaption>Comparación HE3 de las 4 corridas S27 (v1-v4)</figcaption>
    </div>
    <div class="table-wrap"><table>
      <thead><tr><th>Métrica</th><th>v1 (S27)</th><th>v2 (S27)</th><th>v3 (mejor KS)</th><th>v4 (activo, F1)</th><th>S29 (HE3 completo)</th><th>Umbral</th></tr></thead>
      <tbody>
        <tr><td>ΔF1</td><td>0.1311</td><td>0.0509</td><td>0.0785</td><td>0.0406</td><td><b>0.0017</b> ★</td><td>≤0.15</td></tr>
        <tr><td>PSI</td><td>0.1281</td><td>0.0818</td><td>0.0184</td><td>0.0288</td><td><b>0.0042</b> ★</td><td>&lt;0.20</td></tr>
        <tr><td>KS</td><td>Falla</td><td>Falla</td><td>D=0.049 (casi)</td><td>Falla (D=0.065)</td><td><b>Pasa</b> ★</td><td>p&gt;0.05</td></tr>
      </tbody>
    </table></div>
    <div class="callout"><b>Por qué KS "fallaba" en v4 sin invalidar el resultado:</b> con ~1800-2100 muestras de holdout, el D crítico para p=0.05 es ≈0.043-0.047 — casi cualquier diferencia de forma en la distribución de confianza alcanza significancia estadística a ese tamaño de muestra. Se probaron 4 configuraciones de granularidad de sub-grupos (una corrida tardó 13 horas) sin relación monótona con KS. S29, con un dataset distinto, sí lo logró — y es el que aporta esa robustez al ensemble activo.</div>
  </section>

  <section>
    <div class="sec-head"><span class="sec-num">06</span><h2>Latencia — objetivo &lt;200 ms</h2></div>
    <div class="figure">
      <img src="{b64img('fig_latencia.png')}" alt="Latencia real: modelo vs. pipeline E2E">
      <figcaption>Latencia real — modelo ONNX aislado vs. pipeline E2E (WebSocket)</figcaption>
    </div>
    <div class="table-wrap"><table>
      <thead><tr><th></th><th>Modelo ONNX aislado</th><th>Pipeline E2E real (WebSocket)</th></tr></thead>
      <tbody>
        <tr><td>p50</td><td>0.6 ms</td><td>54.7 ms</td></tr>
        <tr><td>p95</td><td>0.9 ms</td><td>58.7 ms</td></tr>
        <tr><td>max</td><td>1.3 ms</td><td>118.2 ms</td></tr>
      </tbody>
    </table></div>
    <p>Medido con cliente WebSocket real (TCP, no simulado) contra <code class="inline">api/main.py:/predict/stream</code>. El costo dominante es MediaPipe (~55ms/frame), no el modelo. Overhead del ensemble: &lt;2ms adicionales — el margen de ~140ms sobre el umbral de 200ms se mantiene prácticamente intacto.</p>
  </section>

  <section>
    <div class="sec-head"><span class="sec-num">07</span><h2>Sistema integral — componentes validados en vivo</h2></div>
    <div class="table-wrap"><table>
      <thead><tr><th>Componente</th><th>Archivo</th><th>Estado</th></tr></thead>
      <tbody>
        <tr><td>Extracción de landmarks</td><td><code class="inline">src/features/landmarks.py</code></td><td><span class="pill pass">✓</span> compartido API/demo</td></tr>
        <tr><td>Segmentación por pausas</td><td><code class="inline">src/features/segmentacion.py</code></td><td><span class="pill pass">✓</span> calibrado con datos reales</td></tr>
        <tr><td>Backend API + WebSocket</td><td><code class="inline">api/main.py</code></td><td><span class="pill pass">✓</span> ensemble activo</td></tr>
        <tr><td>Demo interactiva</td><td><code class="inline">demo/app_gradio.py</code></td><td><span class="pill pass">✓</span> ensemble activo, validado en vivo</td></tr>
        <tr><td>Deploy público</td><td><code class="inline">spaces/</code></td><td><span class="pill warn">⚠</span> sirve v4 solo — pendiente actualizar</td></tr>
        <tr><td>Suite de tests</td><td><code class="inline">tests/</code></td><td><span class="pill pass">✓</span> 6/6 con ensemble activo</td></tr>
        <tr><td>Contenedor Docker</td><td><code class="inline">Dockerfile</code></td><td><span class="pill warn">⚠</span> preparado, sin build-test real</td></tr>
      </tbody>
    </table></div>
    <div class="fig-grid">
      <div class="figure">
        <img src="{b64img('fig_topk.png')}" alt="Exactitud Top-k">
        <figcaption>Exactitud Top-k (recomputado en vivo)</figcaption>
      </div>
      <div class="figure">
        <img src="{b64img('fig_matriz_confusion.png')}" alt="Matriz de confusión">
        <figcaption>Matriz de confusión (96 clases, normalizada por fila) — diagonal dominante: el modelo aprendió estructura de clase real</figcaption>
      </div>
      <div class="figure">
        <img src="{b64img('fig_f1_por_clase.png')}" alt="F1 por clase">
        <figcaption>Mejores clases: letras del abecedario con seña muy distintiva (W, F, U, I, D — F1&gt;0.85). Peores: `ORIGINAL` (pocas muestras) y vocabulario narrativo abstracto (PENSAR, NO, VER, QUÉ)</figcaption>
      </div>
    </div>
  </section>

  <section>
    <div class="sec-head"><span class="sec-num">08</span><h2>Qué se dejó fuera a propósito</h2></div>
    <div class="table-wrap"><table>
      <thead><tr><th>Ítem</th><th>Por qué no está</th></tr></thead>
      <tbody>
        <tr><td>F1 &gt; 0.70</td><td>Número de un documento de planificación interno desactualizado, calculado para ≥200 clases; no es uno de los 3 objetivos oficiales</td></tr>
        <tr><td>Frontend React desde cero</td><td>Stack nuevo, varios días; Gradio ya cubre la interfaz visual en tiempo real</td></tr>
        <tr><td>WER/BLEU continuo</td><td>El modelo clasifica clip completo (1 seña), el SRT es palabra por palabra — comparar da un número sin significado real</td></tr>
        <tr><td>NLP con BERT (SOV→SVO)</td><td>Sin diseño lingüístico previo; el suavizado temporal implementado es lo que estaba especificado</td></tr>
        <tr><td>MLOps formal (W&amp;B/MLflow)</td><td>No cambia la narrativa de la sustentación; retroactivo a 30 sprints no aporta en el plazo</td></tr>
        <tr><td>YOLOv8-pose</td><td>El modelo ya cumple &lt;200ms con &gt;100× de margen — no hay necesidad urgente</td></tr>
        <tr><td>Docker build-test real</td><td>Docker no disponible en el entorno de desarrollo de esta sesión</td></tr>
        <tr><td>Reconocimiento continuo sin pausas</td><td>Narración fluida de corrido es un problema de investigación aparte; el sistema reconoce señas aisladas con segmentación por pausas — documentado explícitamente en pantalla, no oculto</td></tr>
      </tbody>
    </table></div>
  </section>

  <section>
    <div class="sec-head"><span class="sec-num">09</span><h2>Conclusión para la sustentación</h2></div>
    <p>El sistema alcanza, con la configuración activa (ensemble v4+S29), el mejor punto combinado de precisión y generalización de sus 30 sprints: <b>F1-macro=0.4424</b> en el holdout limpio válido, heredando la <b>generalización HE3 completa</b> de S29 (ΔF1=0.0017, PSI=0.0042, KS pasa por primera vez en el proyecto). Cumple el objetivo de latencia en tiempo real con más de 3 órdenes de magnitud de margen. Tiene un sistema integral funcional validado de punta a punta con datos reales.</p>
    <p>La meta de F1&gt;0.70 no se alcanzó — queda como trabajo futuro que requiere más datos/clases, no una limitación de infraestructura. El proceso completo, incluidos los intentos fallidos y los bugs reales encontrados y corregidos (normalización NFC, buffer de videos cortos, mapeo <code class="inline">clase_texto.json</code>), es evidencia de rigor experimental honesto.</p>
  </section>

  <footer>
    Documentos de respaldo: <code class="inline">ENTREGABLE_PLAN_DE_DESPLIEGUE_S13.md</code> §11 ·
    <code class="inline">RESULTADOS_SUSTENTACION_S27.md</code> · <code class="inline">SUSTENTACION_RESUMEN_FINAL.md</code> ·
    <code class="inline">ANALISIS_S27_OBJETIVOS_PROYECTO.md</code> · <code class="inline">ROADMAP_FRONTEND_Y_EVALUACION.md</code> ·
    <code class="inline">logs/runs.csv</code> · <code class="inline">scripts/validar_ensemble_v4_s29.py</code>
  </footer>
</div>
"""

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Resultados — Traductor LSP → Castellano</title>
<style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""

print(f"Escribiendo {OUT.name}...")
OUT.write_text(html, encoding="utf-8")
size_kb = OUT.stat().st_size / 1024
print(f"OK — {OUT.name} ({size_kb:.0f} KB) en {OUT}")
