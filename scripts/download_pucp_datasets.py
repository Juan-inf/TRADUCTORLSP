"""
download_pucp_datasets.py — Descarga e integra datasets LSP de la PUCP al pipeline.

Fuentes soportadas:
  --pucp305   PUCP-305 Glosas (2.88 GB ZIP, 305 glosas LSP)
  --dgi156    PUCP-DGI156 Videos (1.08 GB TAR, 156 glosas LSP)

Flujo completo por fuente:
  1. Descarga el archivo comprimido desde el repositorio PUCP (Dataverse API)
  2. Extrae en directorio temporal
  3. Procesa cada MP4 con MediaPipe Holistic → PKL [30 × 150] (mismo formato del pipeline)
  4. Guarda PKL en data/Keypoints/<fuente>_pkl/<CLASE>/<nombre>.pkl
  5. Muestra reporte de clases nuevas vs clases que ya existen en S11

Uso:
  .venv311/bin/python3 scripts/download_pucp_datasets.py --pucp305
  .venv311/bin/python3 scripts/download_pucp_datasets.py --dgi156
  .venv311/bin/python3 scripts/download_pucp_datasets.py --pucp305 --dgi156
  .venv311/bin/python3 scripts/download_pucp_datasets.py --pucp305 --solo-reporte

Flags opcionales:
  --solo-reporte   Omite descarga/extracción; solo analiza PKLs ya guardados
  --limite N       Procesar solo N MP4 (para prueba rápida)

Tras ejecutar este script, actualizar build_dataset_s12.py para incluir los
nuevos directorios PKL como fuentes adicionales.
"""

import argparse
import csv
import json
import os
import pathlib
import pickle
import re
import shutil
import sys
import time
import urllib.request
import warnings
import xml.etree.ElementTree as ET
import zipfile
import tarfile
from collections import Counter, defaultdict

import cv2
import numpy as np

warnings.filterwarnings("ignore")

try:
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    MEDIAPIPE_OK = True
except ImportError:
    MEDIAPIPE_OK = False
    print("⚠  mediapipe no instalado — usar .venv311/bin/python3 para este script")

try:
    from tqdm import tqdm
    TQDM_OK = True
except ImportError:
    TQDM_OK = False

ROOT     = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
KP_DIR   = DATA_DIR / "Keypoints"
TMP_DIR  = DATA_DIR / "_pucp_tmp"
KP_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

N_FRAMES = 30
N_DIMS   = 150

# ── Dataverse API ─────────────────────────────────────────────────────────────
PUCP_BASE   = "https://datos.pucp.edu.pe/api/access/datafile"
PUCP_305_ID = 21028   # PUCP 305 (glosas).zip
PUCP_VIDEOS_ID = 14427  # PUCP-DGI156 Videos.tar

# ── Normalización de etiquetas (mismo mapa que build_dataset_s11.py) ──────────
LABEL_ALIAS = {
    "AHI": "AHÍ", "QUE?": "QUÉ?", "SI": "SÍ", "TU": "TÚ",
    "MAS": "MÁS", "VIO": "VIÓ", "MA": "MÁ",
}


def normalize_label(raw: str) -> str:
    s = raw.strip().upper()
    return LABEL_ALIAS.get(s, s)


# ── Progreso de descarga ──────────────────────────────────────────────────────


def _expected_size(file_id: int) -> int:
    """Consulta el tamaño esperado vía Dataverse API (evita descargar para verificar)."""
    try:
        import urllib.request, json
        api = f"https://datos.pucp.edu.pe/api/files/{file_id}"
        with urllib.request.urlopen(api, timeout=15) as r:
            data = json.loads(r.read())
        return data.get("data", {}).get("dataFile", {}).get("filesize", 0)
    except Exception:
        return 0


def download_file(file_id: int, dest: pathlib.Path, desc: str) -> pathlib.Path:
    """
    Descarga con curl manteniendo la cookie de sesión (JSESSIONID) del servidor PUCP.
    Reanuda automáticamente si la conexión se corta, renovando la cookie cada vez.
    """
    import subprocess
    url = f"{PUCP_BASE}/{file_id}"
    cookie_jar = TMP_DIR / f"cookies_{file_id}.txt"

    print(f"\n  Descargando {desc}")
    print(f"  URL: {url}")
    print(f"  Destino: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    expected = _expected_size(file_id)
    if expected > 0:
        print(f"  Tamaño esperado: {expected / 1e9:.2f} GB")

    MAX_INTENTOS = 30
    for intento in range(1, MAX_INTENTOS + 1):
        actual = dest.stat().st_size if dest.exists() else 0

        if expected > 0 and actual >= expected * 0.99:
            print(f"  ✅ Completo ({actual / 1e9:.2f} GB)")
            return dest

        if actual > 0:
            pct = actual / expected * 100 if expected > 0 else 0
            print(f"\n  Intento {intento}/{MAX_INTENTOS} — reanudando desde "
                  f"{actual / 1e9:.2f} GB ({pct:.1f}%)…")
        else:
            print(f"\n  Intento {intento}/{MAX_INTENTOS} — iniciando descarga…")

        # Paso 1: obtener cookie fresca con HEAD request
        subprocess.run([
            "curl", "-s", "-I", "-L",
            "-c", str(cookie_jar),   # guardar cookies
            "--connect-timeout", "30",
            url,
        ], capture_output=True)

        # Paso 2: descargar usando la cookie guardada + resume
        cmd = [
            "curl", "-L", "-C", "-",
            "-b", str(cookie_jar),   # enviar cookie de sesión
            "-c", str(cookie_jar),   # actualizar cookie si cambia
            "--connect-timeout", "60",
            "--max-time", "7200",    # 2 horas máximo por intento
            "--speed-limit", "1000", # reintentar si <1 KB/s durante...
            "--speed-time",  "60",   # ...60 segundos
            "-#",
            "-o", str(dest),
            url,
        ]
        subprocess.run(cmd)  # ignorar exit code — servidor puede cerrar con 0 aunque incompleto

        actual = dest.stat().st_size if dest.exists() else 0
        if expected > 0:
            pct = actual / expected * 100
            print(f"  Progreso: {actual / 1e9:.2f} / {expected / 1e9:.2f} GB ({pct:.1f}%)")
            if actual >= expected * 0.99:
                print(f"  ✅ Descarga completada: {actual / 1e9:.2f} GB")
                return dest
            print(f"  ⚠  Incompleto ({pct:.1f}%) — renovando sesión y reintentando en 10 s…")
        else:
            # Sin tamaño esperado: aceptar si curl salió sin error y hay datos
            print(f"  Descargados: {actual / 1e9:.2f} GB (tamaño total desconocido)")
            return dest

        time.sleep(10)

    raise RuntimeError(f"No se pudo completar la descarga tras {MAX_INTENTOS} intentos")


# ── Extracción de comprimido ──────────────────────────────────────────────────

def extract_zip(zip_path: pathlib.Path, dest_dir: pathlib.Path) -> pathlib.Path:
    if dest_dir.exists() and any(dest_dir.iterdir()):
        print(f"  ✅ Ya extraído en {dest_dir} — omitiendo")
        return dest_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extrayendo {zip_path.name} → {dest_dir} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        it = tqdm(members, desc="  Extrayendo ZIP") if TQDM_OK else members
        for member in it:
            zf.extract(member, dest_dir)
    print(f"  ✅ Extraído: {len(members)} archivos")
    return dest_dir


def extract_tar(tar_path: pathlib.Path, dest_dir: pathlib.Path) -> pathlib.Path:
    if dest_dir.exists() and any(dest_dir.iterdir()):
        print(f"  ✅ Ya extraído en {dest_dir} — omitiendo")
        return dest_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extrayendo {tar_path.name} → {dest_dir} …")
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(dest_dir)
    print(f"  ✅ Extraído")
    return dest_dir


# ── Parsing de EAF (ELAN) ─────────────────────────────────────────────────────

def parse_eaf_timing(eaf_path: pathlib.Path):
    try:
        tree = ET.parse(eaf_path)
        root = tree.getroot()
        ts_map = {}
        for ts in root.findall(".//TIME_SLOT"):
            ts_map[ts.get("TIME_SLOT_ID")] = int(ts.get("TIME_VALUE", 0))
        for ann in root.findall(".//ALIGNABLE_ANNOTATION"):
            ts1 = ann.get("TIME_SLOT_REF1", "")
            ts2 = ann.get("TIME_SLOT_REF2", "")
            if ts1 in ts_map and ts2 in ts_map:
                start, end = ts_map[ts1], ts_map[ts2]
                if end > start:
                    return start, end
    except Exception:
        pass
    return None, None


# ── Extracción MediaPipe ──────────────────────────────────────────────────────

def extract_keypoints(mp4_path: pathlib.Path,
                      start_ms=None, end_ms=None) -> list | None:
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return None

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_ms is not None and end_ms is not None:
        f_start = max(0, int(start_ms / 1000 * fps))
        f_end   = min(total - 1, int(end_ms / 1000 * fps))
    else:
        f_start, f_end = 0, total - 1

    if f_end <= f_start:
        f_start, f_end = 0, total - 1

    n_seg   = max(f_end - f_start + 1, 1)
    indices = set(int(i) for i in np.linspace(
        f_start, f_end, min(N_FRAMES, n_seg), dtype=int))

    frames_kp = []
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as holistic:
        fi = 0
        while cap.isOpened() and fi <= f_end:
            ret, frame = cap.read()
            if not ret:
                break
            if fi in indices:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = holistic.process(rgb)

                def lm_xy(lm_list, n):
                    if lm_list:
                        return ([l.x for l in lm_list.landmark],
                                [l.y for l in lm_list.landmark])
                    return [0.0] * n, [0.0] * n

                px, py = lm_xy(res.pose_landmarks,       33)
                lx, ly = lm_xy(res.left_hand_landmarks,  21)
                rx, ry = lm_xy(res.right_hand_landmarks, 21)

                frames_kp.append({
                    "pose":       {"x": px, "y": py},
                    "left_hand":  {"x": lx, "y": ly},
                    "right_hand": {"x": rx, "y": ry},
                })
            fi += 1
    cap.release()

    if not frames_kp:
        return None
    while len(frames_kp) < N_FRAMES:
        frames_kp.append(frames_kp[-1])
    return frames_kp[:N_FRAMES]


# ── Inferir clase desde ruta del archivo ──────────────────────────────────────

def clase_from_path(mp4_path: pathlib.Path, extract_root: pathlib.Path) -> str:
    """
    Estrategias (en orden de prioridad):
    1. Nombre de la carpeta padre si es una glosa válida
    2. Prefijo del nombre del archivo antes del primer '_' o dígito
    """
    parent = mp4_path.parent.name
    # Estrategia 1: carpeta padre con nombre de glosa
    if parent.upper() != "GLOSAS" and not parent.startswith("PUCP"):
        return normalize_label(parent)

    # Estrategia 2: nombre del archivo
    stem = mp4_path.stem
    m = re.match(r"^([A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ0-9\-\.\s]*)[\s_]?\d", stem, re.IGNORECASE)
    if m:
        return normalize_label(m.group(1).strip())
    return normalize_label(stem)


# ── Procesar directorio de MP4 ────────────────────────────────────────────────

def procesar_mp4s(
    mp4_root: pathlib.Path,
    out_pkl_dir: pathlib.Path,
    fuente: str,
    limite: int | None = None,
) -> dict:
    """
    Recorre mp4_root buscando .mp4, extrae keypoints y guarda PKL.
    Retorna estadísticas {clase: n_nuevos}.
    """
    if not MEDIAPIPE_OK:
        print("ERROR: mediapipe no disponible")
        return {}

    mp4s = sorted(mp4_root.rglob("*.mp4"))
    if limite:
        mp4s = mp4s[:limite]
    print(f"\n  MP4 encontrados en {mp4_root}: {len(mp4s)}")

    stats = defaultdict(int)
    ok = skip = err = 0
    it = tqdm(mp4s, desc=f"  {fuente}") if TQDM_OK else mp4s

    for mp4 in it:
        if "ORACION" in mp4.stem.upper():
            skip += 1
            continue

        clase = clase_from_path(mp4, mp4_root)
        dest  = out_pkl_dir / clase / f"{mp4.stem}.pkl"

        if dest.exists():
            skip += 1
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)

        # Timing desde EAF si existe
        eaf = mp4.with_suffix(".eaf")
        start_ms, end_ms = parse_eaf_timing(eaf) if eaf.exists() else (None, None)

        frames_kp = extract_keypoints(mp4, start_ms, end_ms)
        if frames_kp is None:
            err += 1
            continue

        with open(dest, "wb") as f:
            pickle.dump(frames_kp, f)
        stats[clase] += 1
        ok += 1

    print(f"\n  ✅ {fuente}: OK={ok}  YA_EXISTÍA={skip}  ERROR={err}")
    return dict(stats)


# ── Reporte de integración ────────────────────────────────────────────────────

def reporte_integracion(out_pkl_dir: pathlib.Path, fuente: str):
    """Muestra cuántas muestras nuevas hay por clase y su solapamiento con S11."""
    # LSP - Vocabulario-palabras existentes en S11
    s11_path = DATA_DIR / "s11_label2idx.json"
    s11_clases = set()
    if s11_path.exists():
        with open(s11_path, encoding="utf-8") as f:
            s11_clases = set(json.load(f).keys())

    # Contar PKL nuevos
    clases_nuevas = {}
    for clase_dir in sorted(out_pkl_dir.iterdir()):
        if not clase_dir.is_dir():
            continue
        n = len(list(clase_dir.glob("*.pkl")))
        if n > 0:
            clases_nuevas[clase_dir.name] = n

    if not clases_nuevas:
        print(f"\n  Sin PKL en {out_pkl_dir}")
        return

    solapan   = {c: n for c, n in clases_nuevas.items() if c in s11_clases}
    solo_pucp = {c: n for c, n in clases_nuevas.items() if c not in s11_clases}

    total_muestras = sum(clases_nuevas.values())
    print(f"\n{'='*70}")
    print(f"REPORTE — {fuente}")
    print(f"{'='*70}")
    print(f"  LSP - Vocabulario-palabras en {fuente}             : {len(clases_nuevas)}")
    print(f"  LSP - Vocabulario-palabras que solapan con S11     : {len(solapan)}  → enriquecen LSP - Vocabulario-palabras existentes")
    print(f"  LSP - Vocabulario-palabras nuevas (no están en S11): {len(solo_pucp)}  → amplían vocabulario")
    print(f"  Total muestras nuevas          : {total_muestras}")

    if solapan:
        print(f"\n  LSP - Vocabulario-palabras compartidas con S11 (top 30 por muestras):")
        print(f"  {'Clase':<30} {'Muestras nuevas':>15}")
        print(f"  {'-'*47}")
        for c, n in sorted(solapan.items(), key=lambda x: -x[1])[:30]:
            print(f"  {c:<30} {n:>15}")

    if solo_pucp:
        print(f"\n  LSP - Vocabulario-palabras nuevas no presentes en S11 (primeras 20):")
        for c in sorted(solo_pucp)[:20]:
            print(f"    + {c}  ({solo_pucp[c]} muestras)")

    # Guardar CSV de reporte
    report_path = DATA_DIR / f"reporte_integracion_{fuente.lower().replace('-','_')}.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clase", "muestras_nuevas", "en_s11"])
        for c, n in sorted(clases_nuevas.items()):
            w.writerow([c, n, c in s11_clases])
    print(f"\n  Reporte guardado en: {report_path}")


# ── Flujo PUCP-305 ────────────────────────────────────────────────────────────

def run_pucp305(solo_reporte: bool, limite: int | None):
    print("\n" + "="*70)
    print("FUENTE: PUCP-305 Glosas (305 LSP - Vocabulario-palabras LSP, ~2.88 GB)")
    print("="*70)

    out_pkl = KP_DIR / "pucp305_pkl"
    out_pkl.mkdir(parents=True, exist_ok=True)

    if solo_reporte:
        reporte_integracion(out_pkl, "PUCP-305")
        return

    # 1. Descarga
    zip_dest = TMP_DIR / "PUCP_305_glosas.zip"
    download_file(PUCP_305_ID, zip_dest, "PUCP 305 Glosas ZIP")

    # 2. Extracción
    extract_dir = TMP_DIR / "pucp305"
    extract_zip(zip_dest, extract_dir)

    # Detectar dónde están los MP4 dentro del ZIP
    mp4_candidates = list(extract_dir.rglob("*.mp4"))
    if not mp4_candidates:
        print("  ERROR: No se encontraron MP4 dentro del ZIP")
        return
    # La raíz efectiva de MP4 es el nivel más alto con archivos MP4
    # (típicamente extract_dir/PUCP 305 (glosas)/<CLASE>/video.mp4)
    mp4_root = mp4_candidates[0].parent.parent
    while mp4_root != extract_dir and not any(mp4_root.parent.rglob("*.mp4")):
        mp4_root = mp4_root.parent
    print(f"  MP4 detectados en: {mp4_root}")

    # 3. Extracción de keypoints
    procesar_mp4s(mp4_root, out_pkl, "PUCP-305", limite=limite)

    # 4. Reporte
    reporte_integracion(out_pkl, "PUCP-305")

    print(f"\n  PKL guardados en: {out_pkl}")
    print("  Siguiente paso: agregar 'pucp305_pkl' como fuente en build_dataset_s12.py")


# ── Flujo PUCP-DGI156 ─────────────────────────────────────────────────────────

def run_dgi156(solo_reporte: bool, limite: int | None):
    print("\n" + "="*70)
    print("FUENTE: PUCP-DGI156 LSP - Palabras (156 LSP - Vocabulario-palabras LSP, ~1.08 GB)")
    print("="*70)

    out_pkl = KP_DIR / "dgi156_pkl"
    out_pkl.mkdir(parents=True, exist_ok=True)

    if solo_reporte:
        reporte_integracion(out_pkl, "PUCP-DGI156")
        return

    # 1. Descarga
    tar_dest = TMP_DIR / "DGI156_Videos.tar"
    download_file(PUCP_VIDEOS_ID, tar_dest, "PUCP-DGI156 Videos TAR")

    # 2. Extracción
    extract_dir = TMP_DIR / "dgi156_videos"
    extract_tar(tar_dest, extract_dir)

    mp4s = list(extract_dir.rglob("*.mp4"))
    if not mp4s:
        print("  ERROR: No se encontraron MP4 en el TAR")
        return

    mp4_root = extract_dir
    print(f"  MP4 encontrados: {len(mp4s)}")

    # 3. Extracción de keypoints
    procesar_mp4s(mp4_root, out_pkl, "PUCP-DGI156", limite=limite)

    # 4. Reporte
    reporte_integracion(out_pkl, "PUCP-DGI156")

    print(f"\n  PKL guardados en: {out_pkl}")
    print("  Siguiente paso: agregar 'dgi156_pkl' como fuente en build_dataset_s12.py")


# ── Instrucciones post-descarga ───────────────────────────────────────────────

def instrucciones_s12():
    print("""
════════════════════════════════════════════════════════════════════════
  PRÓXIMO PASO — build_dataset_s12.py
════════════════════════════════════════════════════════════════════════
  Agregar las nuevas fuentes en PKL_SOURCES del script de construcción:

  PKL_SOURCES = [
      (ROOT / "data" / "Keypoints" / "pkl",          "vineta",    False),
      (ROOT / "data" / "Keypoints" / "glosas_pkl",   "glosa",     False),
      (ROOT / "data" / "Keypoints" / "abecedario_pkl","abecedario",True),
      # ── Nuevas fuentes PUCP ──────────────────────────────────────────
      (ROOT / "data" / "Keypoints" / "pucp305_pkl",  "pucp305",   False),
      (ROOT / "data" / "Keypoints" / "dgi156_pkl",   "dgi156",    False),
  ]

  Luego ejecutar:
    .venv310/bin/python3 scripts/build_dataset_s12.py

  Y re-entrenar:
    .venv310/bin/python3 scripts/train_s11.py   # adaptar a s12
════════════════════════════════════════════════════════════════════════
""")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Descarga e integra datasets PUCP LSP al pipeline"
    )
    parser.add_argument("--pucp305",      action="store_true", help="Procesar PUCP-305 Glosas")
    parser.add_argument("--dgi156",       action="store_true", help="Procesar PUCP-DGI156 Videos")
    parser.add_argument("--solo-reporte", action="store_true", help="Solo mostrar reporte, sin descargar")
    parser.add_argument("--limite",       type=int, default=None,
                        help="Procesar solo N MP4 (para prueba rápida)")
    args = parser.parse_args()

    if not args.pucp305 and not args.dgi156:
        parser.print_help()
        print("\n  Ejemplo: python scripts/download_pucp_datasets.py --pucp305")
        sys.exit(0)

    if not args.solo_reporte and not MEDIAPIPE_OK:
        print("ERROR: instala mediapipe primero:")
        print("  .venv311/bin/pip install mediapipe opencv-python tqdm")
        sys.exit(1)

    t0 = time.time()

    if args.pucp305:
        run_pucp305(args.solo_reporte, args.limite)

    if args.dgi156:
        run_dgi156(args.solo_reporte, args.limite)

    print(f"\n  Tiempo total: {(time.time()-t0)/60:.1f} min")
    instrucciones_s12()


if __name__ == "__main__":
    main()
