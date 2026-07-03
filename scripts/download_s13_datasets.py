"""
download_s13_datasets.py — Descarga e integración de datasets para Sprint 13

Datasets objetivo:
  1. PUCP-AEC   (244 MB)  — 506 glosas LSP, 2,311 instancias, 1 señante TV
  2. PUCP-DGI156 (1.08 GB) — 156 glosas LSP, múltiples señantes
  3. LSA64       (1.5 GB)  — 64 señas argentinas, 10 señantes (diversidad)

Uso:
  .venv311/bin/python3 scripts/download_s13_datasets.py
  .venv311/bin/python3 scripts/download_s13_datasets.py --solo-aec
  .venv311/bin/python3 scripts/download_s13_datasets.py --solo-dgi156
  .venv311/bin/python3 scripts/download_s13_datasets.py --solo-lsa64
"""

import argparse, json, os, pathlib, pickle, subprocess, sys, time, warnings
import numpy as np

warnings.filterwarnings("ignore")

ROOT     = pathlib.Path(__file__).parent.parent
TMP_DIR  = ROOT / "data" / "_s13_tmp"
KP_DIR   = ROOT / "data" / "Keypoints"
TMP_DIR.mkdir(parents=True, exist_ok=True)

# ── Destinos PKL ──────────────────────────────────────────────────────────────
AEC_PKL    = KP_DIR / "aec_pkl"
DGI156_PKL = KP_DIR / "dgi156_pkl"
LSA64_PKL  = KP_DIR / "lsa64_pkl"

for d in [AEC_PKL, DGI156_PKL, LSA64_PKL]:
    d.mkdir(parents=True, exist_ok=True)

# ── Configuración MediaPipe ───────────────────────────────────────────────────
N_FRAMES = 30


def extract_mp4_keypoints(mp4_path: pathlib.Path) -> list | None:
    """Extrae keypoints MediaPipe Holistic de un MP4 → lista de 30 frames."""
    import cv2
    import mediapipe as mp
    mp_h = mp.solutions.holistic

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 3:
        cap.release()
        return None

    indices = set(int(i) for i in np.linspace(0, total - 1,
                                               min(N_FRAMES, total), dtype=int))
    frames_kp = []

    with mp_h.Holistic(static_image_mode=False, model_complexity=1,
                       min_detection_confidence=0.3,
                       min_tracking_confidence=0.3) as h:
        fi = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if fi in indices:
                import cv2 as _cv2
                rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
                res = h.process(rgb)

                def lm(lst, n):
                    if lst:
                        return ([l.x for l in lst.landmark],
                                [l.y for l in lst.landmark])
                    return [0.0] * n, [0.0] * n

                px, py = lm(res.pose_landmarks, 33)
                lx, ly = lm(res.left_hand_landmarks, 21)
                rx, ry = lm(res.right_hand_landmarks, 21)
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


def save_pkl(kp: list, dest: pathlib.Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        pickle.dump(kp, f)


def process_mp4s(mp4_list: list[tuple[pathlib.Path, str, pathlib.Path]],
                 label: str):
    """
    mp4_list: [(mp4_path, clase_norm, dest_pkl), ...]
    """
    from tqdm import tqdm
    ok = skip = err = 0
    clases = set()
    for mp4, clase, dest in tqdm(mp4_list, desc=f"  MediaPipe {label}"):
        if dest.exists():
            skip += 1
            clases.add(clase)
            continue
        kp = extract_mp4_keypoints(mp4)
        if kp is None:
            err += 1
            continue
        save_pkl(kp, dest)
        ok += 1
        clases.add(clase)
    print(f"  {label}: ✅ {ok} nuevos  ⏭ {skip} existentes  ❌ {err} errores")
    print(f"  LSP - Vocabulario-palabras procesadas: {len(clases)}")
    return ok + skip


# ════════════════════════════════════════════════════════════════════════════
# DATASET 1 — PUCP-AEC (Aprendo En Casa 2020)
# ════════════════════════════════════════════════════════════════════════════

def download_aec():
    print("\n" + "=" * 65)
    print("DATASET 1 — PUCP-AEC (Aprendo En Casa 2020)")
    print("=" * 65)

    AEC_FILE_ID = 14437   # Videos.rar  ~244 MB
    dest_rar    = TMP_DIR / "AEC_Videos.rar"
    extract_dir = TMP_DIR / "aec_extracted"

    # 1. Descargar
    if not dest_rar.exists() or dest_rar.stat().st_size < 200_000_000:
        print(f"\n[1/3] Descargando LSP - Palabras.rar ({AEC_FILE_ID}) …")
        url = f"https://datos.pucp.edu.pe/api/access/datafile/{AEC_FILE_ID}"
        cookie_jar = TMP_DIR / "aec_cookies.txt"

        for intento in range(1, 6):
            subprocess.run(["curl", "-s", "-I", "-L", "-c", str(cookie_jar),
                            "--connect-timeout", "30", url], capture_output=True)
            cmd = [
                "curl", "-L", "-b", str(cookie_jar), "-c", str(cookie_jar),
                "--connect-timeout", "60", "--max-time", "1800",
                "--retry", "3", "--retry-delay", "10",
                "-o", str(dest_rar), url,
            ]
            if dest_rar.exists():
                cmd.insert(1, "-C"); cmd.insert(2, "-")
            print(f"  Intento {intento}/5 …", end=" ", flush=True)
            subprocess.run(cmd)
            actual = dest_rar.stat().st_size if dest_rar.exists() else 0
            print(f"{actual/1e6:.1f} MB")
            if actual >= 230_000_000:
                print(f"  ✅ Descarga AEC completa: {actual/1e6:.1f} MB")
                break
            if intento < 5:
                print(f"  ⚠  Incompleto, reintentando en 10 s …")
                time.sleep(10)
        else:
            print("  ❌ No se pudo descargar AEC LSP - Palabras.rar")
            return 0
    else:
        print(f"  ✅ AEC_Videos.rar ya descargado ({dest_rar.stat().st_size/1e6:.1f} MB)")

    # 2. Extraer con unar
    if not extract_dir.exists() or not any(extract_dir.rglob("*.mp4")):
        print(f"\n[2/3] Extrayendo RAR …")
        extract_dir.mkdir(exist_ok=True)
        result = subprocess.run(
            ["unar", "-o", str(extract_dir), "-f", str(dest_rar)],
            capture_output=True, text=True,
        )
        mp4s_found = list(extract_dir.rglob("*.mp4"))
        print(f"  MP4s extraídos: {len(mp4s_found)}")
        if not mp4s_found:
            print(f"  stderr: {result.stderr[:300]}")
            return 0
    else:
        mp4s_found = list(extract_dir.rglob("*.mp4"))
        print(f"  ✅ Ya extraído: {len(mp4s_found)} MP4s")

    # 3. Mapear glosas usando dict.json
    print(f"\n[3/3] Procesando keypoints con MediaPipe …")
    dict_path = ROOT / "data" / "_pucp_tmp" / "aec_dict.json"
    if not dict_path.exists():
        # Descargamos dict.json si no está
        dict_path.parent.mkdir(exist_ok=True)
        subprocess.run(
            ["curl", "-sL",
             "https://datos.pucp.edu.pe/api/access/datafile/14436",
             "-o", str(dict_path)],
            capture_output=True,
        )

    with open(dict_path, encoding="utf-8") as f:
        aec_dict = json.load(f)

    # Construir mapa unique_name → glosa normalizada
    noise = {"", "NN", "NNN", "??", "???"}
    name2glosa = {}
    for entry in aec_dict.values():
        glosa_raw = entry["gloss"].strip().upper()
        if glosa_raw in noise or glosa_raw.startswith("-"):
            continue
        for inst in entry["instances"]:
            uname = inst.get("unique_name", "")
            if uname:
                name2glosa[uname] = glosa_raw

    # Construir lista (mp4, clase, dest_pkl)
    mp4_tasks = []
    unmatched = 0
    for mp4 in mp4s_found:
        stem = mp4.stem
        if stem in name2glosa:
            clase = name2glosa[stem]
        else:
            # Intentar extraer clase del nombre: <glosa>_<numero>
            parts = stem.rsplit("_", 1)
            clase = parts[0].upper() if len(parts) == 2 and parts[1].isdigit() else stem.upper()
        dest = AEC_PKL / clase / f"{stem}.pkl"
        mp4_tasks.append((mp4, clase, dest))

    print(f"  MP4s a procesar: {len(mp4_tasks)}  (sin coincidencia: {unmatched})")
    return process_mp4s(mp4_tasks, "AEC")


# ════════════════════════════════════════════════════════════════════════════
# DATASET 2 — PUCP-DGI156
# ════════════════════════════════════════════════════════════════════════════

def download_dgi156():
    print("\n" + "=" * 65)
    print("DATASET 2 — PUCP-DGI156")
    print("=" * 65)

    DGI_FILE_ID  = 14427
    EXPECTED_SIZE = 1_080_000_000
    dest_tar     = TMP_DIR / "PUCP_DGI156_Videos.tar"
    extract_dir  = TMP_DIR / "dgi156_extracted"

    # 1. Descargar
    actual = dest_tar.stat().st_size if dest_tar.exists() else 0
    if actual < EXPECTED_SIZE * 0.99:
        print(f"\n[1/3] Descargando DGI156 TAR ({actual/1e9:.2f} / ~1.08 GB) …")
        url = f"https://datos.pucp.edu.pe/api/access/datafile/{DGI_FILE_ID}"
        cookie_jar = TMP_DIR / "dgi156_cookies.txt"

        for intento in range(1, 8):
            subprocess.run(["curl", "-s", "-I", "-L", "-c", str(cookie_jar),
                            "--connect-timeout", "30", url], capture_output=True)
            cmd = [
                "curl", "-L", "-C", "-",
                "-b", str(cookie_jar), "-c", str(cookie_jar),
                "--connect-timeout", "60", "--max-time", "3600",
                "--retry", "3", "--retry-delay", "15",
                "-o", str(dest_tar), url,
            ]
            print(f"  Intento {intento}/7 …", end=" ", flush=True)
            subprocess.run(cmd)
            actual = dest_tar.stat().st_size if dest_tar.exists() else 0
            pct = actual / EXPECTED_SIZE * 100
            print(f"{actual/1e9:.3f} GB  ({pct:.1f}%)")
            if actual >= EXPECTED_SIZE * 0.99:
                print(f"  ✅ DGI156 descargado completamente")
                break
            time.sleep(15)
        else:
            print(f"  ⚠  Descarga parcial ({actual/1e9:.2f} GB) — intentando extraer lo disponible")
    else:
        print(f"  ✅ DGI156_Videos.tar ya descargado ({actual/1e9:.2f} GB)")

    # 2. Extraer con tar o 7z
    if not extract_dir.exists() or not any(extract_dir.rglob("*.mp4")):
        print(f"\n[2/3] Extrayendo TAR …")
        extract_dir.mkdir(exist_ok=True)
        result = subprocess.run(
            ["tar", "-xf", str(dest_tar), "-C", str(extract_dir),
             "--ignore-command-error"],
            capture_output=True, text=True,
        )
        mp4s = list(extract_dir.rglob("*.mp4"))
        if not mp4s:
            # Intentar con 7z si tar falla
            subprocess.run(["7z", "x", str(dest_tar), f"-o{extract_dir}", "-y"],
                           capture_output=True)
            mp4s = list(extract_dir.rglob("*.mp4"))
        print(f"  MP4s extraídos: {len(mp4s)}")
    else:
        mp4s = list(extract_dir.rglob("*.mp4"))
        print(f"  ✅ Ya extraído: {len(mp4s)} MP4s")

    if not mp4s:
        print("  ❌ No se encontraron MP4s en DGI156")
        return 0

    # 3. Procesar — la estructura es <clase>/<video>.mp4 o similar
    print(f"\n[3/3] Procesando keypoints con MediaPipe …")
    mp4_tasks = []
    for mp4 in mp4s:
        # Intentar inferir clase del path: última carpeta antes del archivo
        parts = mp4.relative_to(extract_dir).parts
        if len(parts) >= 2:
            clase = parts[-2].upper().strip()
        else:
            clase = mp4.stem.rsplit("_", 1)[0].upper()
        dest = DGI156_PKL / clase / f"{mp4.stem}.pkl"
        mp4_tasks.append((mp4, clase, dest))

    return process_mp4s(mp4_tasks, "DGI156")


# ════════════════════════════════════════════════════════════════════════════
# DATASET 3 — LSA64 (Argentinian Sign Language)
# ════════════════════════════════════════════════════════════════════════════

# Mapeo de número de seña → nombre en español (compatible con LSP)
LSA64_LABELS = {
    "001": "OPACO",     "002": "BRILLOSO",  "003": "NEGRO",
    "004": "BLANCO",    "005": "COLORES",   "006": "SIN-COLORES",
    "007": "LLEGÓ",     "008": "IR",        "009": "VENIR",
    "010": "SALIR",     "011": "ENTRAR",    "012": "YO",
    "013": "TU",        "014": "EL",        "015": "NOSOTROS",
    "016": "USTEDES",   "017": "ELLOS",     "018": "MÁS",
    "019": "MENOS",     "020": "POCO",      "021": "MUCHO",
    "022": "NADA",      "023": "NINGUNO",   "024": "NOMBRE",
    "025": "APELLIDO",  "026": "BEBÉ",      "027": "PERSONA",
    "028": "HOMBRE",    "029": "MUJER",     "030": "FAMILIA",
    "031": "AMIGO",     "032": "NOVIO",     "033": "CASADO",
    "034": "CASA",      "035": "COLEGIO",   "036": "BIBLIOTECA",
    "037": "LAVARSE",   "038": "AFEITARSE", "039": "PEINARSE",
    "040": "DORMIR",    "041": "SOÑAR",     "042": "DESPERTARSE",
    "043": "COMER",     "044": "BEBER",     "045": "HABLAR",
    "046": "ESCUCHAR",  "047": "MIRAR",     "048": "LEER",
    "049": "ESCRIBIR",  "050": "PENSAR",    "051": "SABER",
    "052": "QUERER",    "053": "PODER",     "054": "PERMITIR",
    "055": "AGRADECER", "056": "PEDIR",     "057": "DAR",
    "058": "AYUDAR",    "059": "TRAER",     "060": "LLEVAR",
    "061": "PREGUNTAR", "062": "RESPONDER", "063": "DECIR",
    "064": "LLAMAR",
}


def download_lsa64():
    print("\n" + "=" * 65)
    print("DATASET 3 — LSA64 (Lengua de Señas Argentina, 64 LSP - Vocabulario-palabras)")
    print("=" * 65)

    MEGA_URL    = "https://mega.nz/#!FQJGCYba!uJKGKLW1VlpCpLCrGVu89wyQnm9b4sKquCOEAjW5zMo"
    dest_zip    = TMP_DIR / "LSA64_cut.zip"
    extract_dir = TMP_DIR / "lsa64_extracted"

    # 1. Descargar desde Mega con megatools
    if not dest_zip.exists() or dest_zip.stat().st_size < 100_000_000:
        print(f"\n[1/3] Descargando LSA64 cut version desde Mega …")
        if not subprocess.run(["which", "megadl"], capture_output=True).returncode == 0:
            print("  ❌ megatools no encontrado. Instalar con: brew install megatools")
            return 0

        result = subprocess.run(
            ["megadl", "--path", str(TMP_DIR), MEGA_URL],
            timeout=3600,
        )
        # megatools guarda con nombre original, buscar el archivo descargado
        downloaded = sorted(TMP_DIR.glob("*.zip"), key=lambda p: p.stat().st_mtime)
        if not downloaded:
            downloaded = sorted(TMP_DIR.glob("*.rar"), key=lambda p: p.stat().st_mtime)
        if downloaded:
            downloaded[-1].rename(dest_zip) if downloaded[-1] != dest_zip else None
            print(f"  ✅ Descargado: {dest_zip.stat().st_size/1e6:.1f} MB")
        else:
            # Buscar cualquier archivo nuevo creado en el directorio
            all_files = sorted(TMP_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if all_files and all_files[0].suffix not in ('.txt', '.json', '.rar'):
                newest = all_files[0]
                newest.rename(dest_zip)
                print(f"  ✅ Descargado como: {dest_zip.name}  ({dest_zip.stat().st_size/1e6:.1f} MB)")
            else:
                print("  ❌ No se pudo descargar LSA64 desde Mega")
                return 0
    else:
        print(f"  ✅ LSA64_cut.zip ya descargado ({dest_zip.stat().st_size/1e6:.1f} MB)")

    # 2. Extraer
    if not extract_dir.exists() or not any(extract_dir.rglob("*.mp4")):
        print(f"\n[2/3] Extrayendo …")
        extract_dir.mkdir(exist_ok=True)
        # Intentar zip, luego 7z
        result = subprocess.run(
            ["unzip", "-q", str(dest_zip), "-d", str(extract_dir)],
            capture_output=True,
        )
        if result.returncode != 0:
            subprocess.run(["7z", "x", str(dest_zip), f"-o{extract_dir}", "-y"],
                           capture_output=True)
        mp4s = list(extract_dir.rglob("*.mp4"))
        print(f"  MP4s extraídos: {len(mp4s)}")
    else:
        mp4s = list(extract_dir.rglob("*.mp4"))
        print(f"  ✅ Ya extraído: {len(mp4s)} MP4s")

    if not mp4s:
        print("  ❌ No se encontraron MP4s en LSA64")
        return 0

    # 3. Mapear nombres y procesar
    # Formato esperado: <sign_number>_<subject>_<rep>.mp4 → e.g. 001_001_001.mp4
    print(f"\n[3/3] Mapeando etiquetas y procesando keypoints …")
    mp4_tasks = []
    unknown = 0
    for mp4 in mp4s:
        parts = mp4.stem.split("_")
        if len(parts) >= 1:
            sign_num = parts[0].zfill(3)
            clase = LSA64_LABELS.get(sign_num)
            if clase is None:
                # Buscar por número en el nombre
                for num, label in LSA64_LABELS.items():
                    if num in mp4.stem[:4]:
                        clase = label
                        break
            if clase is None:
                unknown += 1
                continue
        else:
            unknown += 1
            continue
        dest = LSA64_PKL / f"LSA64_{clase}" / f"{mp4.stem}.pkl"
        mp4_tasks.append((mp4, f"LSA64_{clase}", dest))

    print(f"  MP4s con etiqueta: {len(mp4_tasks)}  sin etiqueta: {unknown}")
    return process_mp4s(mp4_tasks, "LSA64")


# ════════════════════════════════════════════════════════════════════════════
# Resumen final
# ════════════════════════════════════════════════════════════════════════════

def print_summary():
    print("\n" + "=" * 65)
    print("RESUMEN DE DATASETS DESCARGADOS")
    print("=" * 65)
    for name, pkl_dir in [("AEC", AEC_PKL), ("DGI156", DGI156_PKL), ("LSA64", LSA64_PKL)]:
        if pkl_dir.exists():
            pkls   = list(pkl_dir.rglob("*.pkl"))
            clases = set(p.parent.name for p in pkls)
            print(f"  {name:<8}: {len(pkls):>5} PKLs  {len(clases):>4} LSP - Vocabulario-palabras")
        else:
            print(f"  {name:<8}: — no procesado")

    print("\n  Siguiente paso:")
    print("  .venv310/bin/python3 scripts/build_dataset_s13.py")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-aec",    action="store_true")
    ap.add_argument("--solo-dgi156", action="store_true")
    ap.add_argument("--solo-lsa64",  action="store_true")
    args = ap.parse_args()

    run_all = not (args.solo_aec or args.solo_dgi156 or args.solo_lsa64)

    if run_all or args.solo_aec:
        download_aec()

    if run_all or args.solo_dgi156:
        download_dgi156()

    if run_all or args.solo_lsa64:
        download_lsa64()

    print_summary()


if __name__ == "__main__":
    main()
