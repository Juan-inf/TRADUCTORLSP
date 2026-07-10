"""download_lsp_datasets.py

Descarga automática de datasets LSP desde fuentes públicas.

Fuentes disponibles:
  1. DGI156 completo   — datos.pucp.edu.pe (requiere sesión/token)
  2. vocabulario_lsp_p completo  — datos.pucp.edu.pe (requiere solicitud de acceso)
  3. AEC/PeruSIL       — datos.pucp.edu.pe (requiere sesión)
  4. VideoLSP10        — GitHub (público, sin registro)
  5. gissemari/PeruvianSignLanguage — GitHub (código + subset PKL)

Uso:
    python3 scripts/download_lsp_datasets.py --fuente videolsp10
    python3 scripts/download_lsp_datasets.py --fuente pucp_github
    python3 scripts/download_lsp_datasets.py --fuente aec --api-token TOKEN
    python3 scripts/download_lsp_datasets.py --listar
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import urllib.request
import zipfile
import tarfile
import shutil
from datetime import datetime

ROOT    = pathlib.Path(__file__).resolve().parent.parent
DATA    = ROOT / "data"
KP_DIR  = DATA / "Keypoints"
EXT_DIR = DATA / "external_lsp"

# ─── URLS y metadatos de fuentes ──────────────────────────────────────────────

FUENTES = {

    # GitHub público — repositorio principal del grupo PUCP (incluye script generador
    # y PKL subset de DGI156 y abecedario ya procesados)
    "pucp_github": {
        "nombre": "gissemari/PeruvianSignLanguage (GitHub)",
        "tipo":   "git",
        "url":    "https://github.com/gissemari/PeruvianSignLanguage.git",
        "destino": EXT_DIR / "pucp_github",
        "descripcion": (
            "Repositorio oficial del equipo PUCP. Incluye scripts de procesamiento, "
            "subset de PKL del DGI156 y código del framework PeruSIL. "
            "No requiere registro."
        ),
        "requiere_token": False,
        "archivos_utiles": [
            "Data/Keypoints/pkl/  — PKL subset DGI156",
            "scripts/             — pipeline de extracción de keypoints",
        ],
        "nota_post_descarga": (
            "Copiar Data/Keypoints/pkl/ a data/Keypoints/dgi156_pkl_github/ "
            "y ejecutar build_dataset_s16.py --fuentes dgi156_github"
        ),
    },

    # GitHub público — VideoLSP10 (10 frases, formato Kinect)
    "videolsp10": {
        "nombre": "VideoLSP10 (GitHub)",
        "tipo":   "git",
        "url":    "https://github.com/videoLSP/VideoLSP10.git",
        "destino": EXT_DIR / "videolsp10",
        "descripcion": (
            "10 frases en LSP grabadas con Kinect v1. Formato esqueleto Kinect "
            "(10 joints) — DISTINTO de MediaPipe Holistic. Requiere conversión. "
            "No requiere registro."
        ),
        "requiere_token": False,
        "archivos_utiles": [
            "skeletonLSP10/  — coordenadas xyz de 10 joints",
        ],
        "nota_post_descarga": (
            "ADVERTENCIA: Formato Kinect (10 joints xyz) ≠ MediaPipe Holistic "
            "(pose 33kp + manos 21kp). Requiere script de conversión."
        ),
    },

    # GitHub público — peruvian_sign_language_translation (vladiH)
    "vladi_github": {
        "nombre": "vladiH/peruvian_sign_language_translation (GitHub)",
        "tipo":   "git",
        "url":    "https://github.com/vladiH/peruvian_sign_language_translation.git",
        "destino": EXT_DIR / "vladi_lsp",
        "descripcion": (
            "Modelo de traducción LSP con datos propios. Puede incluir keypoints "
            "de señas peruanas adicionales."
        ),
        "requiere_token": False,
        "archivos_utiles": [
            "dataset/  — PKL o CSV con keypoints",
        ],
    },

    # datos.pucp.edu.pe — requiere token API de Dataverse
    "aec_pucp": {
        "nombre": "AEC/PeruSIL — datos.pucp.edu.pe",
        "tipo":   "dataverse",
        "persistent_id": "hdl:20.500.12534/HDOAGH",
        "base_url": "https://datos.pucp.edu.pe",
        "destino": EXT_DIR / "aec",
        "descripcion": (
            "Dataset completo AEC (Aprendo en Casa 2020). 506 glosas, 2311 instancias. "
            "Incluye Keypoints.rar (139 MB) con PKL de MediaPipe Holistic."
        ),
        "requiere_token": True,
        "instrucciones_token": (
            "1. Registrarse en https://datos.pucp.edu.pe\n"
            "2. Ir a Perfil → API Token → Create Token\n"
            "3. Copiar el token y pasarlo con --api-token TOKEN"
        ),
        "archivos_clave": [
            "Keypoints.rar  — PKL keypoints MediaPipe (139 MB) ← ESENCIAL",
            "dict.json      — mapeo glosa→instancias",
        ],
    },

    # datos.pucp.edu.pe — DGI156 completo
    "dgi156_pucp": {
        "nombre": "DGI156 completo — datos.pucp.edu.pe",
        "tipo":   "dataverse",
        "persistent_id": "hdl:20.500.12534/OJYYYS",
        "base_url": "https://datos.pucp.edu.pe",
        "destino": EXT_DIR / "dgi156_full",
        "descripcion": (
            "Dataset DGI156 completo con 156 glosas, 4072 instancias. "
            "Incluye PKL de pose estimation MediaPipe Holistic. "
            "Actualmente solo tenemos 28 de las 156 clases."
        ),
        "requiere_token": True,
        "instrucciones_token": (
            "1. Registrarse en https://datos.pucp.edu.pe\n"
            "2. Ir a Perfil → API Token → Create Token\n"
            "3. Copiar el token y pasarlo con --api-token TOKEN"
        ),
        "archivos_clave": [
            "pose_estimation/pkl/  — PKL MediaPipe Holistic por seña",
        ],
    },

    # datos.pucp.edu.pe — vocabulario_lsp_p
    "vocabulario_lsp_p_full": {
        "nombre": "vocabulario_lsp_p completo — datos.pucp.edu.pe",
        "tipo":   "dataverse_restricted",
        "persistent_id": "hdl:20.500.12534/JU4OLG",
        "base_url": "https://datos.pucp.edu.pe",
        "destino": EXT_DIR / "vocabulario_lsp_p_full",
        "descripcion": (
            "305 glosas LSP grabadas en estudio con múltiples señantes. "
            "Requiere solicitar acceso al equipo PUCP."
        ),
        "requiere_token": True,
        "requiere_acceso_especial": True,
        "contacto": "ldatos@pucp.edu.pe",
        "instrucciones_token": (
            "1. Enviar email a ldatos@pucp.edu.pe solicitando acceso al dataset vocabulario_lsp_p\n"
            "2. Una vez aprobado, seguir instrucciones de acceso del equipo PUCP\n"
            "3. Registrarse en https://datos.pucp.edu.pe y obtener API Token"
        ),
    },
}


# ─── helpers ──────────────────────────────────────────────────────────────────

def run(cmd: list, **kwargs) -> int:
    """Ejecuta un comando y retorna el código de salida."""
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, **kwargs)
    return result.returncode


def git_clone(url: str, destino: pathlib.Path, depth: int = 1) -> bool:
    """Clona un repositorio git. Retorna True si exitoso."""
    if destino.exists() and any(destino.iterdir()):
        print(f"  Ya existe: {destino}")
        resp = input("  ¿Actualizar (git pull)? [s/N]: ").strip().lower()
        if resp == "s":
            rc = run(["git", "-C", str(destino), "pull"])
            return rc == 0
        return True

    destino.parent.mkdir(parents=True, exist_ok=True)
    rc = run(["git", "clone", f"--depth={depth}", url, str(destino)])
    return rc == 0


def download_dataverse(base_url: str, persistent_id: str, destino: pathlib.Path,
                       api_token: str) -> bool:
    """Descarga archivos de un dataset en Dataverse usando la API."""
    api_url = f"{base_url}/api/datasets/:persistentId/?persistentId={persistent_id}"
    headers = {"X-Dataverse-key": api_token}

    req = urllib.request.Request(api_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            meta = json.loads(resp.read().decode())
    except Exception as e:
        print(f"  ERROR al acceder a Dataverse API: {e}")
        print("  Verifique su token y que tenga acceso al dataset.")
        return False

    files = meta.get("data", {}).get("latestVersion", {}).get("files", [])
    print(f"  Dataset tiene {len(files)} archivos")

    destino.mkdir(parents=True, exist_ok=True)

    for file_meta in files:
        label = file_meta["dataFile"]["filename"]
        file_id = file_meta["dataFile"]["id"]
        size_mb = file_meta["dataFile"].get("filesize", 0) / 1e6

        out_path = destino / label
        if out_path.exists():
            print(f"  [SKIP] {label} (ya existe)")
            continue

        print(f"  Descargando {label} ({size_mb:.1f} MB)...")
        dl_url = f"{base_url}/api/access/datafile/{file_id}"
        req_dl = urllib.request.Request(dl_url, headers=headers)
        try:
            with urllib.request.urlopen(req_dl, timeout=300) as resp:
                with open(out_path, "wb") as f:
                    shutil.copyfileobj(resp, f)
            print(f"  ✅ {label}")
        except Exception as e:
            print(f"  ERROR descargando {label}: {e}")

    return True


def extraer_rar_aec(destino: pathlib.Path):
    """Extrae Keypoints.rar de AEC si está disponible."""
    rar_path = destino / "Keypoints.rar"
    if not rar_path.exists():
        print(f"  No se encontró {rar_path}")
        return

    kp_dest = destino / "Keypoints"
    if kp_dest.exists():
        print(f"  Keypoints/ ya extraído en {kp_dest}")
        return

    print(f"  Extrayendo {rar_path}...")
    # Intentar con unrar o 7z
    for cmd in [["unrar", "x", str(rar_path), str(destino) + "/"],
                ["7z", "x", str(rar_path), f"-o{destino}"]]:
        rc = run(cmd)
        if rc == 0:
            print(f"  ✅ Extraído en {destino}")
            return

    print("  ERROR: instale 'unrar' o '7z' para extraer el archivo RAR")
    print("    macOS:  brew install unar   (o 'brew install p7zip')")
    print("    Linux:  sudo apt install unrar  (o p7zip-full)")


def copiar_pkls_a_keypoints(src_dir: pathlib.Path, dest_name: str):
    """Copia PKL encontrados en src_dir hacia data/Keypoints/{dest_name}/"""
    out_base = KP_DIR / dest_name
    n_copiados = 0
    for pkl_path in src_dir.rglob("*.pkl"):
        # Intenta inferir clase del path
        parts = pkl_path.parts
        clase_idx = [i for i, p in enumerate(parts) if p == "pkl"]
        if clase_idx:
            ci = clase_idx[-1]
            if ci + 1 < len(parts) - 1:
                clase = parts[ci + 1]
            else:
                clase = pkl_path.stem
        else:
            clase = pkl_path.parent.name

        dest = out_base / clase / pkl_path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(pkl_path, dest)
            n_copiados += 1

    print(f"  Copiados {n_copiados} PKL → data/Keypoints/{dest_name}/")
    return n_copiados


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Descarga datasets LSP")
    ap.add_argument("--fuente", choices=list(FUENTES.keys()),
                    help="Dataset a descargar")
    ap.add_argument("--api-token", default="",
                    help="Token API de datos.pucp.edu.pe (para fuentes Dataverse)")
    ap.add_argument("--listar", action="store_true",
                    help="Lista todas las fuentes disponibles")
    ap.add_argument("--copiar-pkls", action="store_true",
                    help="Después de descargar, copia PKL a data/Keypoints/")
    args = ap.parse_args()

    if args.listar:
        print("\n=== DATASETS LSP DISPONIBLES ===\n")
        for key, meta in FUENTES.items():
            token_str = "⚠ requiere token API" if meta.get("requiere_token") else "✅ público"
            especial  = "🔒 acceso restringido" if meta.get("requiere_acceso_especial") else ""
            print(f"  --fuente {key:<20}  {token_str}  {especial}")
            print(f"    {meta['nombre']}")
            print(f"    {meta['descripcion'][:80]}…")
            print()
        return

    if not args.fuente:
        ap.print_help()
        print("\nEjemplos:")
        print("  python3 scripts/download_lsp_datasets.py --listar")
        print("  python3 scripts/download_lsp_datasets.py --fuente pucp_github")
        print("  python3 scripts/download_lsp_datasets.py --fuente videolsp10")
        print("  python3 scripts/download_lsp_datasets.py --fuente aec_pucp --api-token TOKEN")
        return

    meta = FUENTES[args.fuente]
    print(f"\n{'='*60}")
    print(f"Descargando: {meta['nombre']}")
    print(f"{'='*60}\n")

    if meta.get("requiere_acceso_especial"):
        print(f"⚠  Este dataset requiere solicitud de acceso especial.")
        print(f"   Contacto: {meta.get('contacto','ldatos@pucp.edu.pe')}")
        print(f"\nInstrucciones:")
        print(meta["instrucciones_token"])
        return

    if meta.get("requiere_token") and not args.api_token:
        print("⚠  Este dataset requiere un token API de datos.pucp.edu.pe")
        print("\nCómo obtener el token:")
        print(meta["instrucciones_token"])
        print("\nLuego ejecutar:")
        print(f"  python3 scripts/download_lsp_datasets.py --fuente {args.fuente} --api-token TU_TOKEN")
        return

    destino = meta["destino"]

    if meta["tipo"] == "git":
        ok = git_clone(meta["url"], destino)
        if ok and args.copiar_pkls:
            copiar_pkls_a_keypoints(destino, args.fuente)

    elif meta["tipo"] in ("dataverse", "dataverse_restricted"):
        ok = download_dataverse(
            meta["base_url"], meta["persistent_id"],
            destino, args.api_token
        )
        if ok:
            # Para AEC, extraer el RAR automáticamente
            if args.fuente == "aec_pucp":
                extraer_rar_aec(destino)
            if args.copiar_pkls:
                copiar_pkls_a_keypoints(destino / "Keypoints", args.fuente)

    print(f"\n✅ Descarga completada: {destino}")
    if "nota_post_descarga" in meta:
        print(f"\nSiguiente paso:")
        print(f"  {meta['nota_post_descarga']}")

    print(f"\nPara integrar al dataset:")
    print(f"  python3 scripts/build_dataset_s16.py --help")


if __name__ == "__main__":
    main()
