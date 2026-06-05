"""
deploy_huggingface.py — Despliega el Traductor LSP en HuggingFace Spaces.

Requisitos previos:
  pip install huggingface_hub
  huggingface-cli login   (o hf auth login)

Uso:
  python scripts/deploy_huggingface.py --repo TU_USUARIO/traductor-lsp
"""

import argparse, pathlib, sys
from huggingface_hub import HfApi, create_repo, upload_folder

ROOT       = pathlib.Path(__file__).parent.parent
SPACES_DIR = ROOT / "spaces"

ARCHIVOS = [
    "app.py",
    "requirements.txt",
    "lstm_signs.onnx",
    "lstm_label2idx.json",
    "README.md",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True,
                        help="Ej: juan-inf/traductor-lsp")
    args = parser.parse_args()

    api = HfApi()

    # Verificar autenticación
    try:
        user = api.whoami()
        print(f"Autenticado como: {user['name']}")
    except Exception:
        print("ERROR: No autenticado. Ejecuta primero:")
        print("  huggingface-cli login")
        sys.exit(1)

    # Verificar archivos
    for f in ARCHIVOS:
        path = SPACES_DIR / f
        if not path.exists():
            print(f"FALTA: {path}")
            sys.exit(1)
    print("✅ Todos los archivos listos en spaces/")

    # Crear Space si no existe
    repo_url = create_repo(
        repo_id=args.repo,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
        private=False,
    )
    print(f"Space: {repo_url}")

    # Subir archivos
    print("Subiendo archivos...")
    upload_folder(
        folder_path=str(SPACES_DIR),
        repo_id=args.repo,
        repo_type="space",
        ignore_patterns=["*.py.bak", "__pycache__"],
    )
    print(f"\n✅ Deploy completado:")
    print(f"   https://huggingface.co/spaces/{args.repo}")

if __name__ == "__main__":
    main()
