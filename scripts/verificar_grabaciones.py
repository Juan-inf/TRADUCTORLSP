"""
verificar_grabaciones.py — Control de calidad de clips recién grabados.

Analiza cada MP4 en el directorio dado y reporta:
  - Frames donde MediaPipe NO detecta alguna mano
  - Clips demasiado cortos (< 10 frames con manos visibles)
  - Clips con nombre de archivo inválido (no sigue la convención)
  - Resumen por clase: cuántos clips válidos hay

Uso:
    .venv311/bin/python3 scripts/verificar_grabaciones.py data/NuevasGrabaciones/
    .venv311/bin/python3 scripts/verificar_grabaciones.py data/NuevasGrabaciones/ --mover-rechazados
"""

import cv2, argparse, pathlib, re, sys
import mediapipe as mp
from collections import defaultdict

mp_holistic = mp.solutions.holistic

NOMBRE_RE = re.compile(r"^([A-ZÁÉÍÓÚÜÑ0-9_\-\.]+)_(S\d{2})_(\d{3})\.mp4$", re.IGNORECASE)
MIN_FRAMES_CON_MANOS = 10


def verificar_clip(mp4_path: pathlib.Path) -> dict:
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return {"ok": False, "motivo": "no_abre", "frames_total": 0, "frames_con_manos": 0}

    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_con_manos = 0
    frames_sin_izq = 0
    frames_sin_der = 0

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)
            tiene_izq = res.left_hand_landmarks is not None
            tiene_der = res.right_hand_landmarks is not None
            if tiene_izq or tiene_der:
                frames_con_manos += 1
            if not tiene_izq:
                frames_sin_izq += 1
            if not tiene_der:
                frames_sin_der += 1

    cap.release()

    ok = frames_con_manos >= MIN_FRAMES_CON_MANOS
    motivo = None
    if not ok:
        motivo = f"solo {frames_con_manos} frames con manos detectadas (mínimo {MIN_FRAMES_CON_MANOS})"

    return {
        "ok": ok,
        "motivo": motivo,
        "frames_total": frames_total,
        "frames_con_manos": frames_con_manos,
        "frames_sin_izq": frames_sin_izq,
        "frames_sin_der": frames_sin_der,
    }


def main():
    parser = argparse.ArgumentParser(description="Verifica calidad de clips grabados")
    parser.add_argument("directorio", type=pathlib.Path, help="Directorio raíz con MP4")
    parser.add_argument("--mover-rechazados", action="store_true",
                        help="Mover clips rechazados a subcarpeta _rechazados/")
    args = parser.parse_args()

    root = args.directorio
    if not root.exists():
        print(f"ERROR: {root} no existe"); sys.exit(1)

    mp4s = sorted(root.rglob("*.mp4"))
    if not mp4s:
        print(f"No se encontraron MP4 en {root}"); sys.exit(0)

    print(f"Verificando {len(mp4s)} clips en {root} …\n")

    por_clase = defaultdict(lambda: {"validos": 0, "rechazados": []})
    rechazados_global = []
    nombre_invalido = []

    for mp4 in mp4s:
        m = NOMBRE_RE.match(mp4.name)
        if not m:
            nombre_invalido.append(mp4)
            continue

        clase, signer, rep = m.group(1), m.group(2), m.group(3)
        resultado = verificar_clip(mp4)

        pct = resultado["frames_con_manos"] / max(resultado["frames_total"], 1) * 100
        estado = "✅" if resultado["ok"] else "❌"
        print(f"  {estado}  {mp4.name:<45}  "
              f"manos={resultado['frames_con_manos']}/{resultado['frames_total']} ({pct:.0f}%)"
              + (f"  ← {resultado['motivo']}" if not resultado["ok"] else ""))

        if resultado["ok"]:
            por_clase[clase]["validos"] += 1
        else:
            por_clase[clase]["rechazados"].append(mp4.name)
            rechazados_global.append(mp4)
            if args.mover_rechazados:
                dest_dir = mp4.parent / "_rechazados"
                dest_dir.mkdir(exist_ok=True)
                mp4.rename(dest_dir / mp4.name)

    # ── Resumen ──────────────────────────────────────────────────────────────────

    print("\n" + "=" * 65)
    print("RESUMEN POR CLASE")
    print("=" * 65)
    print(f"  {'Clase':<30} {'Válidos':>8} {'Rechazados':>12}")
    print(f"  {'-'*52}")
    for clase in sorted(por_clase):
        v = por_clase[clase]["validos"]
        r = len(por_clase[clase]["rechazados"])
        alerta = "  ⚠ < 50" if v < 50 else ""
        print(f"  {clase:<30} {v:>8} {r:>12}{alerta}")

    total_validos    = sum(d["validos"] for d in por_clase.values())
    total_rechazados = len(rechazados_global)

    print(f"\n  Total válidos   : {total_validos}")
    print(f"  Total rechazados: {total_rechazados}")
    print(f"  Nombre inválido : {len(nombre_invalido)}")

    if nombre_invalido:
        print("\n⚠  Archivos con nombre que NO sigue la convención <CLASE>_<S##>_<###>.mp4:")
        for f in nombre_invalido:
            print(f"     {f.relative_to(root)}")

    clases_bajo_50 = [c for c, d in por_clase.items() if d["validos"] < 50]
    if clases_bajo_50:
        print(f"\n⚠  {len(clases_bajo_50)} LSP - Vocabulario-palabras aún por debajo de 50 muestras válidas:")
        for c in sorted(clases_bajo_50):
            v = por_clase[c]["validos"]
            faltantes = 50 - v
            print(f"     {c:<30} ({v} grabados, faltan {faltantes})")

    if args.mover_rechazados and rechazados_global:
        print(f"\n  {total_rechazados} clips rechazados movidos a subcarpeta _rechazados/")

    print()


if __name__ == "__main__":
    main()
