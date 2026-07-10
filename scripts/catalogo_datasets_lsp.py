"""catalogo_datasets_lsp.py

Catálogo completo de datasets de Lengua de Señas Peruana (LSP) disponibles
para el proyecto TRADUCTOR_LSP.

Genera un JSON detallado con metadatos, estado de integración, URLs de descarga,
y una hoja de ruta para expandir el dataset.

Uso:
    python3 scripts/catalogo_datasets_lsp.py [--actualizar]

Flags:
    --actualizar   Re-cuenta muestras locales desde los PKL disponibles
"""

import argparse
import json
import os
import glob
import pathlib
from datetime import datetime

ROOT    = pathlib.Path(__file__).resolve().parent.parent
DATA    = ROOT / "data"
KP_DIR  = DATA / "Keypoints"

# ─── Catálogo principal ────────────────────────────────────────────────────────

CATALOGO = {
    "fecha_actualizacion": None,        # se rellena al ejecutar
    "dataset_activo": "dataset_s15.npz",
    "resumen": {
        "muestras_totales": 13408,
        "clases_totales":   255,
        "clases_min15":     101,
        "muestras_min15":   12259,
        "fuentes":          ["dgi156", "vineta", "abecedario", "aec", "vocabulario_lsp_p", "glosa"],
    },
    "problema_estructural": {
        "descripcion": (
            "GroupShuffleSplit(SEED=42, test_size=0.20) asigna el 20% de los "
            "grupos de señantes al holdout HE3. El abecedario tiene un grupo por "
            "letra (1 señante), por lo que 6 letras (A,E,L,N,T,X) quedan con "
            "0 muestras de entrenamiento → F1=0 en holdout garantizado."
        ),
        "clases_afectadas": ["A", "E", "L", "N", "T", "X", "HISTORIAS_VINETAS_3"],
        "muestras_holdout_afectadas": 1687,
        "porcentaje_holdout": "66%",
        "solucion_requerida": (
            "Obtener datos de letras A,E,L,N,T,X de señantes DISTINTOS al "
            "señante actual del abecedario, para que GroupShuffleSplit pueda "
            "repartir señantes entre train y holdout."
        ),
    },
    "fuentes": {

        # ── 1. PUCP-DGI156 ───────────────────────────────────────────────────
        "PUCP_DGI156": {
            "nombre": "LSP PUCP-DGI156",
            "descripcion": (
                "Dataset de glosas LSP grabadas en contexto discursivo (Historias Viñetas). "
                "156 glosas distintas, múltiples señantes de la comunidad sorda peruana (CEDRO, Lima)."
            ),
            "institucion": "PUCP – Pontificia Universidad Católica del Perú",
            "autores": ["Gissela Quispe", "Joe Huamani", "Esau Vilca", "Guillermo Kemper"],
            "año":     2022,
            "paper":   "https://aclanthology.org/2022.signlang-1.1/",
            "repositorio": "https://github.com/gissemari/PeruvianSignLanguage",
            "descarga": {
                "url":    "https://datos.pucp.edu.pe/dataset.xhtml?persistentId=hdl:20.500.12534/OJYYYS&version=1.1",
                "tipo":   "Repositorio institucional PUCP (registro requerido)",
                "archivos_clave": [
                    "pose_estimation/pkl/  — keypoints MediaPipe Holistic 27 puntos",
                    "videos_segmented/     — MP4 por seña",
                    "subtitles/SRT/        — anotaciones temporales",
                ],
            },
            "contenido": {
                "glosas_unicas":    156,
                "instancias_video": 4072,
                "señantes":         "múltiples (comunidad sorda Lima)",
                "formato_kp":       "PKL MediaPipe Holistic (pose 33kp + manos 21kp c/u)",
                "frames_por_seq":   30,
                "duracion_seña_s":  "0.5–4.0",
            },
            "estado_local": {
                "carpeta": "data/Keypoints/dgi156_pkl/",
                "clases_descargadas": 28,
                "muestras_descargadas": 3642,
                "clases_disponibles_online": 156,
                "muestras_disponibles_online": 4072,
                "pendiente": "Descargar las 128 clases/señas restantes del repositorio PUCP",
            },
            "uso_en_s15": {
                "incluida": True,
                "fuente_label": "dgi156",
                "clases_en_s15": 28,
                "muestras_en_s15": 3642,
                "nota": (
                    "Solo HISTORIAS_VINETAS (clases narrativas). "
                    "Falta integrar las 128 glosas aisladas del DGI156 (HOMBRE, MUJER, CAMINAR, …)"
                ),
            },
            "aporte_potencial": {
                "clases_nuevas":        128,
                "muestras_nuevas_est":  3000,
                "mejora_holdout":       "NO mejora directamente A,E,L,N,T,X (DGI156 no incluye abecedario)",
                "mejora_vocabulario":   "SÍ — añade ~128 glosas frecuentes del LSP",
            },
        },

        # ── 2. AEC / PeruSIL ─────────────────────────────────────────────────
        "AEC_PeruSIL": {
            "nombre": "LSP Aprendo en Casa 2020 (AEC) – PeruSIL Framework",
            "descripcion": (
                "Interpretación continua de LSP extraída del programa educativo televisivo "
                "'Aprendo en Casa' emitido durante la pandemia (2020). Dos intérpretes "
                "principales (Isabel y Moisés). Anotado por voluntarios con conocimiento "
                "intermedio de LSP."
            ),
            "institucion": "PUCP + Ministerio de Educación del Perú (MED)",
            "autores": ["Bejarano et al. 2022", "Fernando Alva-Manchego"],
            "año":     2022,
            "paper":   "https://aclanthology.org/2022.signlang-1.1/",
            "repositorio": "https://github.com/gissemari/PeruvianSignLanguage",
            "descarga": {
                "url":    "https://datos.pucp.edu.pe/dataset.xhtml?persistentId=hdl:20.500.12534/HDOAGH",
                "tipo":   "Repositorio institucional PUCP (registro requerido)",
                "archivos_clave": [
                    "Keypoints.rar  — keypoints segmentados (139 MB)",
                    "dict.json      — diccionario glosa→instancias",
                ],
            },
            "contenido": {
                "glosas_unicas":    506,
                "instancias_video": 2311,
                "señantes":         2,
                "formato_kp":       "PKL dict {pose, left_hand, right_hand} x 30 frames",
                "frames_por_seq":   30,
                "episodios_fuente": 945,
                "materia":          "Todos los grados – Matemática, Comunicación, Personal Social, …",
            },
            "estado_local": {
                "carpeta": "data/Keypoints/aec_pkl/  (también data/external_lsp/aec/Keypoints/)",
                "clases_descargadas": 258,
                "muestras_descargadas": 830,
                "clases_disponibles_online": 506,
                "muestras_disponibles_online": 2311,
                "pendiente": "Descomprimir Keypoints.rar completo y reprocesar con build_dataset_s16",
            },
            "uso_en_s15": {
                "incluida": True,
                "fuente_label": "aec",
                "clases_en_s15": 258,
                "muestras_en_s15": 830,
                "nota": "Solo ~36% de las instancias disponibles están integradas",
            },
            "aporte_potencial": {
                "clases_nuevas":     248,
                "muestras_nuevas_est": 1481,
                "mejora_holdout": (
                    "PARCIAL — AEC tiene 'O' como letra pero no A,E,L,N,T,X. "
                    "Mejora HISTORIAS_VINETAS si se obtienen señas de los mismos "
                    "tipos desde diferentes episodios."
                ),
                "mejora_vocabulario": "SÍ — añade ~248 glosas nuevas del contexto educativo",
            },
        },

        # ── 3. vocabulario_lsp_p ───────────────────────────────────────────────────────
        "vocabulario_lsp_p": {
            "nombre": "LSP PUCP 305 (Glosas)",
            "descripcion": (
                "Dataset de 305 glosas LSP grabadas en estudio por señantes de la comunidad "
                "sorda de Lima. Usado para entrenar el Diccionario de LSP de PUCP. "
                "Múltiples señantes por seña."
            ),
            "institucion": "PUCP",
            "autores": ["Kemper et al."],
            "año":     2021,
            "paper":   "https://datos.pucp.edu.pe/dataset.xhtml?persistentId=hdl:20.500.12534/JU4OLG",
            "repositorio": "https://github.com/gissemari/PeruvianSignLanguage",
            "descarga": {
                "url":    "https://datos.pucp.edu.pe/dataset.xhtml?persistentId=hdl:20.500.12534/JU4OLG",
                "tipo":   "Repositorio institucional PUCP (requiere registro)",
                "archivos_clave": [
                    "ZIP ~2.7 GB — videos MP4 por seña",
                    "ELAN annotations (.eaf)",
                    "keypoints_pkl/ — PKL MediaPipe (si disponible)",
                ],
                "nota_acceso": (
                    "Requiere solicitar acceso en datos.pucp.edu.pe. "
                    "Contactar: ldatos@pucp.edu.pe"
                ),
            },
            "contenido": {
                "glosas_unicas":    305,
                "instancias_video": "~600–900 (estimado)",
                "señantes":         "3–5 por glosa",
                "formato_kp":       "PKL MediaPipe Holistic (misma estructura DGI156)",
                "frames_por_seq":   30,
            },
            "estado_local": {
                "carpeta": "data/Keypoints/vocabulario_lsp_p_pkl/",
                "clases_descargadas": 110,
                "muestras_descargadas": 224,
                "clases_disponibles_online": 305,
                "muestras_disponibles_online": "~900",
                "pendiente": (
                    "Descargar ZIP completo (2.7 GB). "
                    "Solo 110/305 clases con ~2 muestras/clase — insuficiente."
                ),
            },
            "uso_en_s15": {
                "incluida": True,
                "fuente_label": "vocabulario_lsp_p",
                "clases_en_s15": 110,
                "muestras_en_s15": 224,
                "nota": "Solo subset parcial; media de 2 muestras/clase — muy bajo",
            },
            "aporte_potencial": {
                "clases_nuevas":     195,
                "muestras_nuevas_est": 680,
                "mejora_holdout":    "NO directamente (vocabulario_lsp_p no incluye abecedario)",
                "mejora_vocabulario": "SÍ — añade 195 glosas nuevas de vocabulario básico LSP",
            },
        },

        # ── 4. Abecedario LSP (actual) ────────────────────────────────────────
        "ABECEDARIO_LSP": {
            "nombre": "Abecedario LSP (señante único)",
            "descripcion": (
                "24 letras del abecedario dactilológico LSP. Grabadas con un señante "
                "en entorno controlado. 150 muestras/letra. Cada letra = 1 grupo. "
                "PROBLEMA: sin diversidad de señantes → letras van COMPLETAS al holdout HE3."
            ),
            "institucion": "Colección interna del proyecto",
            "descarga": {
                "url": "LOCAL — data/Keypoints/abecedario_pkl/",
                "tipo": "Ya disponible localmente",
            },
            "contenido": {
                "letras":       24,
                "muestras":     3600,
                "señantes":     1,
                "nota_faltante": "Letra J (dinámica) y Z no incluidas",
            },
            "estado_local": {
                "carpeta":              "data/Keypoints/abecedario_pkl/",
                "clases_descargadas":   24,
                "muestras_descargadas": 3600,
                "pendiente":            "Obtener más señantes para las mismas letras",
            },
            "uso_en_s15": {
                "incluida": True,
                "fuente_label": "abecedario",
                "clases_en_s15": 24,
                "muestras_en_s15": 3600,
                "nota": (
                    "1 señante → 1 grupo por letra → 6 letras (A,E,L,N,T,X) van "
                    "al holdout con 0 muestras training. CAUSA RAÍZ del fallo HE3."
                ),
            },
            "aporte_potencial": {
                "accion_requerida": (
                    "Grabar 2–3 señantes adicionales haciendo las 24 letras (A-Y) "
                    "con MediaPipe Holistic → procesarlas con pkl_to_sequence() → "
                    "asignar grupos distintos al señante original. "
                    "Estimado: 2 señantes × 24 letras × 50 muestras = 2,400 muestras nuevas."
                ),
                "impacto_en_he3": "ALTO — resuelve el problema estructural para las letras",
            },
        },

        # ── 5. VideoLSP10 (GitHub) ────────────────────────────────────────────
        "VideoLSP10": {
            "nombre": "VideoLSP10 – 10 frases LSP",
            "descripcion": (
                "10 frases en LSP grabadas con Kinect v1 (skeleton Kinect, NO MediaPipe). "
                "21 clases (phrases), 81 secuencias cada una. Formato esqueleto Kinect "
                "con 10 puntos de unión (no compatible directamente con MediaPipe Holistic)."
            ),
            "institucion": "VideoLSP (Yuri Vladimir Huallpa Vargas)",
            "año":         2020,
            "licencia":    "CC BY-NC 4.0",
            "repositorio": "https://github.com/videoLSP/VideoLSP10",
            "descarga": {
                "url":  "https://github.com/videoLSP/VideoLSP10",
                "tipo": "GitHub público (clone directo)",
                "archivos_clave": [
                    "skeletonLSP10/ — coordenadas xyz de 10 joints Kinect",
                    "depthLSP/       — frames de profundidad",
                    "LSP10/          — RGB frames",
                ],
            },
            "contenido": {
                "frases":       10,
                "instancias":   "81 por frase = 810 total",
                "señantes":     "~10",
                "formato_kp":   "Kinect skeleton (10 joints: manos, muñecas, codos, hombros, cabeza)",
                "nota_formato": (
                    "NO compatible con pipeline actual (requiere conversión "
                    "de Kinect→MediaPipe o reentrenamiento con formato diferente)."
                ),
            },
            "estado_local": {
                "descargado": False,
                "pendiente":  "Clonar repo y evaluar conversión de formato",
            },
            "uso_en_s15": {
                "incluida": False,
                "razon":    "Formato Kinect incompatible con PKL MediaPipe del proyecto",
            },
            "aporte_potencial": {
                "clases_nuevas": 10,
                "muestras_nuevas_est": 810,
                "mejora_holdout": "NO — no incluye letras problemáticas",
                "compatibilidad": "BAJA — requiere script de conversión Kinect→MediaPipe",
            },
        },

        # ── 6. LSA64 (argentino, en el proyecto) ─────────────────────────────
        "LSA64": {
            "nombre": "LSA64 – Lengua de Señas Argentina (referencia)",
            "descripcion": (
                "64 señas de Lengua de Señas Argentina (LSA), NO peruana. "
                "Incluido en sprints anteriores como referencia de transferencia, "
                "pero removido en S15 por no ser LSP."
            ),
            "pais": "Argentina",
            "compatibilidad_lsp": "BAJA — LSA y LSP tienen diferencias significativas",
            "uso_en_s15": {
                "incluida": False,
                "razon":    "Removido en S15 — aumenta confusión al mezclar LSA con LSP",
            },
        },

        # ── 7. SEÑAS NUEVAS (registros pendientes) ────────────────────────────
        "NUEVOS_SEÑANTES_PROPIOS": {
            "nombre": "Grabación propia de señantes adicionales",
            "descripcion": (
                "Solución más directa al problema estructural de HE3: grabar "
                "2-3 señantes adicionales ejecutando las letras A,E,L,N,T,X "
                "y otras señas de las clases con 0 entrenamiento."
            ),
            "estado_local": {
                "descargado": False,
                "pendiente": "Planificar sesión de grabación con señantes de la comunidad sorda",
            },
            "protocolo_sugerido": {
                "software": "MediaPipe Holistic (Python 3.10 + mediapipe 0.10.x)",
                "script_generacion": "scripts/grabar_keypoints.py (pendiente de crear)",
                "señas_prioritarias": ["A", "E", "L", "N", "T", "X", "HISTORIAS_VINETAS_3"],
                "muestras_objetivo": "50+ por seña por señante",
                "señantes_objetivo": 3,
                "formato_salida": "PKL dict {pose:{x,y}, left_hand:{x,y}, right_hand:{x,y}} × 30 frames",
            },
            "aporte_potencial": {
                "impacto_en_he3": "MUY ALTO — resuelve directamente el problema estructural",
                "tiempo_estimado": "2 horas de grabación + 1 hora de procesamiento",
            },
        },
    },

    "hoja_de_ruta": {
        "prioridad_1_inmediata": {
            "accion": "Descomprimir y re-integrar AEC completo (Keypoints.rar → 2311 instancias)",
            "impacto": "Añade ~1481 muestras nuevas + 248 clases nuevas sin descarga adicional",
            "comando": "python3 scripts/build_dataset_s16.py --fuentes dgi156,aec,abecedario,vocabulario_lsp_p,glosa",
            "tiempo_est": "30 minutos",
        },
        "prioridad_2_descarga": {
            "accion": "Descargar DGI156 completo (128 glosas faltantes) desde datos.pucp.edu.pe",
            "url":    "https://datos.pucp.edu.pe/dataset.xhtml?persistentId=hdl:20.500.12534/OJYYYS",
            "impacto": "Añade ~3000 muestras + 128 glosas LSP de contexto discursivo",
            "requiere": "Registro en datos.pucp.edu.pe (gratuito)",
            "tiempo_est": "2 horas de descarga",
        },
        "prioridad_3_abecedario": {
            "accion": "Grabar 2-3 señantes adicionales ejecutando abecedario (A-Y)",
            "impacto": "RESUELVE el problema HE3 estructural para las letras A,E,L,N,T,X",
            "herramienta": "MediaPipe Holistic + scripts/grabar_keypoints.py (a crear)",
            "tiempo_est": "1 día",
        },
        "prioridad_4_vocabulario_lsp_p": {
            "accion": "Solicitar acceso al vocabulario_lsp_p completo (305 glosas, ~900 muestras)",
            "url":    "https://datos.pucp.edu.pe/dataset.xhtml?persistentId=hdl:20.500.12534/JU4OLG",
            "impacto": "Añade 195 clases nuevas con múltiples señantes → mejora diversidad",
            "requiere": "Registro + solicitud de acceso (ldatos@pucp.edu.pe)",
            "tiempo_est": "3-5 días (proceso de solicitud)",
        },
    },

    "impacto_esperado_con_todo": {
        "muestras_totales_est":  "18,000–22,000",
        "clases_totales_est":    "400–550",
        "clases_min15_est":      "200–280",
        "mejora_holdout": (
            "Con señantes adicionales para el abecedario, F1-holdout puede subir de "
            "0.026 (S19) a 0.15+ → HE3 alcanzable si F1-test ≥ 0.35."
        ),
    },
}


# ─── análisis local ───────────────────────────────────────────────────────────

def contar_muestras_locales():
    """Cuenta muestras PKL disponibles por fuente."""
    resultado = {}
    if not KP_DIR.exists():
        return resultado
    for src_dir in sorted(KP_DIR.iterdir()):
        if not src_dir.is_dir():
            continue
        n_clases = 0
        n_muestras = 0
        for clase_dir in src_dir.iterdir():
            if clase_dir.is_dir():
                pkls = list(clase_dir.glob("*.pkl"))
                if pkls:
                    n_clases += 1
                    n_muestras += len(pkls)
        resultado[src_dir.name] = {
            "clases": n_clases,
            "muestras": n_muestras,
        }
    return resultado


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actualizar", action="store_true",
                    help="Re-cuenta muestras locales y actualiza el catálogo")
    args = ap.parse_args()

    catalogo = CATALOGO.copy()
    catalogo["fecha_actualizacion"] = datetime.now().isoformat(timespec="seconds")

    if args.actualizar:
        print("Contando muestras locales…")
        local_counts = contar_muestras_locales()
        catalogo["conteo_local_actualizado"] = local_counts
        print("Conteo:")
        for src, info in local_counts.items():
            print(f"  {src}: {info['clases']} clases, {info['muestras']} muestras")

    out_path = DATA / "catalogo_datasets_lsp.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(catalogo, f, ensure_ascii=False, indent=2)

    print(f"\n✅  Catálogo guardado en: {out_path}")
    print()
    print("=" * 60)
    print("  DATASETS LSP DISPONIBLES Y SU ESTADO")
    print("=" * 60)

    rows = [
        ("PUCP-DGI156",  156,  4072, 28,  3642, "datos.pucp.edu.pe",     "⬇ 128 clases pendientes"),
        ("AEC/PeruSIL",  506,  2311, 258,  830, "datos.pucp.edu.pe",     "⬇ 1481 muestras pendientes"),
        ("vocabulario_lsp_p",     305,  "~900",110,  224, "datos.pucp.edu.pe",    "⬇ solicitar acceso completo"),
        ("Abecedario",    24,  3600,  24, 3600, "LOCAL",                  "⚠ 1 señante — diversidad crítica"),
        ("VideoLSP10",    10,   810,   0,    0, "github.com/videoLSP",   "⚠ formato Kinect — conversión requerida"),
        ("LSA64",         64,  3200,   0,    0, "N/A — excluido S15",   "✗ LSA ≠ LSP"),
        ("Nuevos señantes","?","?",  0,    0, "Grabación propia",       "🎯 MÁXIMO IMPACTO HE3"),
    ]

    print(f"\n{'Dataset':<18} {'Online':>8} {'Local':>8}  {'Fuente descarga':<24} {'Acción'}")
    print("-" * 90)
    for name, n_online, m_online, n_local, m_local, url, accion in rows:
        print(f"{name:<18} {str(n_online)+' cls':>8} {str(n_local)+' cls':>8}  {url:<24} {accion}")

    print()
    print("HOJA DE RUTA (por prioridad):")
    for k, v in catalogo["hoja_de_ruta"].items():
        print(f"\n  [{k.upper()}]")
        print(f"    Acción:  {v['accion']}")
        if "impacto" in v:
            print(f"    Impacto: {v['impacto']}")
        if "url" in v:
            print(f"    URL:     {v['url']}")
        print(f"    Tiempo:  {v.get('tiempo_est','?')}")

    print()
    print("Con todas las fuentes integradas:")
    est = catalogo["impacto_esperado_con_todo"]
    print(f"  Muestras estimadas: {est['muestras_totales_est']}")
    print(f"  Clases estimadas:   {est['clases_totales_est']}")
    print(f"  Mejora HE3:         {est['mejora_holdout'][:80]}…")


if __name__ == "__main__":
    main()
