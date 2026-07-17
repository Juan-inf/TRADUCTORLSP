"""Golden tests — ver ENTREGABLE_PLAN_DE_DESPLIEGUE_S13.md §7.

Casos elegidos empíricamente (2026-07-17): de 7 videos de viñeta probados,
solo 1/7 acertó en top-3 (clips largos, el muestreo uniforme a 30 frames
pierde demasiado contexto). Los clips CORTOS de una sola seña sí funcionan
bien — son la tarea real para la que el modelo fue entrenado y medido
(F1=0.4426). Por eso el set golden usa clips cortos, no viñetas completas.

Las etiquetas esperadas se leen de data/s27_label2idx.json en vez de
escribirse a mano en el código — un video con tildes (TÚ, PROTEÍNA) puede
lucir igual en pantalla y no ser `==` si la normalización Unicode difiere
entre el literal del código y la etiqueta real del modelo. Ya nos pasó
escribiendo este archivo.
"""
import json
import unicodedata


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _clase_real(nombre_clase: str, label2idx: dict) -> str:
    """Devuelve la clave EXACTA tal como está en el vocabulario del modelo,
    comparando en forma NFC — data/s27_label2idx.json guarda las tildes en
    forma decompuesta (ej. 'U' + acento combinante), que no es `==` a un
    literal con tilde precompuesta aunque se vean idénticas en pantalla."""
    for k in label2idx:
        if _nfc(k) == _nfc(nombre_clase):
            return k
    raise AssertionError(f"'{nombre_clase}' no está en el vocabulario de 96 clases")


def _label2idx(data_dir):
    return json.loads((data_dir / "s27_label2idx.json").read_text(encoding="utf-8"))


GOLDEN = [
    # (archivo relativo a data/_s13_tmp/..., nombre de clase esperado)
    ("_s13_tmp/aec_extracted/Videos/SEGMENTED_SIGN/ira_alegria/ahora_31.mp4", "AHORA"),
    ("_s13_tmp/aec_extracted/Videos/SEGMENTED_SIGN/proteinas_porcentajes/tú_1434.mp4", "TÚ"),
    ("_s13_tmp/aec_extracted/Videos/SEGMENTED_SIGN/proteinas_porcentajes/proteína_1385.mp4", "PROTEÍNA"),
]


def test_golden_clips_cortos_en_top3(client, data_dir):
    label2idx = _label2idx(data_dir)
    fallos = []
    for rel_path, esperado in GOLDEN:
        esperado = _clase_real(esperado, label2idx)
        path = data_dir / rel_path
        with open(path, "rb") as f:
            r = client.post("/predict/video", files={"file": (path.name, f, "video/mp4")})
        assert r.status_code == 200, f"{rel_path}: status {r.status_code}"
        top3 = [t["clase"] for t in r.json()["top3"]]
        if esperado not in top3:
            fallos.append(f"{path.name}: esperado={esperado!r} top3={top3!r}")
    assert not fallos, "Regresión en golden set:\n" + "\n".join(fallos)
