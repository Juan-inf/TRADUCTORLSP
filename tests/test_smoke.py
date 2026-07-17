"""Smoke tests — ver ENTREGABLE_PLAN_DE_DESPLIEGUE_S13.md §7.
El servidor debe estar arriba y con el modelo cargado. No prueban precisión,
solo que el sistema responde correctamente en su forma (status, claves)."""


def test_health_responde_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok"
    assert j["model_ready"] is True
    assert j["n_classes"] == 96


def test_classes_devuelve_96_clases(client):
    r = client.get("/classes")
    assert r.status_code == 200
    j = r.json()
    assert j["total"] == 96
    assert len(j["classes"]) == 96
    assert "A" in j["classes"]  # letra del abecedario, siempre presente


def test_predict_video_formato_de_respuesta(client, data_dir):
    """No exige acierto — solo que la respuesta tenga la forma del contrato (§3)."""
    path = data_dir / "videos" / "original" / "Historias vinetas (11).mp4"
    with open(path, "rb") as f:
        r = client.post("/predict/video", files={"file": ("v.mp4", f, "video/mp4")})
    assert r.status_code == 200
    j = r.json()
    for campo in ("clase", "texto_castellano", "confidence", "latency_ms", "top3"):
        assert campo in j, f"falta el campo '{campo}' en la respuesta"
    assert isinstance(j["top3"], list) and len(j["top3"]) == 3
    assert 0.0 <= j["confidence"] <= 1.0
