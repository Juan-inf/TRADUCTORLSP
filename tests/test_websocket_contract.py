"""Test de contrato del WebSocket — ver ENTREGABLE_PLAN_DE_DESPLIEGUE_S13.md §3 y §7.
No prueba precisión de reconocimiento — solo que el protocolo (buffering →
segmento cerrado) se cumple tal como está documentado.

Desde el port de R9 (2026-07-2x), /predict/stream ya no clasifica cada 30
frames fijos: usa SegmentadorPausas (src/features/segmentacion.py), el mismo
criterio que demo/app_gradio.py — cierra un segmento por pausa de manos o al
llegar a max_frames_seña=90 (tope duro). Por eso el test ya no puede asumir
que el frame 30 exacto dispara una predicción; en cambio, verifica que un
segmento SIEMPRE cierra dentro de ese tope, y que la respuesta resultante
respeta el filtro de confianza/clase narrativa (predicción real o
'below_threshold', nunca un campo suelto sin status)."""
import base64
import cv2


def _frame_b64(frame):
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def test_buffering_hasta_cierre_de_segmento(client, data_dir):
    path = data_dir / "videos" / "original" / "Historias vinetas (11).mp4"
    cap = cv2.VideoCapture(str(path))
    frames = []
    for _ in range(95):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    assert len(frames) >= 91, "el video de prueba no tiene suficientes frames"

    with client.websocket_connect("/predict/stream") as ws:
        cerrado = False
        collected_visto = 0
        for i, frame in enumerate(frames):
            ws.send_json({"frame": _frame_b64(frame), "include_landmarks": False})
            resp = ws.receive_json()

            if resp.get("status") == "buffering":
                # frames_collected se resetea a 0 cada vez que un segmento
                # cierra, así que solo se puede afirmar que es un entero
                # no negativo y coherente con el tope informativo de 30.
                assert isinstance(resp["frames_collected"], int)
                assert resp["frames_collected"] >= 0
                assert resp["frames_needed"] == 30
                collected_visto = max(collected_visto, resp["frames_collected"])
                continue

            # Cualquier respuesta que no sea "buffering" significa que
            # SegmentadorPausas cerró un segmento y ya pasó por el filtro
            # de confianza/clase narrativa.
            cerrado = True
            if resp.get("status") == "below_threshold":
                for campo in ("clase_descartada", "confidence"):
                    assert campo in resp, f"falta '{campo}' en below_threshold: {resp}"
            else:
                assert "status" not in resp, f"respuesta inesperada: {resp}"
                for campo in ("clase", "texto_castellano", "confidence", "latency_ms", "top3"):
                    assert campo in resp, f"falta '{campo}' en la predicción del WebSocket"
            break

        assert cerrado, (
            f"ningún segmento cerró en {len(frames)} frames (tope duro es 90) "
            f"— frames_collected llegó a {collected_visto}"
        )


def test_frame_invalido_devuelve_error(client):
    with client.websocket_connect("/predict/stream") as ws:
        ws.send_json({"frame": "no-es-base64-valido", "include_landmarks": False})
        resp = ws.receive_json()
        assert "error" in resp
