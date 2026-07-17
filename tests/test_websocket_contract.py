"""Test de contrato del WebSocket — ver ENTREGABLE_PLAN_DE_DESPLIEGUE_S13.md §3 y §7.
No prueba precisión de reconocimiento — solo que el protocolo (buffering →
predicción) se cumple tal como está documentado."""
import base64
import cv2


def _frame_b64(frame):
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def test_buffering_hasta_30_frames_luego_prediccion(client, data_dir):
    path = data_dir / "videos" / "original" / "Historias vinetas (11).mp4"
    cap = cv2.VideoCapture(str(path))
    frames = []
    for _ in range(35):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    assert len(frames) >= 31, "el video de prueba no tiene suficientes frames"

    with client.websocket_connect("/predict/stream") as ws:
        for i, frame in enumerate(frames[:29]):
            ws.send_json({"frame": _frame_b64(frame), "include_landmarks": False})
            resp = ws.receive_json()
            assert resp.get("status") == "buffering", f"frame {i}: se esperaba buffering, llegó {resp}"
            assert resp["frames_collected"] == i + 1
            assert resp["frames_needed"] == 30

        # frame 30: se completa el buffer, debe devolver una predicción real
        ws.send_json({"frame": _frame_b64(frames[29]), "include_landmarks": False})
        resp = ws.receive_json()
        assert "status" not in resp, f"se esperaba predicción, siguió en buffering: {resp}"
        for campo in ("clase", "texto_castellano", "confidence", "latency_ms", "top3"):
            assert campo in resp, f"falta '{campo}' en la predicción del WebSocket"


def test_frame_invalido_devuelve_error(client):
    with client.websocket_connect("/predict/stream") as ws:
        ws.send_json({"frame": "no-es-base64-valido", "include_landmarks": False})
        resp = ws.receive_json()
        assert "error" in resp
