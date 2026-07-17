.PHONY: install run-demo run-api health test-video test docker-build docker-run

# Instala solo lo necesario para servir (no el entorno completo de entrenamiento)
install:
	python3.10 -m venv .venv310
	.venv310/bin/pip install -r requirements-api.txt

# Demo interactiva: cámara + video + TTS — ver ENTREGABLE_PLAN_DE_DESPLIEGUE_S13.md §5
run-demo:
	.venv310/bin/python demo/app_gradio.py

# Backend FastAPI + WebSocket
run-api:
	.venv310/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# Chequeo de salud rápido (requiere run-api corriendo en otra terminal)
health:
	curl -s http://localhost:8000/health

# Suite de tests (smoke + golden + contrato WebSocket) — ver tests/
test:
	.venv310/bin/python -m pytest tests/ -v

# Prueba E2E con un video de ejemplo real del repo
test-video:
	curl -F "file=@data/videos/original/Historias vinetas (11).mp4" \
	     http://localhost:8000/predict/video

# Sin probar en este entorno (sin Docker) — ver riesgo R5 en el plan de despliegue
docker-build:
	docker build -t traductor-lsp-api .

docker-run:
	docker run -p 8000:8000 traductor-lsp-api
