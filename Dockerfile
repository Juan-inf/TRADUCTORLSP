# Backend FastAPI de inferencia LSP (api/main.py) — BiLSTM S27 + MediaPipe.
# La demo Gradio (demo/app_gradio.py) se despliega por separado vía HuggingFace
# Spaces (ver spaces/), no necesita Docker.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/__init__.py src/__init__.py
COPY src/inference/ src/inference/
COPY src/features/ src/features/
COPY api/ api/
COPY checkpoints/bilstm_s27.onnx checkpoints/bilstm_s27.onnx
COPY data/s27_label2idx.json data/s27_label2idx.json
COPY data/clase_texto.json data/clase_texto.json

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
