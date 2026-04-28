# Multimodal Deepfake — production-oriented CPU image (Hetzner / Linux)
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    FFMPEG_QUIET=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY deploy/requirements-runtime.txt /app/deploy/requirements-runtime.txt

# CPU PyTorch (smaller image; GPU sunucuda bu satiri degistir)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r deploy/requirements-runtime.txt

COPY . /app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8501/ > /dev/null || exit 1

CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
