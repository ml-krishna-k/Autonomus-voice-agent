# syntax=docker/dockerfile:1
#
# Models are baked into the image on purpose. Downloading ~250 MB of weights on
# every cold start is the single biggest avoidable latency in this stack.

FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf \
    MODELS_DIR=/app/models \
    OMP_NUM_THREADS=4

# libgomp is required by onnxruntime and CTranslate2; curl is for HEALTHCHECK.
# espeak-ng data ships inside the piper-tts wheel, so no apt package is needed.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgomp1 \
      curl \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first — this layer is cached across code changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Model download is its own layer so editing app code never re-downloads them.
COPY app/config.py ./app/config.py
COPY app/__init__.py ./app/__init__.py
COPY scripts/fetch_models.py ./scripts/fetch_models.py
RUN python scripts/fetch_models.py && chmod -R a+rX /opt/hf /app/models

COPY app/ ./app/
COPY scripts/ ./scripts/
COPY clients/ ./clients/

RUN useradd --create-home --uid 10001 agent && chown -R agent:agent /app
USER agent

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD curl -fsS http://localhost:8000/readyz || exit 1

# One worker per container. The models are ~1 GB resident; scale with replicas,
# not with `--workers`, or you will load a full copy per worker.
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-graceful-shutdown", "20"]
