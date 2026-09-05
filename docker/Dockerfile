# Enterprise Text-to-SQL API
#
# Phase 13.
#
# Deliberately does NOT contain torch, transformers, peft or bitsandbytes. The
# default backend (MODEL_BACKEND=hf) calls a remotely served model, so the
# image stays around 200 MB instead of several gigabytes. Serving the
# fine-tuned adapter in-process needs a CUDA base image and those packages --
# see docker/README.md.
#
# Build and run from the PROJECT ROOT, not from docker/:
#     docker build -f docker/Dockerfile -t text2sql-api .

FROM python:3.11-slim AS base

# PYTHONDONTWRITEBYTECODE: no .pyc in a layer that is read-only anyway.
# PYTHONUNBUFFERED: logs appear in `docker logs` immediately rather than when
# a buffer happens to flush, which matters when diagnosing a failed start.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Dependencies before source: application code changes far more often than the
# dependency list, and this ordering keeps the expensive layer cached.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y pytest \
    && find /usr/local -name '__pycache__' -type d -prune -exec rm -rf {} +

# Only what the service imports at runtime. The dataset, the experiments, the
# training code and the 174 MB adapter are not needed to serve requests and
# would bloat the image.
COPY src/ ./src/

# A model-generated-SQL service should not also be running as root.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Hits the real readiness endpoint, which executes SELECT 1 against PostgreSQL.
# A container that is listening but cannot reach its database is not healthy,
# and a TCP check would call it healthy.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys,json; \
r=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=8)); \
sys.exit(0 if r.get('status')=='ok' else 1)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
