# Enterprise Text-to-SQL — Hugging Face Space image.
#
# Phase 14. Nearly identical to docker/Dockerfile; the differences are the two
# things a Space requires:
#
#   * it listens on 7860, and honours $PORT if the platform sets one, so the
#     same image also runs on Render, Railway or Fly without modification
#   * no HEALTHCHECK — the Space supervises the container itself
#
# Copy this to the Space repo root as `Dockerfile`, alongside `src/` and
# `requirements.txt`.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    PORT=7860 \
    MODEL_BACKEND=hf \
    HF_PROVIDER=featherless-ai \
    PGSSLMODE=require

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y pytest \
    && find /usr/local -name '__pycache__' -type d -prune -exec rm -rf {} +

COPY src/ ./src/

# Spaces run as uid 1000. Matching it avoids a container that cannot read its
# own files.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

# Shell form on purpose: $PORT has to expand at runtime, and exec form would
# pass the literal string.
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-7860}
