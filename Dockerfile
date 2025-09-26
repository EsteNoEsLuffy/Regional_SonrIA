# ---- Base ----
FROM python:3.11-slim

# Evitar caché y prompts
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libs del SO
# Quita lo que no necesites; deja ffmpeg/libsndfile si tu pipeline lo usa
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# ---- App ----
WORKDIR /app

# Dependencias
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Código: api.py y cualquier módulo local (p.ej. stress_detector_api/)
COPY . /app

# Hugging Face asigna $PORT; no lo fijes a mano
CMD ["bash", "-lc", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-7860}"]
