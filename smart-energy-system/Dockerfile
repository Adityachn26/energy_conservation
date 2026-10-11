# One image for both the backend and the dashboard (they differ only in the start command,
# set in docker-compose.yml). Only RUNTIME dependencies are installed — no pytest/notebooks.
FROM python:3.12-slim

# libgomp1 = OpenMP runtime required by XGBoost on Linux.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY api/ api/
COPY database/ database/
COPY dashboard/ dashboard/
COPY .streamlit/ .streamlit/
COPY scripts/ scripts/
COPY alembic.ini .

# Run as a non-root user.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000 8501
