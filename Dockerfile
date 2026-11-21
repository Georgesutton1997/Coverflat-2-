FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglu1-mesa libxrender1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# One worker only: jobs and uploads are kept in memory.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
