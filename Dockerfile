FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglu1-mesa libxrender1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir cadquery numpy scipy shapely ezdxf fastapi uvicorn python-multipart
WORKDIR /app
COPY . .
RUN test -f app.py || (echo "ERROR: app.py is missing or misnamed in the repository" && exit 1)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
