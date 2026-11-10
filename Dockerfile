FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir . && pip install --no-cache-dir --no-deps basic-pitch==0.4.0
EXPOSE 8000
CMD ["python", "-m", "fretscribe", "serve", "--host", "0.0.0.0", "--port", "8000"]
