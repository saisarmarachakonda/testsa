FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH="rfdetr_model.onnx"
ENV DISABLE_MODEL_SOURCE_CHECK=True

WORKDIR /app

# Install system dependencies (curl for probes, libgomp1 for openmp, libgl1/libglib for OpenCV/Paddle)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download PaddleOCR English detection and recognition models at build time
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)"

# Copy model and application code
COPY rfdetr_model.onnx .
COPY app.py .

# Create non-root user
RUN useradd -u 1000 appuser && chown -R appuser:appuser /app /root/.paddleocr 2>/dev/null || true
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
