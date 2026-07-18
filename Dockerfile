FROM python:3.12-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install the requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and model weights
COPY api.py quality_check.py mobnet_v2.onnx mobnet_v2.onnx.data ./

# Expose the default port (Render will override via PORT env var)
EXPOSE 7860

# Run FastAPI using uvicorn, falling back to port 7860 if PORT is not set
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-7860}"]
