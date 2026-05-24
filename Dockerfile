# ============================================================
# 🐳 Dockerfile for ERAS AI Classifier API
# ============================================================
# This Dockerfile containerizes the FastAPI server so it can be 
# deployed to other machines, servers, or cloud environments.
# ============================================================

# Use a lightweight official Python image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for compiling some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install PyTorch CPU first to keep the image size small
# (If you need GPU support, replace with standard pip install or a CUDA base image)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY api/ ./api/
COPY models/ ./models/
COPY training/ ./training/

# Expose the port FastAPI runs on
EXPOSE 8000

# Health check to ensure the container is running and healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start the FastAPI server on 0.0.0.0 (all interfaces)
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
