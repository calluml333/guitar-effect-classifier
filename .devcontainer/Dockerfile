# Use official Python runtime as base image
FROM mcr.microsoft.com/devcontainers/python:3.11-bullseye

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    sox \
    libsox-dev \
    libsox-fmt-all \
    git \
    vim \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support (CPU only for dev container by default)
# For GPU support, uncomment the appropriate torch installation command
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For NVIDIA GPU support (uncomment if using GPU):
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy project code
WORKDIR /workspace

# Copy requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install development dependencies
COPY pyproject.toml .
RUN pip install -e . && pip install -e ".[dev]"

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set up non-root user
RUN useradd -m -s /bin/bash devuser && \
    chown -R devuser:devuser /workspace

USER devuser