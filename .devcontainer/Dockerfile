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
    pip install poetry && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    echo 'export PATH=/home/vscode/.local/bin:$PATH' >> /home/vscode/.bashrc

# For NVIDIA GPU support (uncomment if using GPU):
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy project code
WORKDIR /workspace
COPY . /workspace

# Configure Poetry to create environment inside the project and install dependencies.
RUN poetry config virtualenvs.in-project true && \
    poetry install --no-interaction --with dev --with demo

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/workspace/.venv/bin:$PATH"

# Set up non-root user
RUN useradd -m -s /bin/bash devuser && \
    chown -R devuser:devuser /workspace

USER devuser