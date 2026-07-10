# Dev Container Setup

## Quick Start

### Option 1: VS Code Dev Container (Recommended)

1. **Install Prerequisites**
   - [VS Code](https://code.visualstudio.com/)
   - [Dev Containers Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
   - [Docker Desktop](https://www.docker.com/products/docker-desktop)

2. **Open in Dev Container**
   - Open the project in VS Code
   - Press `Cmd/Ctrl + Shift + P` → "Dev Containers: Open Folder in Container"
   - VS Code rebuilds the container and reloads
   - All tools and dependencies are ready!

### Option 2: Docker Compose

```bash
# Build and start the dev container
docker-compose -f .devcontainer/docker-compose.yml up -d

# Connect to container shell
docker-compose -f .devcontainer/docker-compose.yml exec dev bash

# Run commands in container
docker-compose -f .devcontainer/docker-compose.yml exec dev python scripts/generate_dataset.py
```

### Option 3: Local Development (Without Container)

```bash
poetry env use python3.11
poetry install --with dev --with demo
source .venv/bin/activate
```

## What's Included

- **Python 3.11**: Latest stable Python
- **PyTorch**: Deep learning framework (CPU by default, GPU optional)
- **Audio Libraries**: libsndfile, ffmpeg, sox
- **Development Tools**: Git, vim, curl, wget
- **VS Code Extensions**: Python, Pylance, Jupyter, Black formatter, Ruff linter
- **Development Dependencies**: pytest, black, flake8

## GPU Support

To enable GPU (NVIDIA CUDA):

1. Ensure NVIDIA Docker runtime is installed
2. In `.devcontainer/Dockerfile`, uncomment:
   ```dockerfile
   # RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
3. In `.devcontainer/docker-compose.yml`, uncomment the GPU deployment section
4. In `.devcontainer/devcontainer.json`, update runArgs if needed

## Common Commands

```bash
# Generate dataset
python scripts/generate_dataset.py --input-dir data/raw --out-dir data/generated

# Train model
python -m src.train --epochs 30 --batch-size 32

# Run tests
pytest tests/

# Format code
black src/ scripts/

# Lint code
flake8 src/ scripts/
```

## Python Path

The project uses `package-mode = false` in `pyproject.toml` — Poetry only manages
the dependency environment, it doesn't install `src` as a package. Scripts and
tests work because they're run from the repository root (or insert it onto
`sys.path` themselves), so `import src` resolves against the working directory:
```bash
python -c "import src; print(src.__version__)"
```

## Troubleshooting

**Container build fails**: 
- Ensure Docker has enough disk space
- Try `docker system prune` to clean up

**GPU not detected**:
- Verify NVIDIA Docker runtime: `docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi`
- Check CUDA version compatibility with PyTorch

**Slow builds**:
- Use BuildKit: `DOCKER_BUILDKIT=1 docker build ...`
- Cache Docker layers by avoiding `pip install` changes

## Resources

- [Dev Containers Documentation](https://containers.dev/)
- [VS Code Remote Development](https://code.visualstudio.com/docs/remote/remote-overview)
- [Docker Documentation](https://docs.docker.com/)
