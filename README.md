# Guitar Effect Classifier

A small end-to-end audio classification project for identifying common guitar effects from short audio clips. The current implementation combines synthetic dataset generation, audio preprocessing, pretrained audio embeddings, a lightweight PyTorch classifier, and a Streamlit demo.

## Objective

The goal of this project is to classify guitar recordings into common effect categories such as:

- clean
- overdrive
- distortion
- fuzz
- chorus
- delay
- reverb

The project is designed as a portfolio-style machine learning workflow with a clear structure for data generation, model training, evaluation, and inference.

## What is implemented so far

- Synthetic dataset generation using DSP-style effect processing
- Audio preprocessing utilities for resampling, mono conversion, normalization, trimming/padding, and log-mel extraction
- Dataset and dataloader support for waveform, log-mel, and Hugging Face embedding features
- Training and validation loops with checkpoint saving
- Inference CLI for single-file prediction
- Streamlit demo app for interactive uploads and predictions
- Smoke tests covering preprocessing and dataset loading

## Project structure

- [app/streamlit_app.py](app/streamlit_app.py) — interactive Streamlit demo
- [scripts/generate_dataset.py](scripts/generate_dataset.py) — generate labeled synthetic audio data
- [scripts/predict.py](scripts/predict.py) — command-line inference entry point
- [src/audio_processing.py](src/audio_processing.py) — audio loading and feature extraction helpers
- [src/dataset.py](src/dataset.py) — dataset class for manifest-based loading
- [src/features.py](src/features.py) — Hugging Face audio embedding wrapper
- [src/model.py](src/model.py) — classifier head definition
- [src/predict.py](src/predict.py) — inference logic and checkpoint loading
- [src/train.py](src/train.py) — training loop and checkpoint saving
- [tests/](tests/) — smoke tests for core functionality

## Setup

This project uses Python 3.11 and Poetry.

1. Install Poetry if needed.
2. Create or select the project environment:

```bash
cd /workspaces/guitar-effect-classifier
poetry env use /usr/local/bin/python3
```

3. Install dependencies:

```bash
poetry install --with dev --with demo
```

4. Activate the environment if desired:

```bash
source .venv/bin/activate
```

## Generate synthetic data

Place clean guitar audio files under [data/raw](data/raw), then run:

```bash
poetry run python scripts/generate_dataset.py \
  --input-dir data/raw \
  --out-dir data/generated \
  --manifest data/manifest.csv
```

This will create processed audio files and a manifest CSV containing labels and effect parameters.

## Train the model

Training uses the manifest generated above:

```bash
poetry run python -m src.train \
  --manifest data/manifest.csv \
  --feature hf \
  --epochs 10 \
  --out-dir models
```

A checkpoint will be saved to [models](models) as [models/best.pth](models/best.pth) when training completes.

## Run inference

Predict an effect from a single audio file:

```bash
poetry run python scripts/predict.py \
  --audio path/to/audio.wav \
  --checkpoint models/best.pth \
  --feature hf
```

## Run the Streamlit demo

Start the interactive demo with:

```bash
poetry run streamlit run app/streamlit_app.py
```

Upload a WAV/MP3/FLAC file, select a checkpoint, and run inference from the browser UI.

## Run tests

```bash
poetry run pytest -q
```

## Notes

- The current implementation focuses on a lightweight and reproducible training pipeline rather than a fully tuned production model.
- The demo app expects a checkpoint at [models/best.pth](models/best.pth) by default, so update the path if you save checkpoints elsewhere.
- The repository currently targets Python 3.11 for the devcontainer and Poetry environment.
