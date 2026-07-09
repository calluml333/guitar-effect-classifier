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

## Install the IDMT-SMT-Guitar dataset locally

If you want to use the IDMT-SMT-Guitar collection as your clean-source pool, you can install it locally without committing any downloaded files to Git.

Preview the workflow first:

```bash
poetry run python scripts/install_idmt_guitar_dataset.py --dry-run
```

Then run the full setup:

```bash
poetry run python scripts/install_idmt_guitar_dataset.py \
  --dataset-ids 2,3,4 \
  --sample-rate 44100 \
  --bit-depth 16
```

This will:
- download the IDMT-SMT-Guitar Dataset into [data/downloads](data/downloads)
- extract the archive locally
- select clean WAV files matching the requested datasets
- copy them into [data/raw/idmt_smt_guitar](data/raw/idmt_smt_guitar)
- write a local manifest at [data/raw/idmt_smt_guitar/manifest.csv](data/raw/idmt_smt_guitar/manifest.csv)

Manual manifest (demo)
----------------------

For quick demos you can prepare a manual CSV manifest and pass it to the installer instead of running discovery. A template is available at `data/idmt_manual_manifest.csv` — edit and uncomment rows to point to either:

- `source_path`: an absolute or repo-relative path to a WAV file
- OR `archive` + `relative_path`: the archive filename and path inside the archive

Run using:

```bash
poetry run python scripts/install_idmt_guitar_dataset.py --manifest data/idmt_manual_manifest.csv
```

This is useful when you have a curated set of files to include for a demo.

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
