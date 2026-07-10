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
- Training and validation loops with checkpoint saving, including per-epoch metrics history and a best-epoch confusion matrix for later analysis
- Inference CLI for single-file prediction, auto-detecting the feature type a checkpoint was trained with
- A standalone evaluation script that generates loss/accuracy curve plots, a confusion matrix heatmap, and a per-file predictions report
- Streamlit demo app for interactive uploads and predictions
- Tests covering preprocessing, dataset loading, DSP effects, training, inference, and evaluation

## Project structure

- [app/streamlit_app.py](app/streamlit_app.py) — interactive Streamlit demo
- [scripts/generate_dataset.py](scripts/generate_dataset.py) — generate labeled synthetic audio data
- [scripts/install_idmt_guitar_dataset.py](scripts/install_idmt_guitar_dataset.py) — download/prepare the IDMT-SMT-Guitar dataset as a clean-source pool
- [scripts/predict.py](scripts/predict.py) — command-line inference entry point
- [scripts/evaluate.py](scripts/evaluate.py) — generates evaluation plots/reports for a trained checkpoint (run manually, not part of training)
- [src/audio_processing.py](src/audio_processing.py) — audio loading and feature extraction helpers
- [src/config.py](src/config.py) — shared constants (sample rate, effect classes, default pretrained model)
- [src/dataset.py](src/dataset.py) — dataset class for manifest-based loading
- [src/effects.py](src/effects.py) — DSP implementations of each guitar effect
- [src/features.py](src/features.py) — Hugging Face audio embedding wrapper
- [src/model.py](src/model.py) — classifier head definition
- [src/predict.py](src/predict.py) — inference logic and checkpoint loading
- [src/train.py](src/train.py) — training loop, checkpoint saving, and metrics history logging
- [src/utils.py](src/utils.py) — small helpers shared across scripts (mono downmixing, duration formatting)
- [data/](data/) — raw/generated audio and manifests (gitignored; regenerate locally)
- [models/](models/) — trained checkpoints and training history (gitignored; regenerate locally)
- [outputs/visualizations/](outputs/visualizations/) — default output location for `scripts/evaluate.py`
- [notebooks/](notebooks/) — reserved for exploratory analysis (currently empty)
- [tests/](tests/) — test suite covering preprocessing, dataset loading, DSP effects, training, inference, and evaluation

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

The default `--feature hf` extracts embeddings from [`MIT/ast-finetuned-audioset-10-10-0.4593`](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) (Audio Spectrogram Transformer, pretrained on AudioSet) — chosen over a speech model since it's pretrained on general, non-speech audio events, which better matches classifying guitar timbre. `--feature log-mel`/`--feature waveform` are also available (see [src/config.py](src/config.py) and `--help` for other options).

A checkpoint is saved to `--out-dir` as `best.pth` whenever validation accuracy improves, alongside `training_history.json` (per-epoch loss/accuracy/precision/recall/F1) and `confusion_matrix.json` (the best epoch's validation confusion matrix) — both consumed by `scripts/evaluate.py` below.

## Run inference

Predict an effect from a single audio file. `--feature`/`--hf-model` are optional — they default to whatever the checkpoint was trained with:

```bash
poetry run python scripts/predict.py \
  --audio path/to/audio.wav \
  --checkpoint models/best.pth
```

## Evaluate the model

Generate evaluation visualizations and a per-file predictions report for a trained checkpoint. This is a manual analysis step — it does not run automatically as part of training:

```bash
poetry run python scripts/evaluate.py \
  --checkpoint models/best.pth \
  --manifest data/manifest.csv
```

This writes to `outputs/visualizations/` (override with `--out-dir`):
- `loss_curve.png` / `accuracy_curve.png` — from the checkpoint's `training_history.json`
- `confusion_matrix.png` — recomputed fresh over `--manifest` (or, if `--manifest` is omitted, read from the checkpoint's saved `confusion_matrix.json`)
- `predictions.csv` — per-file true/predicted label, confidence, and top-k breakdown (only when `--manifest` is given)

It also prints overall accuracy and a handful of correct/incorrect example predictions to the console.

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
- Trained checkpoints (`models/*.pth`) are gitignored, not committed — they're regenerable build output. Run `python -m src.train` to produce one locally.
- The repository currently targets Python 3.11 for the devcontainer and Poetry environment.

## Future improvements

- Report real results here (accuracy/F1 and example predictions) once trained on a larger, non-synthetic-only dataset — the current pipeline has only been validated on tiny toy runs.
- Add an architecture diagram.
- Extend `scripts/generate_dataset.py` to window/segment long source recordings into multiple clips, and/or apply waveform-level augmentation (gain jitter, pitch/time shift, noise) before effects are applied, for more training diversity than effect-parameter randomization alone.
- Compare the AST embedding backbone against BEATs/CLAP now that the feature-extraction path is model-agnostic.
