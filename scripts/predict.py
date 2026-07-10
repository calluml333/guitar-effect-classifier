"""Inference CLI for guitar effect classification."""
import argparse
import sys
from pathlib import Path

# Ensure the repository root is on sys.path so `from src...` imports work
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import config  # noqa: E402
from src.predict import format_predictions, predict_audio  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Predict guitar effect from audio file")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved model checkpoint")
    parser.add_argument("--feature", type=str, default="hf", choices=["hf", "log-mel", "waveform"])
    parser.add_argument("--hf-model", type=str, default=config.DEFAULT_HF_MODEL)
    parser.add_argument("--sr", type=int, default=config.SAMPLE_RATE)
    parser.add_argument("--duration", type=float, default=config.AUDIO_DURATION)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    predictions = predict_audio(
        audio_path=args.audio,
        checkpoint_path=args.checkpoint,
        feature=args.feature,
        hf_model_name=args.hf_model,
        sr=args.sr,
        duration=args.duration,
        topk=args.topk,
        use_cuda=args.use_cuda,
    )

    print("Predicted effect:")
    print(format_predictions(predictions))


if __name__ == "__main__":
    main()
