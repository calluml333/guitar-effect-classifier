"""Inference CLI for guitar effect classification."""
import argparse

from src.predict import format_predictions, predict_audio


def main():
    parser = argparse.ArgumentParser(description="Predict guitar effect from audio file")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved model checkpoint")
    parser.add_argument("--feature", type=str, default="hf", choices=["hf", "log-mel", "waveform"])
    parser.add_argument("--hf-model", type=str, default="facebook/wav2vec2-base-960h")
    parser.add_argument("--sr", type=int, default=32000)
    parser.add_argument("--duration", type=float, default=3.0)
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
