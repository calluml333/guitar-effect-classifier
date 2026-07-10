"""Streamlit demo for guitar effect classification."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.predict import (  # noqa: E402, E501
    format_predictions,
    load_checkpoint,
    predict_audio,
)

st.set_page_config(page_title="Guitar Effect Classifier", layout="wide")

st.title("Guitar Effect Classifier")
st.markdown(
    "Upload a short guitar clip and the model will predict the applied effect."
)

uploaded_file = st.file_uploader("Upload WAV audio", type=["wav", "mp3", "flac"])  # noqa: E501

checkpoint = st.text_input("Checkpoint path", value="models/best.pth")

FEATURE_OPTIONS = ["hf", "log-mel", "waveform"]
detected_feature = None
if Path(checkpoint).exists():
    try:
        detected_feature = load_checkpoint(checkpoint).get("feature")
    except Exception:
        detected_feature = None

default_index = FEATURE_OPTIONS.index(detected_feature) if detected_feature in FEATURE_OPTIONS else 0  # noqa: E501
feature = st.selectbox("Feature type", FEATURE_OPTIONS, index=default_index)
if detected_feature:
    st.caption(f"Auto-detected from checkpoint: '{detected_feature}'")
elif Path(checkpoint).exists():
    st.caption("Checkpoint doesn't record a feature type (older checkpoint) — select manually.")  # noqa: E501

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    if st.button("Predict"):
        with st.spinner("Running inference..."):
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:  # noqa: E501
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            try:
                preds = predict_audio(
                    audio_path=tmp_path,
                    checkpoint_path=checkpoint,
                    feature=feature,
                    topk=5,
                    use_cuda=False,
                )
                st.subheader("Predictions")
                st.write(format_predictions(preds))
            except Exception as exc:
                st.error(f"Inference failed: {exc}")
            finally:
                import os

                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
