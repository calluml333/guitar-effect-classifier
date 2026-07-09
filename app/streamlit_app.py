"""Streamlit demo for guitar effect classification."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.predict import format_predictions, predict_audio

st.set_page_config(page_title="Guitar Effect Classifier", layout="wide")

st.title("Guitar Effect Classifier")
st.markdown(
    "Upload a short guitar clip and the model will predict the applied effect."
)

uploaded_file = st.file_uploader("Upload WAV audio", type=["wav", "mp3", "flac"])

checkpoint = st.text_input("Checkpoint path", value="models/best.pth")
feature = st.selectbox("Feature type", ["hf", "log-mel", "waveform"], index=0)

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    if st.button("Predict"):
        with st.spinner("Running inference..."):
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            try:
                preds = predict_audio(
                    audio_path=tmp_path,
                    checkpoint_path=checkpoint,
                    feature=feature,
                    hf_model_name="facebook/wav2vec2-base-960h",
                    sr=32000,
                    duration=3.0,
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
