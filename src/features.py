"""Wrappers for extracting embeddings from Hugging Face pretrained audio models."""
from typing import Optional

import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel

from src import config


class HFEmbedder:
    """Load a Hugging Face audio model and extract fixed-size embeddings.

    Example:
        emb = HFEmbedder("facebook/wav2vec2-base-960h")
        vec = emb.extract(torch_tensor_waveform, sr=16000)
    """

    def __init__(self, model_name: str = config.DEFAULT_HF_MODEL, device: Optional[str] = None):
        self.model_name = model_name
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.sampling_rate = int(getattr(self.feature_extractor, "sampling_rate", 16000))
        self.model = AutoModel.from_pretrained(model_name)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def extract(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """Extract an embedding for a single waveform tensor.

        Args:
            waveform: 1-D or 2-D torch Tensor (samples,) or (channels, samples)
            sr: sample rate

        Returns:
            1-D torch.Tensor embedding (float32)
        """
        # ensure numpy float32 mono
        if waveform.dim() == 2:
            wave = waveform.mean(dim=0).cpu().numpy()
        else:
            wave = waveform.cpu().numpy()
        wave = wave.astype(np.float32, copy=False)

        # Resample to the model's required rate (e.g. 16k for wav2vec2)
        if sr != self.sampling_rate:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=self.sampling_rate)
            sr = self.sampling_rate

        inputs = self.feature_extractor(wave, sampling_rate=sr, return_tensors="pt")
        # move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs)
        # try to pool last_hidden_state if available
        if hasattr(out, "last_hidden_state"):
            hs = out.last_hidden_state  # (batch, seq, dim)
            emb = hs.mean(dim=1).squeeze(0)
        elif hasattr(out, "pooler_output"):
            emb = out.pooler_output.squeeze(0)
        else:
            # fallback: convert outputs to tensor and flatten
            emb = torch.tensor(np.array(out[0].cpu())).mean(dim=0)
        return emb.cpu()
