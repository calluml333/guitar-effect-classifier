import tempfile
from pathlib import Path
import torchaudio
import torch

from src.audio_processing import load_audio, waveform_to_log_mel
from src.dataset import GuitarEffectsDataset


def test_load_audio_and_log_mel():
    # generate a short sine wave and write to a temp file
    sr = 16000
    t = torch.linspace(0, 1.0, int(sr * 1.0))
    wave = 0.1 * torch.sin(2 * 440.0 * 2 * 3.14159 * t)
    tmpdir = tempfile.TemporaryDirectory()
    p = Path(tmpdir.name) / "test.wav"
    import soundfile as sf
    sf.write(str(p), wave.numpy(), sr)
    loaded = load_audio(str(p), sr=sr, duration=1.0)
    assert isinstance(loaded, torch.Tensor)
    mel = waveform_to_log_mel(loaded, sr=sr)
    assert mel.ndim == 2


def test_dataset_manifest_smoke(tmp_path):
    # create a fake manifest with one example
    wav_path = tmp_path / "example.wav"
    sr = 16000
    t = torch.linspace(0, 1.0, int(sr * 1.0))
    wave = 0.1 * torch.sin(2 * 440.0 * 2 * 3.14159 * t)
    import soundfile as sf
    sf.write(str(wav_path), wave.numpy(), sr)
    manifest = tmp_path / "manifest.csv"
    with open(manifest, "w") as f:
        f.write("filename,label,params\n")
        f.write(f"{wav_path.as_posix()},clean,{{}}\n")
    ds = GuitarEffectsDataset(str(manifest), sr=sr, duration=1.0, feature="log-mel")
    x, y = ds[0]
    assert x.ndim == 2
    assert isinstance(y, int)
