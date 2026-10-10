import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF

SR = 32000
N_FFT = 4096
HOP = 502
N_MELS = 256
TOP_DB = 80.0


def load_audio(path, length=SR):
    wave, sr = torchaudio.load(str(path))
    wave = wave.mean(0)
    if sr != SR:
        wave = AF.resample(wave, sr, SR)
    if length is None:
        return wave
    return F.pad(wave, (0, max(0, length - wave.numel())))[:length]


class LogMel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0
        )

    def forward(self, wave):
        db = 10 * torch.log10(self.mel(wave).clamp_min(1e-10))
        db = db - db.amax(dim=(-2, -1), keepdim=True)
        return db.clamp_min(-TOP_DB)


def to_unit(db):
    return db / (TOP_DB / 2) + 1


def to_db(x):
    return (x - 1) * (TOP_DB / 2)