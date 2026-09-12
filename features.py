"""
Fixed-length windowing for the AASIST-L model.

AASIST-L's SincNet-style front end expects a fixed number of raw waveform
samples per forward pass: 64600 samples (~4.04s at 16kHz), per the official
config (config/AASIST-L.conf, model_config.nb_samp). This is NOT an
arbitrary choice we're inventing -- it's a hard architectural constraint of
the pretrained checkpoint, so we reproduce the official repo's padding
scheme exactly (data_utils.py: pad / pad_random) rather than guessing.

Our pipeline delivers ~2.0s windows (32000 samples), which is shorter than
64600, so every window gets tile-padded (repeat the audio, then truncate)
to reach the required length. This is what the official eval script does
for short utterances -- it is not something we're improvising for this
project.

No separate LFCC/spectrogram feature extraction is implemented here: AASIST
consumes the raw waveform directly and computes its own learned front-end
internally (see _aasist_arch.py's `CONV`/SincNet layer). If we later add
RawNet2 or a spectrogram-based baseline for comparison, that's where
spectrogram code would go.
"""
from __future__ import annotations

import numpy as np

NB_SAMP = 64600  # fixed input length required by AASIST-L, from AASIST-L.conf


def pad_to_fixed_length(waveform: np.ndarray, max_len: int = NB_SAMP) -> np.ndarray:
    """
    Reproduce the official AASIST `pad()` behaviour (deterministic, used at
    eval time): truncate if long enough, otherwise tile-repeat and truncate.
    """
    x_len = waveform.shape[0]
    if x_len >= max_len:
        return waveform[:max_len]

    num_repeats = int(max_len / x_len) + 1
    padded = np.tile(waveform, num_repeats)[:max_len]
    return padded
