"""
Strategy 1: Data Augmentation.

Two techniques, applied on-the-fly during preprocessing (not pre-computed
to disk, to save storage on a Kaggle session):
  1. Speed Perturbation — resample the waveform to 0.9x-1.1x speed.
  2. SpecAugment — mask random frequency bands and time steps on the
     log-mel spectrogram AFTER feature extraction.

Lingala gets a higher augmentation probability (LINGALA_AUGMENT_BOOST in
config.py) since it's the weakest-baseline language and benefits most from
synthetic data diversity given its thinner data volume.
"""

import random
import numpy as np
import librosa

import config


def maybe_speed_perturb(audio_array: np.ndarray, sr: int, lang: str) -> np.ndarray:
    prob = config.SPEED_PERTURB_PROB
    if lang == "lin_asr":
        prob = min(1.0, prob * config.LINGALA_AUGMENT_BOOST)

    if not config.AUGMENT_ENABLED or random.random() > prob:
        return audio_array

    low, high = config.SPEED_PERTURB_RANGE
    rate = random.uniform(low, high)
    try:
        return librosa.effects.time_stretch(audio_array, rate=rate)
    except Exception:
        # time_stretch can fail on very short clips — fall back to original
        return audio_array


def spec_augment(mel_spectrogram: np.ndarray, lang: str) -> np.ndarray:
    """
    Applies frequency and time masking directly to a log-mel spectrogram.
    Expects shape (n_mels, time_steps) as produced by WhisperFeatureExtractor.
    """
    prob = config.SPECAUGMENT_PROB
    if lang == "lin_asr":
        prob = min(1.0, prob * config.LINGALA_AUGMENT_BOOST)

    if not config.AUGMENT_ENABLED or random.random() > prob:
        return mel_spectrogram

    spec = mel_spectrogram.copy()
    n_mels, n_steps = spec.shape

    for _ in range(config.SPECAUGMENT_NUM_MASKS):
        # Frequency mask
        f_width = random.randint(0, config.SPECAUGMENT_FREQ_MASK_PARAM)
        if f_width > 0 and f_width < n_mels:
            f_start = random.randint(0, n_mels - f_width)
            spec[f_start:f_start + f_width, :] = 0.0

        # Time mask
        t_width = random.randint(0, config.SPECAUGMENT_TIME_MASK_PARAM)
        if t_width > 0 and t_width < n_steps:
            t_start = random.randint(0, n_steps - t_width)
            spec[:, t_start:t_start + t_width] = 0.0

    return spec
