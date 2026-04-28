# src/audio_artefact.py
import numpy as np
import librosa

def audio_gan_score(audio_path, return_details=False):
    y, sr = librosa.load(audio_path, sr=None)
    if y is None or len(y) == 0:
        if return_details:
            return 0.0, {"hf_ratio": 0.0, "flatness": 0.0, "hf_n": 0.0, "flat_n": 0.0}
        return 0.0

    # --- HF Ratio ---
    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    hf_band = stft[freqs > 8000]
    lf_band = stft[freqs <= 8000]

    hf_energy = float(hf_band.mean()) if hf_band.size > 0 else 0.0
    lf_energy = float(lf_band.mean()) if lf_band.size > 0 else 0.0
    hf_ratio = hf_energy / (lf_energy + 1e-6)

    # --- Spectral Flatness ---
    flatness = librosa.feature.spectral_flatness(y=y).mean()

    # --- Normalize (heuristic) ---
    hf_n = np.clip(float(hf_ratio) / 0.5, 0, 1)
    flat_n = np.clip(float(flatness) / 0.4, 0, 1)

    Sa = float(np.clip(0.6 * hf_n + 0.4 * flat_n, 0, 1))

    if return_details:
        return Sa, {
            "hf_ratio": float(hf_ratio),
            "flatness": float(flatness),
            "hf_n": float(hf_n),
            "flat_n": float(flat_n),
        }

    return Sa
