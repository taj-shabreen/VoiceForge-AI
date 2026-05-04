"""
audio_processor.py
Handles audio loading, cleaning, merging, and quality analysis.

Key fix: Uses a 3-layer loading strategy so ANY WAV format works:
  1. soundfile  (fastest, handles PCM/float WAV)
  2. pydub      (handles compressed, 32-bit, unusual WAVs via ffmpeg)
  3. librosa    (last resort, suppress deprecated audioread warning)
"""

import os
import warnings
import numpy as np
import soundfile as sf
from pydub import AudioSegment, effects
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
TARGET_SR        = 22050   # XTTS v2 preferred sample rate
MIN_DURATION_SEC = 1.5
MAX_DURATION_SEC = 30.0
SILENCE_TOP_DB   = 30


# ─────────────────────────────────────────────
#  ROBUST AUDIO LOADER  (fixes audioread error)
# ─────────────────────────────────────────────
def _load_audio_robust(path: str, target_sr: int = TARGET_SR):
    """
    Try multiple backends in order.
    Returns (numpy float32 array, sample_rate).
    Raises ValueError only if every backend fails.
    """
    errors = []

    # ── Strategy 1: soundfile (best for standard PCM WAV) ──
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)          # stereo → mono
        if sr != target_sr:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        logger.info(f"Loaded via soundfile: {path}")
        return data, target_sr
    except Exception as e:
        errors.append(f"soundfile: {e}")

    # ── Strategy 2: pydub via ffmpeg (handles unusual WAV, mp3, ogg) ──
    try:
        seg = AudioSegment.from_file(path)
        seg = seg.set_channels(1).set_frame_rate(target_sr).set_sample_width(2)
        samples = np.array(seg.get_array_of_samples(), dtype=np.int16)
        data = samples.astype(np.float32) / 32768.0
        logger.info(f"Loaded via pydub: {path}")
        return data, target_sr
    except Exception as e:
        errors.append(f"pydub: {e}")

    # ── Strategy 3: librosa with audioread warning suppressed ──
    try:
        import librosa
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", FutureWarning)
            data, sr = librosa.load(path, sr=target_sr, mono=True)
        logger.info(f"Loaded via librosa: {path}")
        return data, target_sr
    except Exception as e:
        errors.append(f"librosa: {e}")

    raise ValueError(
        f"Could not load audio file '{path}'.\n"
        f"Tried: {'; '.join(errors)}\n"
        f"Fix: Make sure ffmpeg is installed and the file is a valid audio file."
    )


# ─────────────────────────────────────────────
#  SINGLE FILE CLEANING
# ─────────────────────────────────────────────
def clean_audio(input_path: str, output_path: str = None) -> str:
    """
    Full preprocessing pipeline:
    - Convert to mono, resample to TARGET_SR
    - Normalize amplitude
    - Trim silence
    - Duration guard
    """
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(input_path)), "cleaned.wav")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    audio, sr = _load_audio_robust(input_path, TARGET_SR)

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95

    # Trim silence using librosa (suppress deprecation noise)
    try:
        import librosa
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio, _ = librosa.effects.trim(audio, top_db=SILENCE_TOP_DB)
    except Exception:
        pass  # trimming is optional; skip if it fails

    # Duration guard
    duration = len(audio) / sr
    if duration < MIN_DURATION_SEC:
        raise ValueError(
            f"Audio too short ({duration:.1f}s) after trimming silence. "
            f"Minimum is {MIN_DURATION_SEC}s. Please record a longer sample."
        )
    if duration > MAX_DURATION_SEC:
        audio = audio[:int(MAX_DURATION_SEC * sr)]
        duration = MAX_DURATION_SEC

    sf.write(output_path, audio, sr, subtype="PCM_16")
    logger.info(f"Cleaned audio → {output_path}  ({duration:.1f}s)")
    return output_path


# ─────────────────────────────────────────────
#  MULTI-SAMPLE MERGING
# ─────────────────────────────────────────────
def merge_audio_files(file_paths: list, output_path: str = None) -> str:
    """
    Merge multiple audio files into one speaker sample.
    """
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(file_paths[0])), "merged.wav")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    combined = AudioSegment.empty()
    silence_gap = AudioSegment.silent(duration=300)
    loaded = 0

    for path in file_paths:
        if not os.path.exists(path):
            logger.warning(f"Skipping missing: {path}")
            continue
        try:
            seg = AudioSegment.from_file(path)
            if len(seg) / 1000.0 < MIN_DURATION_SEC:
                logger.warning(f"Skipping short clip: {path}")
                continue
            seg = effects.normalize(seg)
            combined += seg + silence_gap
            loaded += 1
        except Exception as e:
            logger.warning(f"Skipping unreadable {path}: {e}")

    if loaded == 0:
        raise ValueError("No valid audio clips could be loaded for merging.")

    combined = combined.set_frame_rate(TARGET_SR).set_channels(1).set_sample_width(2)
    combined.export(output_path, format="wav")
    logger.info(f"Merged {loaded} clips → {output_path}")
    return output_path


def merge_from_folder(folder_path: str, output_path: str = None) -> str:
    wav_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".wav", ".mp3", ".ogg", ".flac"))
    ])
    if not wav_files:
        raise ValueError(f"No audio files found in '{folder_path}'")
    return merge_audio_files(wav_files, output_path)


# ─────────────────────────────────────────────
#  AUDIO QUALITY ANALYSIS
# ─────────────────────────────────────────────
def analyze_audio(file_path: str) -> dict:
    audio, sr = _load_audio_robust(file_path)
    duration = len(audio) / sr
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    snr = float(20 * np.log10(rms / (1e-9 + np.std(audio - np.mean(audio)))))

    try:
        import librosa
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec_centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
    except Exception:
        spec_centroid = 0.0
        zcr = 0.0

    return {
        "duration_sec": round(duration, 2),
        "sample_rate": sr,
        "rms_energy": round(rms, 4),
        "peak_amplitude": round(peak, 4),
        "snr_db": round(snr, 2),
        "spectral_centroid_hz": round(spec_centroid, 1),
        "zero_crossing_rate": round(zcr, 4),
    }


def get_waveform_data(file_path: str, num_points: int = 300) -> dict:
    audio, sr = _load_audio_robust(file_path)
    indices = np.linspace(0, len(audio) - 1, num_points, dtype=int)
    return {
        "times": (indices / sr).tolist(),
        "samples": audio[indices].tolist(),
    }