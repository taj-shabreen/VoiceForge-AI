"""
tts_engine.py
XTTS v2 model wrapper with caching and generation controls.
"""

import os
import time
import logging
import streamlit as st

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Polish": "pl",
    "Turkish": "tr",
    "Russian": "ru",
    "Dutch": "nl",
    "Czech": "cs",
    "Arabic": "ar",
    "Chinese (Simplified)": "zh-cn",
    "Japanese": "ja",
    "Korean": "ko",
    "Hindi": "hi",
}


@st.cache_resource(show_spinner=False)
def load_tts_model():
    """
    Load XTTS v2 once and cache it for the session.
    Returns the TTS object or raises on failure.
    """
    from TTS.api import TTS  # lazy import so Streamlit starts fast

    logger.info("Loading XTTS v2 model…")
    model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
    logger.info("XTTS v2 loaded successfully.")
    return model


def generate_speech(
    text: str,
    speaker_wav: str,
    language_code: str = "en",
    output_path: str = "outputs/output.wav",
    speed: float = 1.0,
) -> dict:
    """
    Generate cloned speech.

    Returns a result dict:
        {
            "success": bool,
            "output_path": str,
            "duration_sec": float,
            "generation_time_sec": float,
            "error": str | None
        }
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not text.strip():
        return {"success": False, "error": "Text cannot be empty."}

    if not os.path.exists(speaker_wav):
        return {"success": False, "error": f"Speaker file not found: {speaker_wav}"}

    try:
        tts = load_tts_model()

        start = time.time()
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language=language_code,
            file_path=output_path,
            speed=speed,
        )
        elapsed = round(time.time() - start, 2)

        # Get output duration
        import librosa
        audio, sr = librosa.load(output_path, sr=None)
        duration = round(len(audio) / sr, 2)

        return {
            "success": True,
            "output_path": output_path,
            "duration_sec": duration,
            "generation_time_sec": elapsed,
            "error": None,
        }

    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        return {"success": False, "error": str(e)}