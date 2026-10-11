from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "nemo-parakeet-tdt-0.6b-v3"
MODEL_DIR = PROJECT_ROOT / "models" / "parakeet-v3"
VAD_DIR = PROJECT_ROOT / "models" / "silero-vad"

# "cuda" uses the NVIDIA GPU; "cpu" is available for troubleshooting.
DEVICE = "cuda"
# None uses the Windows default microphone. List alternatives: python -m audio.voice --devices
INPUT_DEVICE = None
SAMPLE_RATE = 16000
MAX_RECORD_SECONDS = 120

# Speech output: used only for replies to /voice. Change these and restart,
# or switch in the terminal with /tts MODEL [hr|en].
TTS_MODEL = "omnivoice-cpu"
TTS_LANGUAGE = "en"
OUTPUT_DEVICE = None  # Windows default speakers/headphones
TTS_THREADS = 4
TTS_MODELS = {
    "supertonic": {"backend": "supertonic", "device": "cpu", "voice": 0, "steps": 8},
    "omnivoice-gpu": {"backend": "omnivoice", "device": "cuda", "voice": "male", "steps": 32},
    "omnivoice-cpu": {"backend": "omnivoice", "device": "cpu", "voice": "male", "steps": 32},
    "pocket": {"backend": "pocket", "device": "cpu", "voice": "alba"},  # English only
}
# Supertonic speaker IDs: 0-4 = M1-M5, 5-9 = F1-F5.
# OmniVoice "voice" describes the voice; it is not a fixed speaker identity.
# Fewer "steps" generate faster, potentially at lower quality.
