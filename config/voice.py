from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "nemo-parakeet-tdt-0.6b-v3"
MODEL_DIR = PROJECT_ROOT / "models" / "parakeet-v3"
VAD_DIR = PROJECT_ROOT / "models" / "silero-vad"

# "cuda" uses the NVIDIA GPU; "cpu" is available for troubleshooting.
DEVICE = "cuda"
# None uses the Windows default microphone. List alternatives: python voice.py --devices
INPUT_DEVICE = None
SAMPLE_RATE = 16000
MAX_RECORD_SECONDS = 120
