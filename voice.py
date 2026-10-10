"""Local Parakeet dictation, also usable without starting the agent."""

import argparse
from pathlib import Path
import sys
from time import perf_counter

from huggingface_hub import snapshot_download
import numpy as np
import onnx_asr
import onnxruntime as rt
import sounddevice as sd

from config.voice import (
    DEVICE, INPUT_DEVICE, MAX_RECORD_SECONDS, MODEL_DIR, MODEL_NAME,
    SAMPLE_RATE, VAD_DIR,
)


def configure_terminal():
    # Windows redirected streams can default to cp1252, which cannot print ć/č/đ.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


class VoiceInput:
    def __init__(self, device=DEVICE, microphone=INPUT_DEVICE):
        self.device = device
        self.microphone = microphone
        self.model = None

    def load(self):
        """Download missing files, verify the GPU, and keep a warmed model in memory."""
        if self.model is not None:
            return
        started = perf_counter()
        print(f"Loading Parakeet v3 on {self.device}...", flush=True)
        # Download explicitly so an interrupted first download is resumable.
        # Complete local directories never need a network request.
        downloads = (
            (MODEL_DIR, "istupakov/parakeet-tdt-0.6b-v3-onnx", [
                "config.json", "vocab.txt", "encoder-model.onnx",
                "encoder-model.onnx.data", "decoder_joint-model.onnx",
            ]),
            (VAD_DIR, "istupakov/silero-vad-onnx", ["config.json", "silero_vad.onnx"]),
        )
        for folder, repository, filenames in downloads:
            if not all((folder / name).is_file() for name in filenames):
                print(f"Downloading {repository} to {folder}...", flush=True)
                snapshot_download(repository, local_dir=folder, allow_patterns=filenames)
        options = rt.SessionOptions()
        options.intra_op_num_threads = 4
        options.log_severity_level = 3
        if self.device == "cuda":
            rt.preload_dlls()
            providers = [
                ("CUDAExecutionProvider", {
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "HEURISTIC",
                }),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]

        model = onnx_asr.load_model(
            MODEL_NAME, MODEL_DIR, providers=providers, sess_options=options,
            preprocessor_config={"providers": ["CPUExecutionProvider"]},
            resampler_config={"providers": ["CPUExecutionProvider"]},
        )
        # ORT can silently fall back to CPU when CUDA DLLs fail to load.
        # These sessions belong to the pinned onnx-asr Parakeet implementation.
        if self.device == "cuda":
            for session in (model.asr._encoder, model.asr._decoder_joint):
                if "CUDAExecutionProvider" not in session.get_providers():
                    raise RuntimeError("CUDA failed to load. Check CUDA/cuDNN, or set DEVICE = 'cpu' in config/voice.py.")
                session.disable_fallback()

        vad = onnx_asr.load_vad(
            "silero", VAD_DIR, providers=["CPUExecutionProvider"], sess_options=options,
        )
        # Run the encoder/decoder once before the user starts speaking.
        model.recognize(np.zeros(SAMPLE_RATE, dtype=np.float32), sample_rate=SAMPLE_RATE)
        self.model = model.with_vad(
            vad, max_speech_duration_s=25, min_silence_duration_ms=500,
            speech_pad_ms=250, batch_size=1,
        )
        print(f"Parakeet ready on {self.device} ({perf_counter() - started:.1f}s).", flush=True)

    def record(self):
        """Capture audio in RAM until Enter; Ctrl+C cancels and closes the mic."""
        # Capture at the microphone's native rate; onnx-asr resamples to 16 kHz.
        sample_rate = int(sd.query_devices(self.microphone, "input")["default_samplerate"])
        self.load()
        chunks, recording_errors = [], []
        captured = 0
        max_samples = int(MAX_RECORD_SECONDS * sample_rate)

        def callback(data, frames, time_info, status):
            nonlocal captured
            if status:
                recording_errors.append(str(status))
            count = min(frames, max_samples - captured)
            if count > 0:
                chunks.append(data[:count, 0].copy())
                captured += count
            if captured >= max_samples:
                print(f"\n{MAX_RECORD_SECONDS}s limit reached. Press Enter to transcribe.", flush=True)
                raise sd.CallbackStop

        with sd.InputStream(
            device=self.microphone, channels=1, dtype="float32",
            samplerate=sample_rate, callback=callback,
        ):
            input(f"Recording: speak English or Croatian. Press Enter to stop (max {MAX_RECORD_SECONDS}s): ")
        if recording_errors:
            raise RuntimeError(f"Microphone audio was interrupted ({recording_errors[0]}). Please try again.")
        return (np.concatenate(chunks), sample_rate) if chunks else None

    def transcribe(self, audio, sample_rate=SAMPLE_RATE):
        self.load()
        started = perf_counter()
        text = " ".join(
            segment.text.strip()
            for segment in self.model.recognize(audio, sample_rate=sample_rate, channel="mean")
            if segment.text.strip()
        ).strip()
        print(f"Transcription: {perf_counter() - started:.2f}s ({self.device})")
        if not text:
            print("No speech detected; nothing sent.")
        else:
            print(f"Transcript: {text}")
        return text

    def dictate(self):
        recording = self.record()
        return self.transcribe(*recording) if recording else ""


def review_transcript(text):
    """Only a reviewed final transcript may become an agent message."""
    if not text:
        return None
    choice = input("Enter to send, /cancel to discard, or type a corrected message: ").strip()
    return None if choice.lower() == "/cancel" else (choice or text)


def main():
    configure_terminal()
    parser = argparse.ArgumentParser(description="Test local English/Croatian dictation without calling the agent.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--setup", action="store_true", help="Download models and verify inference")
    mode.add_argument("--devices", action="store_true", help="List microphone inputs")
    mode.add_argument("--file", type=Path, help="Transcribe a PCM WAV file")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of configured device")
    parser.add_argument("--mic", type=int, default=INPUT_DEVICE, help="Microphone index from --devices")
    args = parser.parse_args()
    try:
        voice = VoiceInput("cpu" if args.cpu else DEVICE, args.mic)
        if args.devices:
            print(sd.query_devices())
        elif args.setup:
            voice.load()
            print(f"Model files: {MODEL_DIR}")
        elif args.file:
            voice.transcribe(args.file)
        else:
            print("Transcription test only. No messages are sent to the agent.")
            while input("\nEnter to start recording, or /quit to exit: ").strip().lower() not in {"/quit", "quit", "exit"}:
                voice.dictate()
    except (EOFError, KeyboardInterrupt):
        print()
    except Exception as error:
        print(f"Voice error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
