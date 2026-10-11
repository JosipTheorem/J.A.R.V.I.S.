"""Local speech output; standalone usage: python -m audio.speech."""

import argparse
from pathlib import Path
import tarfile
from time import perf_counter
from urllib.request import urlretrieve

import numpy as np
import sounddevice as sd

from config.voice import OUTPUT_DEVICE, PROJECT_ROOT, TTS_LANGUAGE, TTS_MODEL, TTS_MODELS, TTS_THREADS


class VoiceOutput:
    def __init__(self, name=TTS_MODEL, language=TTS_LANGUAGE):
        if name not in TTS_MODELS:
            raise ValueError(f"Unknown TTS model. Choose: {', '.join(TTS_MODELS)}")
        if language not in {"hr", "en"}:
            raise ValueError("TTS language must be hr or en.")
        self.name, self.language = name, language
        self.options = TTS_MODELS[name]
        if self.options["backend"] == "pocket" and language != "en":
            raise ValueError("Pocket supports English here. Use /tts pocket en.")
        self.model = None

    def load(self):
        if self.model is not None:
            return
        print(f"Loading TTS: {self.name} ({self.options['device']})...", flush=True)
        started = perf_counter()
        backend = self.options["backend"]
        if backend in {"omnivoice", "pocket"}:
            import torch

            torch.set_num_threads(TTS_THREADS)
        if backend == "supertonic":
            import sherpa_onnx as sherpa

            folder_name = "sherpa-onnx-supertonic-3-tts-int8-2026-05-11"
            root = PROJECT_ROOT / "models" / "tts"
            folder = root / folder_name
            files = dict(
                duration_predictor="duration_predictor.int8.onnx",
                text_encoder="text_encoder.int8.onnx",
                vector_estimator="vector_estimator.int8.onnx",
                vocoder="vocoder.int8.onnx", tts_json="tts.json",
                unicode_indexer="unicode_indexer.bin", voice_style="voice.bin",
            )
            if not all((folder / filename).is_file() for filename in files.values()):
                root.mkdir(parents=True, exist_ok=True)
                archive = root / f"{folder_name}.tar.bz2"
                print("Downloading Supertonic 3...", flush=True)
                urlretrieve(f"https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/{archive.name}", archive)
                with tarfile.open(archive) as bundle:
                    bundle.extractall(root, filter="data")
                archive.unlink()
            model = sherpa.OfflineTts(sherpa.OfflineTtsConfig(
                model=sherpa.OfflineTtsModelConfig(
                    supertonic=sherpa.OfflineTtsSupertonicModelConfig(
                        **{key: str(folder / filename) for key, filename in files.items()},
                    ), num_threads=TTS_THREADS, provider=self.options["device"],
                ),
            ))
        elif backend == "omnivoice":
            from omnivoice import OmniVoice

            device = self.options["device"]
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable. Install requirements.txt or choose omnivoice-cpu.")
            model = OmniVoice.from_pretrained(
                "k2-fsa/OmniVoice", device_map=device,
                dtype=torch.float16 if device == "cuda" else torch.float32,
            )
        elif backend == "pocket":
            from pocket_tts import TTSModel

            model = TTSModel.load_model(language="english")
            self.voice_state = model.get_state_for_audio_prompt(self.options["voice"])
        else:
            raise ValueError(f"Unknown TTS backend: {backend}")
        self.model = model
        print(f"TTS ready ({perf_counter() - started:.1f}s).", flush=True)

    def synthesize(self, text):
        self.load()
        started = perf_counter()
        backend = self.options["backend"]
        if backend == "supertonic":
            import sherpa_onnx as sherpa

            settings = sherpa.GenerationConfig()
            settings.sid = self.options["voice"]
            settings.num_steps = self.options["steps"]
            settings.extra = {"lang": self.language}
            audio = self.model.generate(text, settings)
            samples, rate = audio.samples, audio.sample_rate
        elif backend == "omnivoice":
            samples = self.model.generate(
                text=text, language=self.language, instruct=self.options["voice"],
                num_step=self.options["steps"],
            )[0]
            rate = self.model.sampling_rate
        else:
            samples = self.model.generate_audio(self.voice_state, text).detach().cpu().numpy()
            rate = self.model.sample_rate
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not samples.size or not np.isfinite(samples).all():
            raise RuntimeError("TTS returned empty or invalid audio.")
        print(f"Speech: {perf_counter() - started:.2f}s generation, {samples.size / rate:.1f}s audio ({self.name}).", flush=True)
        return samples, rate

    def speak(self, text):
        if text.strip():
            samples, rate = self.synthesize(text)
            sd.play(samples, rate, device=OUTPUT_DEVICE)
            try:
                sd.wait()
            finally:
                sd.stop()


if __name__ == "__main__":
    from audio.voice import configure_terminal

    configure_terminal()
    parser = argparse.ArgumentParser(description="Test TTS without the agent or microphone.")
    parser.add_argument("--model", choices=TTS_MODELS, default=TTS_MODEL)
    parser.add_argument("--lang", choices=["hr", "en"], default=TTS_LANGUAGE)
    parser.add_argument("--text", required=True)
    parser.add_argument("--save", type=Path, help="Save a WAV instead of playing it")
    args = parser.parse_args()
    try:
        voice = VoiceOutput(args.model, args.lang)
        if args.save:
            import soundfile as sf

            samples, rate = voice.synthesize(args.text)
            sf.write(args.save, samples, rate)
            print(f"Saved: {args.save}")
        else:
            voice.speak(args.text)
    except KeyboardInterrupt:
        sd.stop()
    except Exception as error:
        print(f"Speech error: {error}")
        raise SystemExit(1)
