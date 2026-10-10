"""Check that voice input cannot accidentally send a test or cancelled turn."""

import io
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from voice import VoiceInput, review_transcript


class VoiceTests(unittest.TestCase):
    def test_review_preserves_croatian_and_allows_edit_or_cancel(self):
        text = "Otvori datoteku i pronađi pogrešku."
        with patch("builtins.input", return_value=""):
            self.assertEqual(review_transcript(text), text)
        with patch("builtins.input", return_value="Ispravljen zahtjev"):
            self.assertEqual(review_transcript(text), "Ispravljen zahtjev")
        with patch("builtins.input", return_value="/cancel"):
            self.assertIsNone(review_transcript(text))
        with patch("builtins.input") as read:
            self.assertIsNone(review_transcript(""))
            read.assert_not_called()

    def test_voice_tests_and_cancelled_transcripts_do_not_reach_agent(self):
        backend = Mock()
        backend.invoke.return_value = {"messages": [SimpleNamespace(text="done")]}
        microphone = Mock()
        microphone.dictate.side_effect = ["Test hrvatskog", "Discard this", "Pronađi pogrešku."]
        with (
            patch("langchain.agents.create_agent", return_value=backend),
            patch("config.models.get_model"),
            patch("voice.VoiceInput", return_value=microphone),
            patch("builtins.input", side_effect=["/voice-test", "/voice", "/cancel", "/voice", "", "exit"]),
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            runpy.run_path(str(Path(__file__).resolve().parents[1] / "agent.py"), run_name="__main__")
        backend.invoke.assert_called_once()
        self.assertEqual(
            backend.invoke.call_args.args[0],
            {"messages": [{"role": "user", "content": "Pronađi pogrešku."}]},
        )

    def test_microphone_failure_keeps_typed_chat_available(self):
        backend = Mock()
        backend.invoke.return_value = {"messages": [SimpleNamespace(text="done")]}
        microphone = Mock()
        microphone.dictate.side_effect = RuntimeError("No microphone")
        with (
            patch("langchain.agents.create_agent", return_value=backend),
            patch("config.models.get_model"),
            patch("voice.VoiceInput", return_value=microphone),
            patch("builtins.input", side_effect=["/voice", "Hello", "exit"]),
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            runpy.run_path(str(Path(__file__).resolve().parents[1] / "agent.py"), run_name="__main__")
        backend.invoke.assert_called_once()
        self.assertEqual(backend.invoke.call_args.args[0]["messages"][0]["content"], "Hello")

    def test_cancel_closes_microphone(self):
        voice = VoiceInput()
        voice.load = Mock()
        with (
            patch("voice.sd") as audio,
            patch("builtins.input", side_effect=KeyboardInterrupt),
        ):
            audio.query_devices.return_value = {"default_samplerate": 44100}
            with self.assertRaises(KeyboardInterrupt):
                voice.record()
            audio.InputStream.return_value.__exit__.assert_called_once()

    def test_recording_limit_bounds_buffer(self):
        voice = VoiceInput()
        voice.load = Mock()
        with (
            patch("voice.sd") as audio,
            patch("voice.MAX_RECORD_SECONDS", 1),
            patch("builtins.input", return_value=""),
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            audio.query_devices.return_value = {"default_samplerate": 44100}
            audio.CallbackStop = type("CallbackStop", (Exception,), {})

            def open_stream(**kwargs):
                stream = MagicMock()
                def capture():
                    with self.assertRaises(audio.CallbackStop):
                        kwargs["callback"](np.zeros((88200, 1), dtype=np.float32), 88200, None, False)
                stream.__enter__.side_effect = capture
                return stream

            audio.InputStream.side_effect = open_stream
            samples, rate = voice.record()
        self.assertEqual(rate, 44100)
        self.assertEqual(len(samples), 44100)


if __name__ == "__main__":
    unittest.main()
