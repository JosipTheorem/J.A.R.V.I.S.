# J.A.R.V.I.S.

Terminal agent with local English/Croatian transcription and speech.

## Setup

Use Python 3.12. From the project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe agent.py
```

After pulling updates, rerun the pip command in your existing environment.
All Python dependencies are in `requirements.txt`.

The default LLM needs [Ollama](https://ollama.com/) running and its model downloaded:

```powershell
ollama pull qwen3.5:2b-q4_K_M
```

GPU voice mode needs an NVIDIA GPU and a driver supporting CUDA 13.
The Python packages provide the CUDA runtime; no separate CUDA Toolkit is needed.
Speech models download automatically on first use, so internet and disk space
are needed initially. Model files are not included in Git.

## Usage

- Type a message to chat, or `exit` to quit.
- `/voice`: speak, press Enter to stop, then confirm or edit the transcript.
  The agent prints its answer and reads it aloud.
- `/voice-test`: transcribe without sending a request.
- `/tts`: list speech models; switch with `/tts supertonic hr`,
  `/tts omnivoice-gpu hr`, `/tts omnivoice-cpu en`, or `/tts pocket en`.
- Ctrl+C stops speech playback.

## Configuration

- `config/models.py`: choose the LLM. For OpenAI, set `OPENAI_API_KEY` in a root `.env`.
- `config/voice.py`: transcription device, microphones, TTS models, voices, and language.
  Set `DEVICE = "cpu"` for transcription without an NVIDIA GPU.
- `prompts/system.txt`: agent instructions.
- `audio/`: speech recognition and synthesis code.

The Codex tool also needs Codex installed and signed in separately.
