# J.A.R.V.I.S.

Run the terminal agent with the project's virtual environment:

```powershell
.\.venv\Scripts\python.exe agent.py
```

Type `exit` or `quit`, or press Ctrl+C, to leave. Conversation memory lasts
while the script is running.

## Voice input (English and Croatian)

Parakeet v3 transcribes locally on the NVIDIA GPU. Type `/voice` at `You:`,
speak, and press Enter to stop recording. The transcript appears with its
processing time. Press Enter again to send it, type a corrected message,
or enter `/cancel` to discard it. Typed messages continue to work normally.

Use `/voice-test` to record and see a transcript without sending anything
to the agent. Test English, Croatian, and sentences mixing the two.

The voice model loads on the first voice command and stays loaded until
you exit. Silence is filtered with local Silero VAD. Longer requests are
split into speech segments; recording is capped at two minutes. At the cap,
press Enter to transcribe the captured audio. Audio stays in memory and is
not saved or uploaded. Only a transcript you send enters the agent conversation;
that transcript goes to the configured LLM, including an API if selected.

One-time installation and download (already completed on this machine):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe voice.py --setup
```

This installs the CUDA/cuDNN runtime inside the virtual environment and downloads
the full-precision ONNX export of Parakeet v3 (about 2.55 GB) to
`models/parakeet-v3/`, plus a small VAD model to `models/silero-vad/`.
Completed model folders work offline and are ignored by Git. Interrupted downloads
resume on the next attempt. An NVIDIA driver supporting CUDA 13 is required by
the pinned GPU runtime. GPU use is verified; failed CUDA setup reports an error
instead of silently using the CPU.

To test transcription without starting the agent or Ollama:

```powershell
.\.venv\Scripts\python.exe voice.py
```

Press Enter to record, speak, and press Enter to stop. Repeat as needed, then
type `/quit`. To list microphones, transcribe an existing PCM WAV file, or test CPU:

```powershell
.\.venv\Scripts\python.exe voice.py --devices
.\.venv\Scripts\python.exe voice.py --file C:\path\to\recording.wav
.\.venv\Scripts\python.exe voice.py --cpu
```

Edit `config/voice.py` to change the GPU/CPU setting or the microphone index.
The default microphone follows Windows Sound settings; the standalone test also
accepts `--mic INDEX`. If an input is missing, connect or enable your microphone
and restart the terminal. When using the local LLM, Ollama must be running.

Model sources: [NVIDIA Parakeet v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
(CC BY 4.0), [ONNX export](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx),
[onnx-asr](https://github.com/istupakov/onnx-asr), and
[Silero VAD export](https://huggingface.co/istupakov/silero-vad-onnx).

Run the voice integration checks with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Project layout:

```text
agent.py             Agent creation and terminal chat loop
voice.py             Local microphone recording, transcription, standalone test
requirements.txt     All project dependencies
config/
  models.py          Model selection, including commented alternatives
  settings.py        Project paths, .env loading, and conversation configuration
  voice.py           Speech model, GPU/CPU, microphone, and recording settings
prompts/
  system.txt         Editable system prompt
  __init__.py        Prompt loader
tools/
  get_weather/
    tool.py          Weather tool
    __init__.py      Weather tool export
  send_to_codex/
    config.py        Codex CLI discovery, model, reasoning, and timeout
    tool.py          Send a message to Codex and receive its answer
    __init__.py      Codex tool export
  __init__.py        Tool exports
```

Keep `.env` in the project root. Both `.env` and the prompt are loaded using
paths relative to the project files, so you can also launch `agent.py` using
its absolute path from another working directory. Switch models in
`config/models.py` by commenting out the active return and uncommenting the
alternative. The OpenAI option reads `OPENAI_API_KEY` from the environment
or the root `.env` file.

The `send_to_codex(message, model=None, reasoning_effort=None)` tool uses
`codex exec` and your existing Codex login. Run `codex login` if needed.
Each request is a fresh session with full filesystem and command access,
including file edits, without approval prompts. The working directory is
this project, with a 120-second timeout. It does not send the main
agent's conversation history. User CLI configuration is skipped so configured
MCP servers and other customizations aren't loaded; saved login is still used.

Its default model is `gpt-6-luna` with `low` reasoning. Edit `CODEX_MODEL`,
`CODEX_REASONING_EFFORT`, and `CODEX_TIMEOUT_SECONDS` directly in
`tools/send_to_codex/config.py` if needed. CLI discovery checks PATH
and the Windows VS Code extension; `CODEX_EXECUTABLE` can specify a path
explicitly. Existing model options for the main agent remain in `config/models.py`.

The main agent can override the Codex model and reasoning effort per task:

```python
send_to_codex("Explain this project") #uses the defaults in the tool's config.py
send_to_codex("Fix this bug", model="gpt-6.1-sol", reasoning_effort="high")
```

The selected model must be available to your Codex account and support the
requested reasoning level. Calls currently use `--ephemeral`, so they do not
save sessions for resuming.

To test the tool directly, without calling the main agent's model:

```powershell
.\.venv\Scripts\python.exe -c "from tools import send_to_codex; print(send_to_codex('Return only the word CODEX_OK with no punctuation. Do not use tools.'))"
```
