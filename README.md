# J.A.R.V.I.S.

Run the terminal agent with the project's virtual environment:

```powershell
.\.venv\Scripts\python.exe agent.py
```

Type `exit` or `quit`, or press Ctrl+C, to leave. Conversation memory lasts
while the script is running.

Project layout:

```text
agent.py             Agent creation and terminal chat loop
config/
  models.py          Model selection, including commented alternatives
  settings.py        Project paths, .env loading, and conversation configuration
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
