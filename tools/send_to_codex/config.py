from pathlib import Path
from shutil import which

from config.settings import PROJECT_ROOT


# This is separate from the main agent's model in config/models.py.
CODEX_MODEL = "gpt-6-luna"
CODEX_REASONING_EFFORT = "low"
CODEX_TIMEOUT_SECONDS = 120
# Optional: full path to codex.exe if it isn't found automatically.
CODEX_EXECUTABLE = None #for example: "C:/path/to/codex.exe"


def get_codex_executable() -> str:
    """Find the CLI without depending on the terminal's working directory."""
    configured = CODEX_EXECUTABLE
    if configured:
        executable = Path(configured).expanduser()
        if not executable.is_absolute():
            executable = PROJECT_ROOT / executable
        if not executable.is_file():
            raise FileNotFoundError("CODEX_EXECUTABLE must point to an existing Codex executable.")
        return str(executable)

    executable = which("codex")
    if executable:
        return executable

    # The Windows VS Code extension bundles a CLI even when it isn't on PATH.
    bundled = list((Path.home() / ".vscode" / "extensions").glob(
        "openai.chatgpt-*/bin/windows-x86_64/codex.exe"
    ))
    if bundled:
        return str(max(bundled, key=lambda path: path.stat().st_mtime))

    raise FileNotFoundError(
        "Codex CLI was not found. Install it or set CODEX_EXECUTABLE in tools/send_to_codex/config.py."
    )
