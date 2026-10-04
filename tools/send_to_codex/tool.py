import json
import subprocess

from .config import (
    CODEX_MODEL,
    CODEX_REASONING_EFFORT,
    CODEX_TIMEOUT_SECONDS,
    get_codex_executable,
)
from config.settings import PROJECT_ROOT


def send_to_codex(
    message: str,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """Send a question or coding request to Codex and return its final answer.

    Each call starts a fresh session with no previous conversation history.
    Codex has full filesystem and command access, with no approval prompts.
    Include any relevant context from earlier turns in the message.
    Preserve the user's exact website names, URLs, and constraints; do not
    replace a named website with a generic category. For web research, ask
    Codex to search and return source URLs. Do not claim it searched or used
    a specific website unless the returned answer supports that claim.

    Args:
        message: A self-contained task, including context, required sources,
            and the desired output (e.g. title, ingredients, baking time, URL).
        model: Which Codex model handles the task. Use gpt-6-luna for simple
            lookups and small tasks; consider gpt-6.1-sol for complex coding
            or analysis. Follow the user's requested model when specified.
        reasoning_effort: How much reasoning the model applies. Use low for
            straightforward lookups, medium for tasks with several steps,
            and high/xhigh for difficult debugging or deeper analysis.
            Higher effort generally takes longer; supported levels vary by model.

    Omitted model/effort parameters use this tool's config.py defaults
    (currently gpt-6-luna / low), which are explicitly sent to Codex.
    When reporting settings, distinguish omitted arguments from these
    applied defaults; do not describe an omitted effort as "not configured".
    """
    if not message.strip():
        return "Codex error: message must not be empty."

    selected_model = CODEX_MODEL if model is None else model.strip()
    selected_effort = CODEX_REASONING_EFFORT if reasoning_effort is None else reasoning_effort.strip()
    if not selected_model or not selected_effort:
        return "Codex error: model and reasoning_effort must not be empty."

    try:
        result = subprocess.run(
            [
                get_codex_executable(),
                "exec",
                "--ignore-user-config",
                "--ephemeral",
                "--sandbox", "danger-full-access",
                "--config", 'approval_policy="never"',
                "--model", selected_model,
                "--config", f"model_reasoning_effort={json.dumps(selected_effort)}",
                "--color", "never",
                "-", #read the message from stdin, so it isn't interpreted as CLI options
            ],
            input=message,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=PROJECT_ROOT,
            timeout=CODEX_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"Codex error: request timed out after {CODEX_TIMEOUT_SECONDS} seconds."
    except OSError as error:
        return f"Codex error: {error}"

    if result.returncode != 0:
        details = (result.stderr or result.stdout).strip()
        return f"Codex error (exit {result.returncode}): {details[-2000:]}"

    # codex exec writes progress to stderr and its final answer to stdout.
    return result.stdout.strip() or "Codex error: no answer was returned."
