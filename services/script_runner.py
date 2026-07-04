from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass

from data_access.paths import PROJECT_ROOT


@dataclass
class ScriptResult:
    command: list[str]
    returncode: int
    output: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    @property
    def rate_limited(self) -> bool:
        text = self.output.lower()
        return "429" in self.output or "rate limited" in text


def python_command(script: str, *args: str) -> list[str]:
    return [sys.executable, script, *args]


def run_command(command: list[str], timeout: int = 300) -> ScriptResult:
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return ScriptResult(command=command, returncode=1, output=str(exc), timed_out=True)
    except OSError as exc:
        return ScriptResult(command=command, returncode=1, output=str(exc))

    output = f"{result.stdout}\n{result.stderr}".strip()
    return ScriptResult(command=command, returncode=result.returncode, output=output)

