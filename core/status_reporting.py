"""
Lightweight status reporting for the orchestrator providing structured CLI feedback.
"""

import sys
import os
from typing import Optional


class ConsoleStatusReporter:
    """
    Default reporter rendering structured events to stdout using consistent
    prefixes. Designed to emulate modern CLI agent output.
    """

    TAG_MAP = {
        "log_sample": "LOG",
        "llm_response": "LLM",
        "llm_missing": "LLM",
        "learning_status": "LEARN",
        "conflict": "CONFLICT",
        "conflict_detail": "CONFLICT",
        "conflict_plan": "PLAN",
        "conflict_failure": "CONFLICT",
        "refinement": "REFINE",
        "refinement_failure": "REFINE",
    "resolution": "RESOLVE",
    "resolution_update": "RESOLVE",
    "error": "ERROR",
    "progress": "PROG",
    }

    COLOR_MAP = {
        "success": "\033[32m",
        "error": "\033[31m",
        "warning": "\033[33m",
    }

    def __init__(self) -> None:
        self.stream = sys.stdout
        color_mode = os.environ.get("AGENT1_COLOR", "auto").lower()
        if color_mode not in {"auto", "always", "never"}:
            color_mode = "auto"

        if color_mode == "never":
            self.use_color = False
        elif color_mode == "always":
            self.use_color = True
        else:
            is_tty = hasattr(self.stream, "isatty") and self.stream.isatty()
            self.use_color = is_tty

    def emit(
        self,
        event: str,
        *,
        line_number: Optional[int],
        message: str,
        detail: Optional[str] = None,
        severity: str = "info",
    ) -> None:
        tag = self.TAG_MAP.get(event, event.upper())
        prefix_parts = [f"[{tag}]"]
        if line_number is not None:
            prefix_parts.append(f"[L{line_number}]")
        prefix = " ".join(prefix_parts)

        self._print_with_prefix(prefix, message, severity)
        if detail:
            self._print_with_prefix(f"{prefix}   ", detail, severity)

    def _print_with_prefix(self, prefix: str, text: str, severity: str) -> None:
        if not text:
            self.stream.write(f"{prefix}\n")
            return
        lines = text.splitlines()
        color_prefix = ""
        color_suffix = ""
        if self.use_color:
            color_prefix = self.COLOR_MAP.get(severity, "")
            if color_prefix:
                color_suffix = "\033[0m"
        for line in lines:
            if color_prefix:
                self.stream.write(f"{color_prefix}{prefix} {line}{color_suffix}\n")
            else:
                self.stream.write(f"{prefix} {line}\n")
