"""
Utilities for preprocessing log lines.

This module handles basic log line normalization.
"""

from dataclasses import dataclass
import re


@dataclass
class ProcessedLogLine:
    raw: str
    transformed: str


def preprocess_log_line(line: str) -> ProcessedLogLine:
    """Normalize whitespace in log line."""
    raw_line = line.rstrip()
    normalized_line = re.sub(r" {2,}", " ", raw_line)
    return ProcessedLogLine(raw=raw_line, transformed=normalized_line)
