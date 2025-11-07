"""
Utilities for detecting and extracting JSON-like payloads from log lines.

The extractor replaces inline JSON (or JSON-adjacent) blocks with stable
placeholders and provides lightweight structural metadata so that the main
parsing agent can operate on simplified text while still preserving rich
payload details for post-processing.
"""

from dataclasses import dataclass, field
import json
import re
from typing import Dict, List, Optional, Set, Tuple

PLACEHOLDER_TEMPLATE = "<json_payload_{index}>"
MAX_KEY_PATHS = 500

_UNQUOTED_KEY_RE = re.compile(r"(?P<prefix>[{,]\s*)(?P<key>[A-Za-z0-9_\-.$]+)\s*(?=:\s*)")
_SINGLE_QUOTED_VALUE_RE = re.compile(r"'([^'\\]*(?:\\.[^'\\]*)*)'")
_OBJECT_ID_RE = re.compile(r"ObjectId\('([^']*)'\)")
_NUMBER_LONG_RE = re.compile(r"NumberLong\((?:'([^']*)'|([0-9]+))\)")
_SIMPLE_KEY_RE = re.compile(r'(["\']?)([A-Za-z0-9_\-.$]+)\1\s*:')
_OWNER_KEY_RE = re.compile(r'(["\']?)([A-Za-z0-9_\-.$]+)\1\s*$')


@dataclass
class JsonPayload:
    placeholder: str
    raw_text: str
    owner: Optional[str] = None
    normalized_text: Optional[str] = None
    key_paths: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "placeholder": self.placeholder,
            "owner": self.owner,
            "keys": self.key_paths,
            "raw": self.raw_text,
            "normalized": self.normalized_text,
        }


@dataclass
class ProcessedLogLine:
    raw: str
    transformed: str
    payloads: List[JsonPayload] = field(default_factory=list)


def preprocess_log_line(line: str) -> ProcessedLogLine:
    """Replace inline JSON-like payloads with placeholders and capture metadata."""
    raw_line = line.rstrip()
    normalized_line = re.sub(r" {2,}", " ", raw_line)
    segments = _find_json_segments(normalized_line)
    if not segments:
        return ProcessedLogLine(raw=raw_line, transformed=normalized_line, payloads=[])

    payloads: List[JsonPayload] = []
    rebuilt: List[str] = []
    cursor = 0

    for index, (start, end, owner) in enumerate(segments, start=1):
        if start < cursor:
            # Overlapping segment; skip to avoid corruption.
            continue
        placeholder = PLACEHOLDER_TEMPLATE.format(index=index)
        payload_text = normalized_line[start:end]
        normalized, key_paths = _normalize_json_like(payload_text, owner)
        payloads.append(
            JsonPayload(
                placeholder=placeholder,
                raw_text=payload_text,
                owner=owner,
                normalized_text=normalized,
                key_paths=key_paths,
            )
        )
        rebuilt.append(normalized_line[cursor:start])
        rebuilt.append(placeholder)
        cursor = end

    rebuilt.append(normalized_line[cursor:])
    transformed = "".join(rebuilt)
    return ProcessedLogLine(raw=raw_line, transformed=transformed, payloads=payloads)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _find_json_segments(line: str) -> List[Tuple[int, int, Optional[str]]]:
    """
    Identify JSON-like segments by tracking balanced braces/brackets preceded by a colon.
    Returns a list of tuples: (start_index, end_index_exclusive, owner_key).
    """
    segments: List[Tuple[int, int, Optional[str]]] = []
    length = len(line)
    i = 0
    in_string = False
    string_char = ""

    while i < length:
        char = line[i]
        if in_string:
            if char == string_char and (i == 0 or line[i - 1] != "\\"):
                in_string = False
                string_char = ""
            i += 1
            continue

        if char in ("'", '"'):
            in_string = True
            string_char = char
            i += 1
            continue

        if char in "{[":
            start_idx = i
            colon_idx = _find_owning_colon(line, i)
            if colon_idx is None:
                i += 1
                continue
            segment, end_idx = _capture_balanced_segment(line, i)
            if segment is None:
                i += 1
                continue
            if not _segment_is_probable_json(segment):
                i = start_idx + 1
                continue
            owner = _extract_owner_key(line, colon_idx)
            segments.append((i, end_idx + 1, owner))
            i = end_idx + 1
            continue

        i += 1

    return segments


def _find_owning_colon(line: str, brace_index: int) -> Optional[int]:
    idx = brace_index - 1
    while idx >= 0 and line[idx].isspace():
        idx -= 1
    if idx >= 0 and line[idx] == ":":
        return idx
    return None


def _capture_balanced_segment(text: str, start: int) -> Tuple[Optional[str], Optional[int]]:
    stack: List[str] = []
    in_string = False
    string_char = ""
    i = start
    length = len(text)

    while i < length:
        char = text[i]
        if in_string:
            if char == string_char and text[i - 1] != "\\":
                in_string = False
                string_char = ""
            i += 1
            continue

        if char in ("'", '"'):
            in_string = True
            string_char = char
            i += 1
            continue

        if char in "{[":
            stack.append("}" if char == "{" else "]")
            i += 1
            continue

        if char in "}]" and stack:
            expected = stack.pop()
            if char != expected:
                return None, None
            i += 1
            if not stack:
                return text[start:i], i - 1
            continue

        i += 1

    return None, None


def _extract_owner_key(line: str, colon_idx: int) -> Optional[str]:
    prefix = line[:colon_idx]
    match = _OWNER_KEY_RE.search(prefix)
    if match:
        return match.group(2)
    return None


def _normalize_json_like(raw: str, root: Optional[str]) -> Tuple[Optional[str], List[str]]:
    """
    Attempt to coerce Mongo-style extended JSON into valid JSON so that it can
    be parsed. Returns the normalized JSON string (or None) and a list of key
    paths (prefixed by the owning key when available).
    """
    candidate = raw.strip()
    if not candidate:
        return None, []

    candidate = _OBJECT_ID_RE.sub(lambda m: f'"{m.group(1)}"', candidate)
    candidate = _NUMBER_LONG_RE.sub(lambda m: m.group(1) or m.group(2) or "", candidate)
    candidate = _SINGLE_QUOTED_VALUE_RE.sub(
        lambda m: '"' + m.group(1).replace('"', '\\"') + '"', candidate
    )
    candidate = _UNQUOTED_KEY_RE.sub(
        lambda m: f'{m.group("prefix")}"{m.group("key")}":', candidate
    )
    candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None, _fallback_key_paths(raw, root)

    paths = _collect_key_paths(data)
    if root:
        paths = {f"{root}.{path}" if path else root for path in paths}
    result = sorted(paths)[:MAX_KEY_PATHS]
    return candidate, result


def _collect_key_paths(node, prefix: str = "") -> Set[str]:
    paths: Set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            key_str = str(key)
            new_prefix = f"{prefix}.{key_str}" if prefix else key_str
            paths.add(new_prefix)
            paths.update(_collect_key_paths(value, new_prefix))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            new_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            paths.add(new_prefix)
            paths.update(_collect_key_paths(value, new_prefix))
    return paths


def _fallback_key_paths(raw: str, root: Optional[str]) -> List[str]:
    keys = {_sanitize_key(match.group(2)) for match in _SIMPLE_KEY_RE.finditer(raw)}
    keys.discard("")
    limited = sorted(keys)[:MAX_KEY_PATHS]
    if root:
        return [f"{root}.{key}" for key in limited]
    return limited


def _sanitize_key(key: Optional[str]) -> str:
    if not key:
        return ""
    return key.strip().strip('"').strip("'")


def _segment_is_probable_json(segment: str) -> bool:
    """Heuristic to avoid treating arbitrary bracketed content as JSON."""
    stripped = segment.strip()
    if not stripped:
        return False
    first = stripped[0]
    if first == "{":
        if stripped in ("{}", "{ }"):
            return False
        return ":" in stripped or "," in stripped
    if first == "[":
        if "{" in stripped and ":" in stripped:
            return True
        return False
    return False
