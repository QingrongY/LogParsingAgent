"""
Timestamp extraction agent with heuristic and LLM-backed inference.
"""

from dataclasses import dataclass, asdict
import json
import re
from typing import Dict, Iterable, List, Optional


@dataclass
class TimestampSpec:
    regex: str
    format: str
    has_year: bool
    example: Optional[str] = None
    source: str = "heuristic"

    def to_dict(self) -> Dict:
        payload = asdict(self)
        return payload

    @staticmethod
    def from_dict(payload: Dict) -> "TimestampSpec":
        return TimestampSpec(
            regex=payload["regex"],
            format=payload["format"],
            has_year=payload.get("has_year", True),
            example=payload.get("example"),
            source=payload.get("source", "heuristic"),
        )


class TimestampAgent:
    """Determines timestamp parsing strategy for logs."""

    HEURISTICS = [
        TimestampSpec(
            regex=r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})",
            format="%Y-%m-%d %H:%M:%S",
            has_year=True,
            source="heuristic",
        ),
        TimestampSpec(
            regex=r"(?P<ts>[A-Z][a-z]{2}\s{1,2}\d{1,2} \d{2}:\d{2}:\d{2})",
            format="%b %d %H:%M:%S",
            has_year=False,
            source="heuristic",
        ),
        TimestampSpec(
            regex=r"(?P<ts>\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2} (AM|PM))",
            format="%m/%d/%Y %I:%M:%S %p",
            has_year=True,
            source="heuristic",
        ),
        TimestampSpec(
            regex=r"(?P<ts>\d{2}:\d{2}:\d{2})",
            format="%H:%M:%S",
            has_year=False,
            source="heuristic",
        ),
    ]

    def __init__(self, api_client=None) -> None:
        self.api_client = api_client

    def infer_timestamp_spec(
        self,
        log_samples: Iterable[str],
        *,
        existing_spec: Optional[TimestampSpec] = None,
        allow_llm: bool = True,
    ) -> Optional[TimestampSpec]:
        """
        Determine timestamp extraction instructions from log samples.
        """
        if existing_spec:
            return existing_spec

        samples = [line for line in log_samples if line]
        if not samples:
            return None

        heuristic = self._apply_heuristics(samples)
        if heuristic:
            return heuristic

        if allow_llm and self.api_client:
            return self._llm_infer(samples)

        return None

    def validate(self, line: str, spec: TimestampSpec) -> bool:
        return bool(re.search(spec.regex, line))

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _apply_heuristics(self, samples: List[str]) -> Optional[TimestampSpec]:
        for spec in self.HEURISTICS:
            pattern = re.compile(spec.regex)
            matches = [pattern.search(line) for line in samples]
            hits = [m.group("ts") for m in matches if m]
            if len(hits) >= max(1, len(samples) // 2):
                candidate = TimestampSpec(
                    regex=spec.regex,
                    format=spec.format,
                    has_year=spec.has_year,
                    example=hits[0],
                    source="heuristic",
                )
                return candidate
        return None

    def _llm_infer(self, samples: List[str]) -> Optional[TimestampSpec]:
        prompt = (
            "You are a log timestamp extraction agent. Given sample log lines, "
            "return JSON describing how to extract the timestamp. "
            "Use fields: regex (must include named group 'ts'), "
            "format (Python strptime), has_year (true/false), example. "
            "Respond with JSON only."
        )
        blob = "\n".join(samples[:12])
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": blob},
        ]
        try:
            response = self.api_client.chat(messages)
        except Exception:
            return None
        spec_dict = self._extract_json(response)
        if not spec_dict:
            return None
        try:
            spec = TimestampSpec.from_dict(spec_dict)
            spec.source = "llm"
            return spec
        except KeyError:
            return None

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict]:
        try:
            if text.strip().startswith("{"):
                return json.loads(text.strip())
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except json.JSONDecodeError:
            return None
        return None
