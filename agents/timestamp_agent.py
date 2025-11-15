"""
Timestamp extraction agent using LLM.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional

from agents.base_agent import BaseAgent


@dataclass
class TimestampSpec:
    regex: str
    format: str
    has_year: bool
    example: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(payload: Dict) -> "TimestampSpec":
        return TimestampSpec(
            regex=payload["regex"],
            format=payload["format"],
            has_year=payload.get("has_year", True),
            example=payload.get("example"),
        )


class TimestampAgent(BaseAgent):
    """Determines timestamp parsing strategy for logs using LLM."""

    def infer_timestamp_spec(self, log_samples: Iterable[str]) -> Optional[TimestampSpec]:
        """Determine timestamp extraction instructions from log samples using LLM."""
        samples = [line for line in log_samples if line]
        if not samples:
            return None

        system_prompt = (
            "You are a log timestamp extraction agent. Given sample log lines, "
            "return JSON describing how to extract the timestamp. "
            "Use fields: regex (must include named group 'ts'), "
            "format (Python strptime), has_year (true/false), example. "
            "Respond with JSON only."
        )
        user_prompt = "\n".join(samples[:12])

        spec_dict = self._call_llm(system_prompt, user_prompt)
        if spec_dict:
            try:
                return TimestampSpec.from_dict(spec_dict)
            except KeyError:
                pass

        return None
