"""LLM-based variable classification: closed enumeration vs open domain."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, Optional, Set

from agents.enumeration_analysis_agent import EnumerationAnalysisAgent


@dataclass
class EnumVariable:
    """Variable with observed values and LLM classification."""
    variable_name: str
    observed_values: Set[str] = field(default_factory=set)
    occurrence_count: int = 0
    is_closed: bool = False
    llm_reasoning: str = ""

    def to_dict(self) -> Dict:
        return {
            "variable_name": self.variable_name,
            "observed_values": sorted(list(self.observed_values)),
            "occurrence_count": self.occurrence_count,
            "is_closed": self.is_closed,
            "llm_reasoning": self.llm_reasoning,
        }


class EnumerationLibrary:
    """LLM-based variable classification library."""

    MIN_OBSERVATIONS = 10

    def __init__(self, storage_path: Path, *, api_client=None) -> None:
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.variables: Dict[str, EnumVariable] = {}
        self.analysis_agent = EnumerationAnalysisAgent(api_client) if api_client else None

        if self.storage_path.exists():
            self._load()

    def observe_variable(self, variable_name: str, value: str) -> None:
        """Record variable observation and trigger LLM analysis when ready."""
        if variable_name not in self.variables:
            self.variables[variable_name] = EnumVariable(variable_name=variable_name)

        var = self.variables[variable_name]
        var.observed_values.add(value)
        var.occurrence_count += 1

        if (
            var.occurrence_count >= self.MIN_OBSERVATIONS
            and not var.llm_reasoning
        ):
            self._classify_with_llm(variable_name)

    def is_closed_enumeration(self, variable_name: str) -> bool:
        """Check if variable is closed enumeration."""
        var = self.variables.get(variable_name)
        return var.is_closed if var else False

    def get_all_closed_enumerations(self) -> Dict[str, Set[str]]:
        """Get all closed enumerations with their value sets."""
        return {n: v.observed_values for n, v in self.variables.items() if v.is_closed}

    def _classify_with_llm(self, variable_name: str) -> None:
        """Classify variable using LLM agent."""
        if not self.analysis_agent:
            return

        var = self.variables[variable_name]
        analysis = self.analysis_agent.analyze_variable(
            variable_name, var.observed_values, var.occurrence_count
        )

        if analysis:
            var.is_closed = (analysis.classification == "closed_enumeration")
            var.llm_reasoning = analysis.reasoning or analysis.classification

    def _load(self) -> None:
        """Load from disk."""
        with self.storage_path.open("r") as f:
            data = json.load(f)
        for v in data.get("variables", []):
            var = EnumVariable(
                variable_name=v["variable_name"],
                observed_values=set(v.get("observed_values", [])),
                occurrence_count=v.get("occurrence_count", 0),
                is_closed=v.get("is_closed", False),
                llm_reasoning=v.get("llm_reasoning", ""),
            )
            self.variables[var.variable_name] = var

    def save(self) -> None:
        """Save to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with self.storage_path.open("w") as f:
            json.dump({"variables": [v.to_dict() for v in self.variables.values()]}, f)
