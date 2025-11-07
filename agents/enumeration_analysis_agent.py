"""
Enumeration Analysis Agent - Uses LLM to classify variables as closed enumerations vs open domains.

This agent analyzes observed variable values and uses LLM reasoning to determine:
- Closed Enumeration: Finite, system-predefined value sets (log levels, event types)
- Open Domain: Unbounded, instance-specific values (IPs, timestamps, user IDs)
"""

from dataclasses import dataclass
import json
import re
from typing import Dict, List, Optional, Set


@dataclass
class EnumerationAnalysis:
    """Result of enumeration analysis by LLM."""
    variable_name: str
    classification: str  # "closed_enumeration" or "open_domain"
    reasoning: str
    evidence: List[str]
    suggested_values: Optional[List[str]] = None  # For closed enums, complete value set


class EnumerationAnalysisAgent:
    """
    LLM-based agent that analyzes variables to classify them as:
    - Closed enumerations (branches/structure)
    - Open domains (fruits/business data)

    Uses semantic understanding rather than heuristic rules.
    """

    def __init__(self, api_client, *, min_observations: int = 10, max_values_to_analyze: int = 30) -> None:
        self.api_client = api_client
        self.min_observations = min_observations
        self.max_values_to_analyze = max_values_to_analyze

    def analyze_variable(
        self,
        variable_name: str,
        observed_values: Set[str],
        occurrence_count: int,
        *,
        context: Optional[str] = None,
    ) -> Optional[EnumerationAnalysis]:
        """Analyze variable using LLM to classify as enumeration or open domain."""
        if not self.api_client or occurrence_count < self.min_observations:
            return None

        # Prepare data for LLM
        value_list = sorted(list(observed_values))[:self.max_values_to_analyze]
        unique_count = len(observed_values)

        # Build prompt
        prompt = self._build_analysis_prompt(
            variable_name,
            value_list,
            unique_count,
            occurrence_count,
            context,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a log parsing expert specializing in semantic analysis. "
                    "Your task is to classify variables as closed enumerations or open domains "
                    "based on their semantic role in log messages. Respond with JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.api_client.chat(messages)
            result = self._parse_response(response, variable_name)
            return result
        except Exception as exc:
            print(f"[EnumerationAnalysisAgent] Failed to analyze '{variable_name}': {exc}")
            return None

    def _build_analysis_prompt(
        self,
        var_name: str,
        values: List[str],
        unique_count: int,
        occurrence_count: int,
        context: Optional[str],
    ) -> str:
        """Build LLM prompt for variable analysis."""
        context_str = f" | Context: {context}" if context else ""
        values_display = ", ".join(f"'{v}'" for v in values[:10])
        if len(values) > 10:
            values_display += f", ... ({unique_count - 10} more)"

        return f"""Classify this log variable:

Variable: {var_name}
Values: {values_display}
Stats: {unique_count} unique / {occurrence_count} total (ratio: {unique_count/occurrence_count:.3f}){context_str}

CLOSED ENUMERATION (structural): Finite, system-defined value sets
  • Examples: log_level={{INFO,WARN,ERROR}}, http_method={{GET,POST}}, wifi_action={{deauth,disassoc}}
  • Criteria: Can enumerate all values (≤20), changing value changes event type, system-defined

OPEN DOMAIN (business data): Unbounded, instance-specific values
  • Examples: ip_address, mac_address, username, timestamp, session_id, device_name
  • Criteria: Cannot enumerate all values, changing value doesn't change event type, externally determined

Decision:
  1. Can enumerate all values? YES→closed, NO→open
  2. Changing value changes event type? YES→closed, NO→open
  3. System-defined or external? System→closed, External→open

Return JSON:
{{
  "classification": "closed_enumeration" or "open_domain",
  "reasoning": "brief explanation",
  "evidence": ["point1", "point2", "point3"],
  "suggested_values": ["val1", "val2", ...]  // only for closed_enumeration
}}

When uncertain, prefer open_domain. Consider semantics, not statistics."""

    def _parse_response(
        self,
        response: str,
        variable_name: str,
    ) -> Optional[EnumerationAnalysis]:
        """Parse LLM response into EnumerationAnalysis."""
        try:
            # Extract JSON
            cleaned = response.strip()
            if cleaned.startswith("{"):
                data = json.loads(cleaned)
            else:
                match = re.search(r"\{.*\}", cleaned, re.DOTALL)
                if match:
                    data = json.loads(match.group())
                else:
                    return None

            # Validate and extract fields
            classification = data.get("classification", "open_domain")
            if classification not in ["closed_enumeration", "open_domain"]:
                classification = "open_domain"

            reasoning = data.get("reasoning", "")
            evidence = data.get("evidence", [])
            suggested_values = data.get("suggested_values")

            return EnumerationAnalysis(
                variable_name=variable_name,
                classification=classification,
                reasoning=reasoning,
                evidence=evidence,
                suggested_values=suggested_values,
            )

        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            print(f"[EnumerationAnalysisAgent] Failed to parse response: {exc}")
            return None
