"""
Agent responsible for fixing validation issues on freshly generated templates
when no conflicts with existing templates are present.
"""

import re
from typing import List, Optional

from agents.base_agent import BaseAgent
from utils.json_payloads import ProcessedLogLine
from agents.parsing_agent import ParsingAgent
from agents.router_agent import RoutingResult
from libraries.template_library import TemplateRecord
from agents.timestamp_agent import TimestampSpec


class TemplateRefinementAgent(BaseAgent):
    """Uses the LLM to refine a single candidate template so it satisfies validators."""

    def __init__(self, api_client, parsing_agent: ParsingAgent) -> None:
        super().__init__(api_client)
        self.parsing_agent = parsing_agent
        self.last_raw_response: str = ""
        self.last_failure_reason: str = ""

    def refine_template(
        self,
        *,
        candidate_record: TemplateRecord,
        candidate_sample: ProcessedLogLine,
        routing: RoutingResult,
        timestamp_spec: Optional[TimestampSpec],
        issues: List[str],
    ) -> Optional[TemplateRecord]:
        self.last_raw_response = ""
        self.last_failure_reason = ""

        if not self.api_client:
            self.last_failure_reason = "no api client available"
            return None

        if not issues:
            self.last_failure_reason = "no refinement issues provided"
            return None

        prompt = self._build_prompt(
            candidate_record=candidate_record,
            candidate_sample=candidate_sample,
            routing=routing,
            timestamp_spec=timestamp_spec,
            issues=issues,
        )

        system_prompt = (
            "You are a log parsing engineer. Refine the provided regex so it satisfies "
            "validator requirements while preserving the structural vs variable boundary. "
            "Respond with JSON only."
        )
        payload = self._call_llm(system_prompt, prompt, save_raw=True)
        if not payload:
            self.last_failure_reason = "could not parse refinement response"
            return None

        regex = payload.get("regex")
        reasoning = payload.get("reasoning", "")
        if not isinstance(regex, str) or not regex.strip():
            self.last_failure_reason = "refinement response missing regex"
            return None

        new_record = self._build_record_from_regex(
            regex=regex.strip(),
            sample=candidate_sample,
            routing=routing,
            timestamp_spec=timestamp_spec,
            reasoning=reasoning,
            raw_response=response,
        )
        if not new_record:
            self.last_failure_reason = "refined regex did not match sample"
            return None
        return new_record

    def _build_prompt(
        self,
        *,
        candidate_record: TemplateRecord,
        candidate_sample: ProcessedLogLine,
        routing: RoutingResult,
        timestamp_spec: Optional[TimestampSpec],
        issues: List[str],
    ) -> str:
        context = f"device_type={routing.device_type}, vendor={routing.vendor}"
        ts_hint = (
            f"Known timestamp format: {timestamp_spec.format} (regex {timestamp_spec.regex}).\n"
            if timestamp_spec
            else ""
        )
        issues_block = "\n".join(f"- {issue}" for issue in issues)
        var_names = candidate_record.get_variable_names()
        group_names = ", ".join(var_names) if var_names else "none"

        instructions = (
            "Refine the candidate template so it satisfies all validator issues.\n"
            "Rules:\n"
            "  • Preserve the distinction between structural constants and business-data variables.\n"
            "  • Maintain the existing capture group names; do not add or rename groups unless required.\n"
            "  • Use only (?P<name>.*?) for variables and escape constants properly.\n"
            "  • Preserve syntactic markers (colons, pipes, parentheses) unless they are clearly variable.\n"
            "Return JSON:\n"
            "{\n"
            '  "regex": "<refined regex>",\n'
            '  "reasoning": "<brief explanation>"\n'
            "}\n"
            "Respond with JSON only."
        )

        return (
            f"Context: {context}\n"
            f"{ts_hint}"
            "Validator issues:\n"
            f"{issues_block}\n\n"
            "Candidate template:\n"
            f"  regex: {candidate_record.regex}\n"
            f"  capture_groups: {group_names}\n"
            f"  example_transformed: {candidate_sample.transformed}\n"
            f"  example_raw: {candidate_sample.raw}\n\n"
            f"{instructions}"
        )

    def _build_record_from_regex(
        self,
        *,
        regex: str,
        sample: ProcessedLogLine,
        routing: RoutingResult,
        timestamp_spec: Optional[TimestampSpec],
        reasoning: str,
        raw_response: str,
    ) -> Optional[TemplateRecord]:
        outcome = self.parsing_agent.build_outcome_from_regex(
            regex=regex,
            sample=sample,
            routing=routing,
            timestamp_spec=timestamp_spec,
            reasoning=reasoning,
            raw_response=raw_response,
        )
        if not outcome:
            return None
        return outcome.template_record

