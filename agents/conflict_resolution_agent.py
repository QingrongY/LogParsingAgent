"""
Conflict resolution agent that can either adjust the candidate regex or update
conflicting templates when a newly learned pattern collides with existing ones.
"""

from dataclasses import dataclass, field
import re
from typing import Iterable, List, Optional

from agents.base_agent import BaseAgent
from utils.json_payloads import ProcessedLogLine
from agents.parsing_agent import ParsingAgent
from agents.router_agent import RoutingResult
from libraries.template_library import TemplateRecord
from agents.timestamp_agent import TimestampSpec


@dataclass
class ResolutionPlan:
    decision: str  # replace_conflicting | refine_candidate
    new_template_regex: str
    new_template_notes: str = ""
    replaced_template_ids: List[str] = field(default_factory=list)
    reasoning: str = ""


class ConflictResolutionAgent(BaseAgent):
    """Uses the LLM to propose conflict resolution plans."""

    def __init__(self, api_client, parsing_agent: ParsingAgent) -> None:
        super().__init__(api_client)
        self.parsing_agent = parsing_agent
        self.last_raw_response: str = ""
        self.last_failure_reason: str = ""

    def resolve_conflict(
        self,
        *,
        candidate_record: TemplateRecord,
        candidate_sample: ProcessedLogLine,
        routing: RoutingResult,
        timestamp_spec: Optional[TimestampSpec],
        conflicting_records: Iterable[TemplateRecord],
        conflicting_samples: Iterable[ProcessedLogLine],
        issues: Optional[List[str]] = None,
    ) -> Optional[ResolutionPlan]:
        self.last_raw_response = ""
        self.last_failure_reason = ""

        if not self.api_client:
            self.last_failure_reason = "no api client available"
            return None

        records = list(conflicting_records)
        samples = list(conflicting_samples)
        if not records or not samples:
            self.last_failure_reason = "missing conflicting exemplars"
            return None

        prompt = self._build_prompt(
            candidate_record=candidate_record,
            candidate_sample=candidate_sample,
            routing=routing,
            timestamp_spec=timestamp_spec,
            conflicting_records=records,
            conflicting_samples=samples,
            issues=issues,
        )

        system_prompt = (
            "You are a log parsing architect. Resolve conflicts while preserving variable semantics. "
            "Respond with JSON only."
        )
        payload = self._call_llm(system_prompt, prompt, save_raw=True)
        if not payload:
            self.last_failure_reason = "could not parse LLM response"
            return None

        plan = self._parse_plan(payload)
        if not plan:
            self.last_failure_reason = "invalid conflict resolution plan"
            return None
        return plan

    def build_record_from_regex(
        self,
        *,
        regex: str,
        sample: ProcessedLogLine,
        routing: RoutingResult,
        timestamp_spec: Optional[TimestampSpec],
        reasoning: str = "",
        raw_response: str = "",
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

    def _build_prompt(
        self,
        *,
        candidate_record: TemplateRecord,
        candidate_sample: ProcessedLogLine,
        routing: RoutingResult,
        timestamp_spec: Optional[TimestampSpec],
        conflicting_records: List[TemplateRecord],
        conflicting_samples: List[ProcessedLogLine],
        issues: Optional[List[str]] = None,
    ) -> str:
        context = f"device_type={routing.device_type}, vendor={routing.vendor}"
        ts_hint = (
            f"Known timestamp format: {timestamp_spec.format} (regex {timestamp_spec.regex}).\n"
            if timestamp_spec
            else ""
        )
        var_names = candidate_record.get_variable_names()
        group_names = ", ".join(var_names) if var_names else "none"

        conflicts_blob = "\n".join(
            [
                (
                    f"- template_id: {record.template_id}\n"
                    f"  regex: {record.regex}\n"
                    f"  capture_groups: {', '.join(record.get_variable_names())}\n"
                    f"  example_transformed: {sample.transformed}\n"
                    f"  example_raw: {sample.raw}"
                )
                for record, sample in zip(conflicting_records, conflicting_samples)
            ]
        )

        if not conflicts_blob:
            conflicts_blob = "  (none)"

        issues_section = ""
        if issues:
            issues_section = "Issues to address:\n" + "\n".join(f"  • {msg}" for msg in issues) + "\n\n"

        instructions = (
            "Analyze the conflict and choose one decision:\n\n"
            "1. replace_conflicting:\n"
            "   Use when the candidate template correctly identifies business variables that conflicting templates incorrectly hardcoded.\n"
            "   Example: Conflicting templates have 'user=alice' and 'user=bob', candidate has 'user=(?P<user>.*?)'\n"
            "   Result: Delete/deactivate all conflicting templates, use the candidate template instead.\n"
            "   This merges multiple overly-specific templates into one properly generalized template.\n\n"
            "2. refine_candidate:\n"
            "   Use when conflicting templates represent distinct event types/variants that should remain separate.\n"
            "   Example: Conflicting templates have different suffixes/prefixes that are structural, not variable.\n"
            "   Result: Adjust the candidate regex to be more specific (add distinguishing structural constants) so it doesn't conflict.\n"
            "   This maintains template independence by making the new template capture a distinct variant.\n\n"
            "Rules:\n"
            "  • BUSINESS DATA (variables): Instance-specific unbounded values (timestamps, IPs, MACs, usernames, IDs, paths).\n"
            "    Capture with (?P<name>.*?)\n"
            "  • STRUCTURE (constants): System-defined phrases that determine event type (log levels, event verbs, module names, keywords, messages).\n"
            "    Keep literal and escape regex metacharacters.\n"
            "  • When choosing replace_conflicting, the candidate regex must match all conflicting template examples.\n"
            "  • When choosing refine_candidate, add structural constants to distinguish the candidate from conflicting templates.\n\n"
            "Return JSON:\n"
            "{\n"
            '  "reasoning": "explanation of why this decision is correct",\n'
            '  "decision": "replace_conflicting" | "refine_candidate",\n'
            '  "new_template": {\n'
            '    "regex": "the regex to use (candidate as-is for replace_conflicting, or refined for refine_candidate)",\n'
            '    "notes": "brief description of changes"\n'
            "  },\n"
            '  "replaced_templates": ["template_id1", "template_id2"]  // ONLY for replace_conflicting, list all conflicting template IDs\n'
            "}\n"
            "Respond with JSON only."
        )

        return (
            f"Context: {context}\n"
            f"{ts_hint}"
            "Candidate template:\n"
            f"  regex: {candidate_record.regex}\n"
            f"  capture_groups: {group_names}\n"
            f"  example_transformed: {candidate_sample.transformed}\n"
            f"  example_raw: {candidate_sample.raw}\n\n"
            "Conflicting templates:\n"
            f"{conflicts_blob}\n\n"
            f"{issues_section}"
            f"{instructions}"
        )


    @staticmethod
    def _normalize(value: Optional[str]) -> str:
        if not value:
            return ""
        return value.strip().lower()

    def _parse_plan(self, payload: dict) -> Optional[ResolutionPlan]:
        decision = self._normalize(payload.get("decision"))
        if decision not in {"replace_conflicting", "refine_candidate"}:
            return None

        new_template = payload.get("new_template", {})
        new_regex = new_template.get("regex")
        new_notes = new_template.get("notes", "")

        if not isinstance(new_regex, str) or not new_regex.strip():
            return None

        if not isinstance(new_notes, str):
            new_notes = str(new_notes) if new_notes else ""

        replaced_ids = []
        if decision == "replace_conflicting":
            replaced_ids = payload.get("replaced_templates", [])
            if not replaced_ids or not isinstance(replaced_ids, list):
                return None

        reasoning = payload.get("reasoning", "")
        if not isinstance(reasoning, str):
            reasoning = ""

        return ResolutionPlan(
            decision=decision,
            new_template_regex=new_regex.strip(),
            new_template_notes=new_notes,
            replaced_template_ids=replaced_ids,
            reasoning=reasoning,
        )
