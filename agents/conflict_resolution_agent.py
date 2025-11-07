"""
Conflict resolution agent that can either adjust the candidate regex or update
conflicting templates when a newly learned pattern collides with existing ones.
"""

from dataclasses import dataclass
import json
import re
from typing import Iterable, List, Optional

from utils.json_payloads import ProcessedLogLine
from agents.parsing_agent import ParsingAgent
from agents.router_agent import RoutingResult
from libraries.template_library import TemplateRecord
from agents.timestamp_agent import TimestampSpec


@dataclass
class ExistingUpdate:
    template_id: str
    regex: str
    notes: str = ""


@dataclass
class ResolutionPlan:
    decision: str  # update_existing | adjust_candidate
    existing_updates: List[ExistingUpdate]
    candidate_regex: Optional[str] = None
    candidate_notes: str = ""
    reasoning: str = ""


class ConflictResolutionAgent:
    """Uses the LLM to propose conflict resolution plans."""

    def __init__(self, api_client, parsing_agent: ParsingAgent) -> None:
        self.api_client = api_client
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

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a log parsing architect. Resolve conflicts while preserving variable semantics. "
                    "Respond with JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.api_client.chat(messages)
            self.last_raw_response = response
        except Exception as exc:
            self.last_failure_reason = f"llm request failed: {exc}"
            return None

        payload = self._extract_json(response)
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
        group_names = ", ".join(var["name"] for var in candidate_record.variables) or "none"

        conflicts_blob = "\n".join(
            [
                (
                    f"- template_id: {record.template_id}\n"
                    f"  regex: {record.regex}\n"
                    f"  template: {record.template}\n"
                    f"  variables: {[var['name'] for var in record.variables]}\n"
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
            "Resolve this conflict by choosing exactly one decision:\n"
            "  • update_existing: revise the conflicting templates so they now match this candidate log while preserving their original intent. "
            "In most cases this means relaxing overly specific literals into variable captures so the existing template can absorb the new log.\n"
            "After these updates the candidate regex will be discarded.\n"
            "  • adjust_candidate: refine only the candidate regex so it no longer conflicts but still captures the candidate log semantics.\n"
            "  • discard_candidate: keep all existing templates unchanged and reject the candidate template.\n"
            "Rules:\n"
            "  • BUSINESS DATA (variables): Instance-specific, unbounded values such as timestamps, IPs, MACs, usernames, device names, IDs, metrics, paths, messages.\n"
            "    - Capture them with (?P<name>.*?) and keep the group semantics consistent.\n"
            "  • STRUCTURE (constants): System-defined phrases (event verbs, log levels, module names, protocol keywords, delimiters).\n"
            "    - These are short phrases that determine the event type; changing them alters semantics, so keep them literal and escape regex metacharacters.\n"
            "  • Keep structural constants literal and treat business data as variables.\n"
            "    - Constants are small, system-defined phrases that determine the event type.\n"
            "    - Variables are instance-specific values such as timestamps, IPs, MACs, IDs, names.\n"
            "  • When you claim update_existing, the revised regex must differ from the original and actually cover the candidate log; otherwise choose discard_candidate.\n"
            "  • Maintain capture group names and semantics; avoid renaming groups unless mandated by the examples.\n"
            "  • Use only (?P<name>.*?) for variables and escape constants properly.\n"
            "  • Do NOT hardcode example-specific values (such as literal IDs, paths, or hostnames) into constants unless the examples prove they are structural.\n"
            "  • When you choose update_existing, provide revised regexes for every template that must change, and ensure the candidate log will match at least one of them afterwards.\n"
            "  • When you choose adjust_candidate, provide a single revised candidate regex that resolves the conflict while preserving variable semantics.\n"
            "Return JSON:\n"
            "{\n"
            '  "reasoning": "...",\n'
            '  "decision": "update_existing" | "adjust_candidate" | "discard_candidate",\n'
            '  "updated_existing": [{"template_id": "...", "regex": "...", "notes": "..."}],\n'
            '  "updated_candidate": {"regex": "...", "notes": "..."}\n'
            "}\n"
            "Omit updated_existing when no template updates are needed, omit updated_candidate when update_existing is chosen, and leave both empty when discard_candidate is chosen. Respond with JSON only."
        )

        return (
            f"Context: {context}\n"
            f"{ts_hint}"
            "Candidate template:\n"
            f"  regex: {candidate_record.regex}\n"
            f"  template: {candidate_record.template}\n"
            f"  variables: {[var['name'] for var in candidate_record.variables]}\n"
            f"  example_transformed: {candidate_sample.transformed}\n"
            f"  example_raw: {candidate_sample.raw}\n"
            f"  required_capture_groups: {group_names}\n\n"
            "Conflicting templates:\n"
            f"{conflicts_blob}\n\n"
            f"{issues_section}"
            f"{instructions}"
        )

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        try:
            cleaned = text.strip()
            if cleaned.startswith("{"):
                return json.loads(cleaned)
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group())
        except json.JSONDecodeError:
            return None
        return None

    @staticmethod
    def _normalize(value: Optional[str]) -> str:
        if not value:
            return ""
        return value.strip().lower()

    def _parse_plan(self, payload: dict) -> Optional[ResolutionPlan]:
        decision = self._normalize(payload.get("decision"))
        if decision not in {"update_existing", "adjust_candidate", "discard_candidate"}:
            return None

        existing_updates: List[ExistingUpdate] = []
        for item in payload.get("updated_existing", []):
            template_id = item.get("template_id")
            regex = item.get("regex")
            notes = item.get("notes", "")
            if not template_id or not isinstance(regex, str) or not regex.strip():
                return None
            if notes is None:
                notes = ""
            elif not isinstance(notes, str):
                notes = str(notes)
            existing_updates.append(
                ExistingUpdate(
                    template_id=template_id,
                    regex=regex.strip(),
                    notes=notes,
                )
            )

        candidate_section = payload.get("updated_candidate") or {}
        candidate_regex = candidate_section.get("regex")
        candidate_notes = candidate_section.get("notes", "")
        if candidate_notes is None:
            candidate_notes = ""
        elif not isinstance(candidate_notes, str):
            candidate_notes = str(candidate_notes)
        if decision == "update_existing":
            if not existing_updates:
                return None
            candidate_regex_value: Optional[str] = None
        elif decision == "adjust_candidate":
            if not isinstance(candidate_regex, str) or not candidate_regex.strip():
                return None
            candidate_regex_value = candidate_regex.strip()
        else:  # discard_candidate
            if existing_updates or candidate_regex:
                return None
            candidate_regex_value = None

        reasoning = payload.get("reasoning", "")
        return ResolutionPlan(
            decision=decision,
            existing_updates=existing_updates,
            candidate_regex=candidate_regex_value,
            candidate_notes=candidate_notes,
            reasoning=reasoning,
        )
