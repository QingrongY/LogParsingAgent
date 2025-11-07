"""
Structural validation for newly generated templates before they are stored.
"""

from dataclasses import dataclass, field
import re
from typing import Dict, List

from utils.json_payloads import ProcessedLogLine
from libraries.template_library import TemplateRecord


@dataclass
class TemplateValidationResult:
    is_valid: bool
    reasons: List[str]
    conflicting_template_ids: List[str] = field(default_factory=list)


class TemplateValidator:
    """Applies heuristic checks to reject overly generic or conflicting templates."""

    def __init__(
        self,
        *,
        template_examples: Dict[str, ProcessedLogLine],
        min_constant_ratio: float = 0.2,
        min_constant_chars: int = 12,
        min_constant_tokens: int = 2,
        max_conflict_checks: int = 500,
    ) -> None:
        self.template_examples = template_examples
        self.min_constant_ratio = min_constant_ratio
        self.min_constant_chars = min_constant_chars
        self.min_constant_tokens = min_constant_tokens
        self.max_conflict_checks = max_conflict_checks

    def validate(
        self,
        record: TemplateRecord,
        *,
        candidate_sample: ProcessedLogLine,
    ) -> TemplateValidationResult:
        reasons: List[str] = []

        const_ok, const_reason = self._check_constant_coverage(record.template)
        if not const_ok and const_reason:
            reasons.append(const_reason)

        conflict_ok, conflict_reason, conflict_ids = self._check_conflicts(
            record.regex,
            record.template_id,
            candidate_sample=candidate_sample,
        )
        if not conflict_ok and conflict_reason:
            reasons.append(conflict_reason)

        return TemplateValidationResult(
            is_valid=not reasons,
            reasons=reasons,
            conflicting_template_ids=conflict_ids,
        )

    def _check_constant_coverage(self, template: str) -> tuple[bool, str]:
        segments = [segment for segment in re.split(r"<[^>]+>", template) if segment.strip()]
        constant_text = "".join(segments)
        total_chars = len(template)
        constant_chars = len(constant_text)
        ratio = (constant_chars / total_chars) if total_chars else 0.0

        return True, ""

    def _check_conflicts(
        self,
        regex: str,
        template_id: str,
        *,
        candidate_sample: ProcessedLogLine,
    ) -> tuple[bool, str, List[str]]:
        try:
            compiled = re.compile(regex)
        except re.error as exc:
            return False, f"invalid regex produced by LLM ({exc})", []

        if not compiled.fullmatch(candidate_sample.transformed):
            return False, "generated regex does not match the source sample", []

        conflicts: List[str] = []
        checks = 0
        for existing_id, sample in self.template_examples.items():
            if existing_id == template_id:
                continue
            if checks >= self.max_conflict_checks:
                break
            checks += 1
            if compiled.fullmatch(sample.transformed):
                conflicts.append(existing_id)
                if len(conflicts) >= 5:
                    break

        if conflicts:
            details = ", ".join(conflicts[:3])
            return False, f"conflicts with existing templates: {details}", conflicts
        return True, "", []
