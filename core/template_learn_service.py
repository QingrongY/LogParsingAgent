"""
Service responsible for learning new templates from unmatched log lines.

This service encapsulates the complex logic of template derivation, validation,
conflict resolution, and refinement.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

from agents.parsing_agent import ParsingAgent, ParsingOutcome
from agents.router_agent import RoutingResult
from agents.timestamp_agent import TimestampSpec
from core.status_reporting import ConsoleStatusReporter
from libraries.template_library import TemplateLibrary, TemplateRecord
from utils.preprocessing import ProcessedLogLine
from utils.template_validator import TemplateValidator, TemplateValidationResult


class TemplateLearnService:
    """
    Service for learning new templates from unmatched log lines.

    Orchestrates the interaction between parsing, validation, conflict resolution,
    and template refinement agents to derive high-quality templates.
    """

    def __init__(
        self,
        parsing_agent: ParsingAgent,
        reporter: ConsoleStatusReporter,
    ) -> None:
        self.parsing_agent = parsing_agent
        self.reporter = reporter

    def learn_templates(
        self,
        unmatched: List[Tuple[int, ProcessedLogLine]],
        *,
        library: TemplateLibrary,
        routing: RoutingResult,
        timestamp_spec: Optional[TimestampSpec],
        template_validator: TemplateValidator,
        template_examples: Dict[str, ProcessedLogLine],
    ) -> Tuple[List[str], List[Dict], List[Tuple[int, ProcessedLogLine]]]:
        """
        Learn new templates from unmatched log lines.

        Returns:
            Tuple of (new_template_ids, matched_entries, remaining_unmatched)
        """
        new_template_ids: List[str] = []
        matched_entries: List[Dict] = []
        if not unmatched:
            return new_template_ids, matched_entries, []

        remaining = {idx: text for idx, text in unmatched}
        attempted: set[int] = set()
        queue = deque(sorted(remaining.keys()))

        def log_status(
            line_number: Optional[int],
            message: str,
            detail: Optional[str] = None,
            *,
            severity: str = "info",
        ) -> None:
            self.reporter.emit(
                "learning_status",
                line_number=line_number,
                message=message,
                detail=detail,
                severity=severity,
            )

        while queue:
            line_number = queue.popleft()

            processed_line = remaining.get(line_number)
            if processed_line is None:
                continue

            last_resolution_note: Optional[str] = None

            # Skip if already matched by an existing or newly learned template
            existing_match = library.match(processed_line.transformed)
            if existing_match:
                remaining.pop(line_number, None)
                attempted.discard(line_number)
                continue

            if line_number in attempted:
                continue

            self.reporter.emit(
                "log_sample",
                line_number=line_number,
                message=processed_line.raw,
                detail=None,
            )
            outcome = self.parsing_agent.derive_template(
                [processed_line],
                routing=routing,
                timestamp_spec=timestamp_spec,
            )
            raw_response = self.parsing_agent.last_raw_response or ""
            if raw_response:
                self.reporter.emit(
                    "llm_response",
                    line_number=line_number,
                    message=raw_response,
                    detail="parsing_agent",
                )
            else:
                self.reporter.emit(
                    "llm_missing",
                    line_number=line_number,
                    message="No response received.",
                    detail="parsing_agent",
                    severity="warning",
                )
            if not outcome:
                outcome = self._retry_invalid_regex(
                    processed_line=processed_line,
                    routing=routing,
                    timestamp_spec=timestamp_spec,
                    raw_response=raw_response,
                    log_status=log_status,
                    line_number=line_number,
                    attempted=attempted,
                )
                if not outcome:
                    continue
            record = outcome.template_record
            validation = template_validator.validate(
                record,
                candidate_sample=processed_line,
            )

            conflict_pairs: List[Tuple[TemplateRecord, ProcessedLogLine]] = []
            if validation.conflicting_template_ids:
                conflict_pairs = self._gather_conflict_pairs(
                    library,
                    template_examples,
                    validation.conflicting_template_ids,
                )

            if not validation.is_valid:
                if conflict_pairs and outcome:
                    attempts = 0
                    while attempts < 3 and not validation.is_valid:
                        attempts += 1
                        conflict_ids_str = ", ".join(
                            item.template_id for item, _ in conflict_pairs
                        )
                        self.reporter.emit(
                            "conflict",
                            line_number=line_number,
                            message=f"{record.template_id or '[unnamed]'} vs {conflict_ids_str}",
                            detail=f"attempt {attempts}",
                            severity="warning",
                        )
                        for existing_record, sample in conflict_pairs:
                            self.reporter.emit(
                                "conflict_detail",
                                line_number=line_number,
                                message=f"existing {existing_record.template_id}",
                                detail=f"regex={existing_record.regex}",
                                severity="warning",
                            )
                            self.reporter.emit(
                                "conflict_detail",
                                line_number=line_number,
                                message="example",
                                detail=sample.transformed,
                                severity="warning",
                            )

                        resolution_dict = self.parsing_agent.resolve_conflict(
                            initial_outcome=outcome,
                            candidate_sample=processed_line,
                            conflicting_records=[pair[0] for pair in conflict_pairs],
                            conflicting_samples=[pair[1] for pair in conflict_pairs],
                        )
                        if not resolution_dict:
                            reason = self.parsing_agent.last_error
                            if reason:
                                self.reporter.emit(
                                    "conflict_failure",
                                    line_number=line_number,
                                    message=reason,
                                    detail=None,
                                    severity="error",
                                )
                                if self.parsing_agent.last_raw_response:
                                    self.reporter.emit(
                                        "conflict_plan",
                                        line_number=line_number,
                                        message=self.parsing_agent.last_raw_response,
                                        detail="raw_response",
                                        severity="warning",
                                    )
                            break
                        if self.parsing_agent.last_raw_response:
                            self.reporter.emit(
                                "conflict_plan",
                                line_number=line_number,
                                message=self.parsing_agent.last_raw_response,
                                detail="proposal",
                                severity="info",
                            )

                        current_conflict_ids = [
                            pair[0].template_id for pair in conflict_pairs
                        ]
                        record, validation, resolved_via_existing, resolution_note = self._apply_resolution_plan(
                            plan=resolution_dict,
                            record=record,
                            processed_line=processed_line,
                            library=library,
                            routing=routing,
                            timestamp_spec=timestamp_spec,
                            template_examples=template_examples,
                            template_validator=template_validator,
                            conflict_ids=current_conflict_ids,
                            line_number=line_number,
                        )
                        if resolution_note:
                            last_resolution_note = resolution_note
                        if resolved_via_existing:
                            remaining.pop(line_number, None)
                            attempted.discard(line_number)
                            break
                        if record is None:
                            break
                        if validation.conflicting_template_ids:
                            conflict_pairs = self._gather_conflict_pairs(
                                library,
                                template_examples,
                                validation.conflicting_template_ids,
                            )
                        else:
                            conflict_pairs = []

                    if record is None:
                        note = last_resolution_note or "candidate removed after resolution"
                        lowered = note.lower()
                        if lowered.startswith("resolution failed"):
                            severity = "error"
                        elif lowered.startswith("resolved"):
                            severity = "success"
                        else:
                            severity = "info"
                        log_status(
                            line_number,
                            "resolution outcome",
                            detail=note,
                            severity=severity,
                        )
                        continue
                    if not validation.is_valid and conflict_pairs:
                        remaining_conflicts = [
                            pair[0].template_id for pair in conflict_pairs
                        ]
                        reason = "conflict persists after reconciliation attempts"
                        validation = TemplateValidationResult(
                            False, [reason], remaining_conflicts
                        )
                else:
                    attempts = 0
                    while attempts < 3 and not validation.is_valid:
                        attempts += 1
                        self.reporter.emit(
                            "refinement",
                            line_number=line_number,
                            message=f"{record.template_id or '[unnamed]'}",
                            detail=f"attempt {attempts}",
                            severity="warning",
                        )
                        refined = self.parsing_agent.refine_template(
                            candidate_record=record,
                            candidate_sample=processed_line,
                            routing=routing,
                            timestamp_spec=timestamp_spec,
                            issues=validation.reasons,
                        )
                        if not refined:
                            reason = self.parsing_agent.last_error or "refinement failed"
                            if reason:
                                self.reporter.emit(
                                    "refinement_failure",
                                    line_number=line_number,
                                    message=reason,
                                    detail=None,
                                    severity="error",
                                )
                                if self.parsing_agent.last_raw_response:
                                    self.reporter.emit(
                                        "refinement_failure",
                                        line_number=line_number,
                                        message=self.parsing_agent.last_raw_response,
                                        detail="raw_response",
                                        severity="error",
                                    )
                            break
                        if self.parsing_agent.last_raw_response:
                            self.reporter.emit(
                                "refinement",
                                line_number=line_number,
                                message="updated regex",
                                detail=self.parsing_agent.last_raw_response,
                                severity="success",
                            )

                        refined.template_id = record.template_id
                        record = refined
                        validation = template_validator.validate(
                            record,
                            candidate_sample=processed_line,
                        )
                    if record is None:
                        continue

            if not validation.is_valid:
                reasons_summary = "; ".join(validation.reasons) if validation.reasons else "unspecified validator failure"
                if last_resolution_note:
                    reasons_summary = f"{last_resolution_note}; {reasons_summary}"
                log_status(
                    line_number,
                    "rejected",
                    detail=f"{record.template_id or '[unnamed]'} — {reasons_summary}",
                    severity="error",
                )
                attempted.add(line_number)
                continue
            previous_id = record.template_id
            stored = library.add_template(record)
            if stored:
                new_id = record.template_id
                new_template_ids.append(new_id)
                remaining.pop(line_number, None)
                attempted.discard(line_number)
                if previous_id != new_id:
                    template_examples.pop(previous_id, None)
                template_examples.setdefault(new_id, processed_line)
                match_after_store = library.match(processed_line.transformed)
                if match_after_store:
                    stored_record, groups = match_after_store
                    matched_entries.append(
                        {
                            "line_number": line_number,
                            "template_id": stored_record.template_id,
                            "variables": groups,
                            "raw": processed_line.raw,
                        }
                    )
                detail_parts = [f"template {new_id}"]
                if last_resolution_note:
                    detail_parts.append(last_resolution_note)
                log_status(
                    line_number,
                    "stored template",
                    detail="; ".join(detail_parts),
                    severity="success",
                )
            else:
                library.mark_failure(record.template_id)
                attempted.add(line_number)
                detail_parts = [f"template {record.template_id}"]
                if last_resolution_note:
                    detail_parts.append(last_resolution_note)
                log_status(
                    line_number,
                    "storage failed",
                    detail="; ".join(detail_parts),
                    severity="error",
                )

        remaining_items = sorted(
            ((idx, text) for idx, text in remaining.items()), key=lambda x: x[0]
        )
        return new_template_ids, matched_entries, remaining_items

    def _gather_conflict_pairs(
        self,
        library: TemplateLibrary,
        template_examples: Dict[str, ProcessedLogLine],
        conflict_ids: List[str],
    ) -> List[Tuple[TemplateRecord, ProcessedLogLine]]:
        pairs: List[Tuple[TemplateRecord, ProcessedLogLine]] = []
        for conflict_id in conflict_ids:
            record = library.templates.get(conflict_id)
            sample = template_examples.get(conflict_id)
            if record and sample:
                pairs.append((record, sample))
        return pairs

    def _retry_invalid_regex(
        self,
        *,
        processed_line: ProcessedLogLine,
        routing: RoutingResult,
        timestamp_spec: Optional[TimestampSpec],
        raw_response: str,
        log_status,
        line_number: int,
        attempted: set[int],
    ) -> Optional[ParsingOutcome]:
        error_detail = self.parsing_agent.last_error or "invalid regex"
        feedback = (
            f"Regex compilation failed: {error_detail}. "
            "Return valid JSON with a corrected regex that fully matches the sample."
        )
        if raw_response:
            snippet = raw_response.strip()
            if len(snippet) > 800:
                snippet = snippet[:800] + " …"
            feedback += f"\nPrevious response:\n{snippet}"

        for attempt in range(2):
            revised = self.parsing_agent.derive_template(
                [processed_line],
                routing=routing,
                timestamp_spec=timestamp_spec,
                feedback=f"{feedback}\nRetry {attempt + 1}.",
            )
            new_raw = self.parsing_agent.last_raw_response or ""
            if new_raw:
                self.reporter.emit(
                    "llm_response",
                    line_number=line_number,
                    message=new_raw,
                    detail=f"parsing_agent_retry_{attempt + 1}",
                )
            if revised:
                return revised
            error_detail = self.parsing_agent.last_error or error_detail
            feedback = (
                f"Regex still invalid: {error_detail}. "
                "Generate a correct regex that matches the provided log line."
            )

        log_status(
            line_number,
            "derivation failed",
            detail=f"invalid or non-matching regex ({error_detail})",
            severity="error",
        )
        attempted.add(line_number)
        return None

    def _apply_resolution_plan(
        self,
        *,
        plan: Dict,
        record: TemplateRecord,
        processed_line: ProcessedLogLine,
        library: TemplateLibrary,
        routing: RoutingResult,
        timestamp_spec: Optional[TimestampSpec],
        template_examples: Dict[str, ProcessedLogLine],
        template_validator: TemplateValidator,
        conflict_ids: List[str],
        line_number: int,
    ) -> Tuple[Optional[TemplateRecord], TemplateValidationResult, bool, Optional[str]]:
        """
        Apply conflict resolution plan.

        Args:
            plan: Dict with keys: decision, new_regex, replaced_ids, reasoning

        Returns:
            Tuple of (record, validation, resolved_via_existing, status_note)
        """
        reasoning = plan.get("reasoning", "")
        if reasoning:
            self.reporter.emit(
                "resolution",
                line_number=line_number,
                message=reasoning,
                detail=None,
            )

        new_regex = plan.get("new_regex", "")
        new_outcome = self.parsing_agent.build_outcome_from_regex(
            regex=new_regex,
            sample=processed_line,
            routing=routing,
            timestamp_spec=timestamp_spec,
            reasoning=reasoning,
            raw_response=self.parsing_agent.last_raw_response,
        )
        if not new_outcome:
            reason = "failed to build template from resolution"
            return (
                record,
                TemplateValidationResult(False, [reason], conflict_ids),
                False,
                f"resolution failed: {reason}",
            )

        new_record = new_outcome.template_record
        decision = plan.get("decision", "")

        if decision == "replace_conflicting":
            replaced_ids = plan.get("replaced_ids", [])
            for template_id in replaced_ids:
                if template_id in library.templates:
                    library.templates[template_id].is_active = False
                    library._dirty_since_save = True
                    template_examples.pop(template_id, None)
                    self.reporter.emit(
                        "resolution_merge",
                        line_number=line_number,
                        message=f"deactivated {template_id}",
                        detail="replaced by new template",
                        severity="info",
                    )

            validation = template_validator.validate(
                new_record,
                candidate_sample=processed_line,
            )
            status_note = f"replaced {len(replaced_ids)} conflicting templates"
            return new_record, validation, False, status_note

        # decision == "refine_candidate"
        self.reporter.emit(
            "resolution_refine",
            line_number=line_number,
            message=f"refined regex",
            detail=new_regex,
            severity="info",
        )
        validation = template_validator.validate(
            new_record,
            candidate_sample=processed_line,
        )
        if validation.is_valid:
            self.reporter.emit(
                "resolution_refine",
                line_number=line_number,
                message="validation passed",
                detail=f"new template_id: {new_record.template_id}",
                severity="success",
            )
        else:
            conflict_msg = ", ".join(validation.conflicting_template_ids) if validation.conflicting_template_ids else "unknown"
            self.reporter.emit(
                "resolution_refine",
                line_number=line_number,
                message="validation failed",
                detail=f"still conflicts with: {conflict_msg}",
                severity="warning",
            )
        status_note = "refined candidate to avoid conflicts"
        return new_record, validation, False, status_note
