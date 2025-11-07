"""
High level orchestrator wiring all agents into a cohesive parsing workflow.
"""

from collections import deque, OrderedDict
from dataclasses import dataclass
import json
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from core.api_client import AIMLAPIClient
from agents.router_agent import RouterAgent, RoutingResult
from libraries.template_library import TemplateLibraryManager, TemplateRecord
from libraries.terminology import TerminologyLibrary
from agents.timestamp_agent import TimestampAgent, TimestampSpec
from agents.parsing_agent import ParsingAgent, ParsingOutcome
from libraries.enumeration_library import EnumerationLibrary
from utils.json_payloads import ProcessedLogLine, preprocess_log_line
from utils.template_validator import TemplateValidator, TemplateValidationResult
from agents.conflict_resolution_agent import (
    ConflictResolutionAgent,
    ResolutionPlan,
)
from agents.template_refinement_agent import TemplateRefinementAgent
from core.status_reporting import ConsoleStatusReporter


def _preprocess_batch_worker(lines_batch):
    """Worker: preprocess batch of raw log lines."""
    from utils.json_payloads import preprocess_log_line
    return [preprocess_log_line(line) for line in lines_batch]


def _process_batch_worker(args):
    """Worker: match batch of log lines against templates."""
    import re

    batch_lines, templates_data, batch_start_idx = args

    # Compile regex patterns
    patterns = {}
    for tid, regex in templates_data.items():
        try:
            patterns[tid] = re.compile(regex)
        except re.error:
            pass

    matched = []
    unmatched = []

    for idx, processed in enumerate(batch_lines, batch_start_idx):
        found = False
        for tid, pattern in patterns.items():
            m = pattern.fullmatch(processed.transformed)
            if m:
                matched.append({
                    "line_number": idx,
                    "template_id": tid,
                    "variables": m.groupdict(),
                    "raw": processed.raw,
                    "json_payloads": [p.to_dict() for p in processed.payloads],
                })
                found = True
                break
        if not found:
            unmatched.append((idx, processed))

    return matched, unmatched


@dataclass
class ParseReport:
    routing: RoutingResult
    processed_lines: int
    matched_lines: int
    unmatched_lines: int
    new_templates: List[str]
    structured_output: Path
    template_snapshot: Path
    anomalies: List[str]


class LogParsingOrchestrator:
    """
    Full pipeline that reads a log file, routes it, loads or creates the
    template library, parses lines, and persists outputs.
    """

    def __init__(
        self,
        *,
        config_path: Path = Path("config.json"),
        model: str = "gemini-2.0-flash",
        progress_interval: int = 10000,
        batch_size: int = 50000,
        num_workers: Optional[int] = None,
    ) -> None:
        self.state_dir = Path("state")
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.config = self._load_config(config_path)
        self.api_client = AIMLAPIClient(model=model, config=self.config)

        self.router = RouterAgent(api_client=self.api_client)
        self.timestamp_agent = TimestampAgent(api_client=self.api_client)

        self.template_store_dir = self.state_dir / "template_libraries"
        self.library_manager = TemplateLibraryManager(self.template_store_dir)

        self.terminology_path = self.state_dir / "terminology.json"
        self.terminology_library = TerminologyLibrary(self.terminology_path)

        # New: Enumeration library for tracking closed/open domain variables
        self.enumeration_path = self.state_dir / "enumerations.json"
        self.enumeration_library = EnumerationLibrary(
            self.enumeration_path,
            api_client=self.api_client,  # Pass API client for LLM-based analysis
        )

        self.parsing_agent = ParsingAgent(
            api_client=self.api_client,
            terminology_library=self.terminology_library,
            enumeration_library=self.enumeration_library,
        )
        self.conflict_agent = ConflictResolutionAgent(
            api_client=self.api_client,
            parsing_agent=self.parsing_agent,
        )
        self.refinement_agent = TemplateRefinementAgent(
            api_client=self.api_client,
            parsing_agent=self.parsing_agent,
        )
        self.reporter = ConsoleStatusReporter()
        self.progress_interval = max(progress_interval, 1)
        self.batch_size = batch_size
        self.num_workers = num_workers or cpu_count()

        self.outputs_dir = self.state_dir / "outputs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def process_log_file(
        self,
        log_path: Path,
    ) -> ParseReport:
        output_dir = self.outputs_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        lines = self._read_log_lines(log_path)

        # Parallel preprocessing
        self.reporter.emit("progress", line_number=None, message="Preprocessing log lines (parallel)...", detail=None)
        line_batches = [lines[i:i + self.batch_size] for i in range(0, len(lines), self.batch_size)]
        with Pool(processes=self.num_workers) as pool:
            processed_batches = list(tqdm(
                pool.imap(_preprocess_batch_worker, line_batches),
                total=len(line_batches),
                desc="Preprocessing",
                unit="batch"
            ))
        processed_lines = [item for batch in processed_batches for item in batch]

        samples = [line for line in lines if line][:12]
        routing = self.router.identify_log_type(samples)

        library = self.library_manager.get_library(
            routing.device_type, routing.vendor
        )

        timestamp_spec = self._get_or_create_timestamp_spec(library, samples)

        parsed_results: List[Dict] = []
        template_examples: Dict[str, ProcessedLogLine] = {}
        unmatched: List[Tuple[int, ProcessedLogLine]] = []

        self.reporter.emit(
            "progress",
            line_number=None,
            message="Scanning logs with existing templates (parallel)...",
            detail=None,
        )

        # Prepare templates data for workers
        templates_data = {tid: rec.regex for tid, rec in library.templates.items() if rec.is_active}

        # Split into batches and process in parallel
        batches = []
        for i in range(0, len(processed_lines), self.batch_size):
            batch = processed_lines[i:i + self.batch_size]
            batches.append((batch, templates_data, i + 1))

        with Pool(processes=self.num_workers) as pool:
            results = list(tqdm(
                pool.imap(_process_batch_worker, batches),
                total=len(batches),
                desc="Template matching",
                unit="batch"
            ))

        # Merge results
        for batch_matched, batch_unmatched in results:
            for item in batch_matched:
                parsed_results.append(item)
                tid = item["template_id"]
                library.templates[tid].usage_count += 1
                if tid not in template_examples:
                    idx = item["line_number"] - 1
                    template_examples[tid] = processed_lines[idx]
            unmatched.extend(batch_unmatched)

        library._dirty_since_save = True
        matched_existing = len(parsed_results)

        self.reporter.emit(
            "progress",
            line_number=len(processed_lines),
            message=f"Initial pass: {len(processed_lines)} lines, matched {matched_existing}",
            detail=None,
        )

        new_templates: List[str] = []
        anomalies: List[str] = []

        template_validator = TemplateValidator(
            template_examples=template_examples,
        )

        if unmatched:
            added, learned_matches, still_unmatched = self._learn_new_templates(
                unmatched,
                library=library,
                routing=routing,
                timestamp_spec=timestamp_spec,
                template_validator=template_validator,
                template_examples=template_examples,
            )
            new_templates.extend(added)
            parsed_results.extend(learned_matches)
            unmatched = still_unmatched

            # Re-attempt parsing for lines covered by the new templates
            if new_templates:
                reparsed: List[Tuple[int, ProcessedLogLine]] = []
                for idx, processed in unmatched:
                    match = library.match(
                        processed.transformed, payloads=processed.payloads
                    )
                    if match:
                        record, groups = match
                        if record.template_id not in template_examples:
                            template_examples[record.template_id] = processed
                        parsed_results.append(
                            {
                                "line_number": idx,
                                "template_id": record.template_id,
                                "variables": groups,
                                "raw": processed.raw,
                                "json_payloads": [
                                    payload.to_dict() for payload in processed.payloads
                                ],
                            }
                        )
                    else:
                        reparsed.append((idx, processed))
                unmatched = reparsed

        if unmatched:
            anomalies.append(
                f"{len(unmatched)} lines remain unmatched after template learning"
            )
            for idx, processed in unmatched[:5]:
                anomalies.append(f"Unmatched line {idx}: {processed.raw[:120]}")
            for idx, processed in unmatched:
                parsed_results.append(
                    {
                        "line_number": idx,
                        "template_id": None,
                        "variables": {},
                        "raw": processed.raw,
                        "json_payloads": [
                            payload.to_dict() for payload in processed.payloads
                        ],
                    }
                )

        # Persist updated usage statistics once parsing completes.
        library.save_if_dirty()

        # Persist results
        structured_output = output_dir / f"{log_path.stem}.parsed.json"
        template_snapshot = output_dir / f"{log_path.stem}.templates.json"

        self.reporter.emit("progress", line_number=None, message="Writing results to disk...", detail=None)
        self._write_structured_output(structured_output, parsed_results)
        self._write_template_snapshot(template_snapshot, library)

        parsed_lines = sum(1 for item in parsed_results if item["template_id"])

        self.enumeration_library.save()

        return ParseReport(
            routing=routing,
            processed_lines=len(lines),
            matched_lines=parsed_lines,
            unmatched_lines=len([item for item in parsed_results if not item["template_id"]]),
            new_templates=new_templates,
            structured_output=structured_output,
            template_snapshot=template_snapshot,
            anomalies=anomalies,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _read_log_lines(self, log_path: Path) -> List[str]:
        with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
            return [line.rstrip("\n") for line in handle]

    def _load_config(self, config_path: Path) -> Dict:
        with Path(config_path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _get_or_create_timestamp_spec(
        self,
        library,
        samples: List[str],
    ) -> Optional[TimestampSpec]:
        if library.timestamp_spec:
            try:
                return TimestampSpec.from_dict(library.timestamp_spec)
            except KeyError:
                pass
        spec = self.timestamp_agent.infer_timestamp_spec(samples)
        if spec:
            library.update_timestamp_spec(spec.to_dict())
        return spec

    def _learn_new_templates(
        self,
        unmatched: List[Tuple[int, ProcessedLogLine]],
        *,
        library,
        routing: RoutingResult,
        timestamp_spec: Optional[TimestampSpec],
        template_validator: TemplateValidator,
        template_examples: Dict[str, ProcessedLogLine],
    ) -> Tuple[List[str], List[Dict[str, object]], List[Tuple[int, ProcessedLogLine]]]:
        new_template_ids: List[str] = []
        matched_entries: List[Dict[str, object]] = []
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
            existing_match = library.match(
                processed_line.transformed, payloads=processed_line.payloads
            )
            if existing_match:
                matched_record, groups = existing_match
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
                if conflict_pairs and self.conflict_agent:
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

                        plan = self.conflict_agent.resolve_conflict(
                            candidate_record=record,
                            candidate_sample=processed_line,
                            routing=routing,
                            timestamp_spec=timestamp_spec,
                            conflicting_records=[pair[0] for pair in conflict_pairs],
                            conflicting_samples=[pair[1] for pair in conflict_pairs],
                            issues=validation.reasons,
                        )
                        if not plan:
                            reason = self.conflict_agent.last_failure_reason
                            if reason:
                                self.reporter.emit(
                                    "conflict_failure",
                                    line_number=line_number,
                                    message=reason,
                                    detail=None,
                                    severity="error",
                                )
                                if self.conflict_agent.last_raw_response:
                                    self.reporter.emit(
                                        "conflict_plan",
                                        line_number=line_number,
                                        message=self.conflict_agent.last_raw_response,
                                        detail="raw_response",
                                        severity="warning",
                                    )
                            break
                        if self.conflict_agent.last_raw_response:
                            self.reporter.emit(
                                "conflict_plan",
                                line_number=line_number,
                                message=self.conflict_agent.last_raw_response,
                                detail="proposal",
                                severity="info",
                            )

                        current_conflict_ids = [
                            pair[0].template_id for pair in conflict_pairs
                        ]
                        record, validation, resolved_via_existing, resolution_note = self._apply_resolution_plan(
                            plan=plan,
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
                elif self.refinement_agent:
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
                        refined = self.refinement_agent.refine_template(
                            candidate_record=record,
                            candidate_sample=processed_line,
                            routing=routing,
                            timestamp_spec=timestamp_spec,
                            issues=validation.reasons,
                        )
                        if not refined:
                            reason = self.refinement_agent.last_failure_reason
                            if reason:
                                self.reporter.emit(
                                    "refinement_failure",
                                    line_number=line_number,
                                    message=reason,
                                    detail=None,
                                    severity="error",
                                )
                                if self.refinement_agent.last_raw_response:
                                    self.reporter.emit(
                                        "refinement_failure",
                                        line_number=line_number,
                                        message=self.refinement_agent.last_raw_response,
                                        detail="raw_response",
                                        severity="error",
                                    )
                            break
                        if self.refinement_agent.last_raw_response:
                            self.reporter.emit(
                                "refinement",
                                line_number=line_number,
                                message="updated regex",
                                detail=self.refinement_agent.last_raw_response,
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
                match_after_store = library.match(
                    processed_line.transformed,
                    payloads=processed_line.payloads,
                )
                if match_after_store:
                    stored_record, groups = match_after_store
                    matched_entries.append(
                        {
                            "line_number": line_number,
                            "template_id": stored_record.template_id,
                            "variables": groups,
                            "raw": processed_line.raw,
                            "json_payloads": [
                                payload.to_dict() for payload in processed_line.payloads
                            ],
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
        library,
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
        plan: ResolutionPlan,
        record: TemplateRecord,
        processed_line: ProcessedLogLine,
        library,
        routing: RoutingResult,
        timestamp_spec: Optional[TimestampSpec],
        template_examples: Dict[str, ProcessedLogLine],
        template_validator: TemplateValidator,
        conflict_ids: List[str],
        line_number: int,
    ) -> Tuple[Optional[TemplateRecord], TemplateValidationResult, bool, Optional[str]]:
        if plan.reasoning:
            self.reporter.emit(
                "resolution",
                line_number=line_number,
                message=plan.reasoning,
                detail=None,
            )

        status_note: Optional[str] = None

        if plan.decision == "discard_candidate":
            self.reporter.emit(
                "resolution",
                line_number=line_number,
                message="candidate discarded, existing templates kept",
                detail=None,
            )
            return None, TemplateValidationResult(True, []), True, "candidate discarded"

        if plan.decision == "update_existing":
            if not plan.existing_updates:
                reason = "update_existing decision missing template updates"
                return (
                    record,
                    TemplateValidationResult(False, [reason], conflict_ids),
                    False,
                    f"resolution failed: {reason}",
                )
            for update in plan.existing_updates:
                sample = template_examples.get(update.template_id)
                if not sample:
                    reason = f"update for {update.template_id} missing example"
                    return (
                        record,
                        TemplateValidationResult(False, [reason], conflict_ids),
                        False,
                        f"resolution failed: {reason}",
                    )
                new_record = self.conflict_agent.build_record_from_regex(
                    regex=update.regex,
                    sample=sample,
                    routing=routing,
                    timestamp_spec=timestamp_spec,
                    reasoning=update.notes,
                    raw_response=self.conflict_agent.last_raw_response,
                )
                if not new_record:
                    reason = f"update for {update.template_id} invalid"
                    return (
                        record,
                        TemplateValidationResult(False, [reason], conflict_ids),
                        False,
                        f"resolution failed: {reason}",
                    )
                new_record.template_id = update.template_id
                stored = library.add_template(new_record)
                if not stored:
                    reason = f"library rejected update for {update.template_id}"
                    return (
                        record,
                        TemplateValidationResult(False, [reason], conflict_ids),
                        False,
                        f"resolution failed: {reason}",
                    )
                library.reload_template(update.template_id)
                template_examples[new_record.template_id] = sample
                self.reporter.emit(
                    "resolution_update",
                    line_number=line_number,
                    message=f"updated template {new_record.template_id}",
                    detail=None,
                    severity="success",
                )

            match_after = library.match(
                processed_line.transformed,
                payloads=processed_line.payloads,
            )
            if match_after:
                matched_record, _ = match_after
                template_examples.setdefault(matched_record.template_id, processed_line)
                status_note = (
                    f"resolved via existing template {matched_record.template_id}"
                )
                return None, TemplateValidationResult(True, []), True, status_note

            reason = "updated existing templates but log remains unmatched"
            return (
                record,
                TemplateValidationResult(False, [reason], conflict_ids),
                False,
                f"resolution failed: {reason}",
            )

        # decision == "adjust_candidate"
        target_regex = plan.candidate_regex or record.regex
        if not target_regex:
            return (
                record,
                TemplateValidationResult(
                    False, ["candidate resolution missing regex"], conflict_ids
                ),
                False,
                "resolution failed: candidate regex missing",
            )
        new_candidate = self.conflict_agent.build_record_from_regex(
            regex=target_regex,
            sample=processed_line,
            routing=routing,
            timestamp_spec=timestamp_spec,
            reasoning=plan.candidate_notes,
            raw_response=self.conflict_agent.last_raw_response,
        )
        if not new_candidate:
            return (
                record,
                TemplateValidationResult(
                    False, ["failed to rebuild candidate template"], conflict_ids
                ),
                False,
                "resolution failed: candidate rebuild failed",
            )

        candidate_to_validate = new_candidate
        validation = template_validator.validate(
            candidate_to_validate,
            candidate_sample=processed_line,
        )
        status_note = "adjusted candidate regex via conflict resolution"
        return candidate_to_validate, validation, False, status_note

    def _write_structured_output(self, path: Path, rows: List[Dict]) -> None:
        # Group by template_id (rows already mostly sorted by parallel processing)
        grouped: "OrderedDict[Optional[str], List[Dict]]" = OrderedDict()
        for row in tqdm(rows, desc="Grouping results", unit="line", disable=len(rows) < 10000):
            template_id = row.get("template_id")
            bucket = grouped.setdefault(template_id, [])
            bucket.append(
                {
                    "line_number": row["line_number"],
                    "variables": row.get("variables", {}),
                    "raw": row.get("raw", ""),
                    "json_payloads": row.get("json_payloads", []),
                }
            )

        def sort_key(item: Tuple[Optional[str], List[Dict]]) -> Tuple[int, str]:
            template_id, _ = item
            if template_id is None:
                return (1, "")
            return (0, template_id)

        # Write JSON in chunks to avoid large memory allocation
        sorted_groups = sorted(grouped.items(), key=sort_key)
        with path.open("w", encoding="utf-8") as handle:
            handle.write('[')
            first = True
            for template_id, entries in tqdm(sorted_groups, desc="Writing JSON", unit="template"):
                if not first:
                    handle.write(',')
                first = False
                json.dump(
                    {"template_id": template_id, "logs": entries},
                    handle,
                    ensure_ascii=False
                )
            handle.write(']')

    def _write_template_snapshot(self, path: Path, library) -> None:
        data = {
            "metadata": library.metadata,
            "timestamp_spec": library.timestamp_spec,
            "templates": library.list_templates(),
            "enumerations": {n: sorted(v) for n, v in self.enumeration_library.get_all_closed_enumerations().items()},
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
