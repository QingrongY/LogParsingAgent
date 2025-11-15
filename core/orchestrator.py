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
from agents.timestamp_agent import TimestampAgent, TimestampSpec
from agents.parsing_agent import ParsingAgent, ParsingOutcome
from utils.preprocessing import ProcessedLogLine, preprocess_log_line
from utils.template_validator import TemplateValidator, TemplateValidationResult
from core.status_reporting import ConsoleStatusReporter
from core.template_learn_service import TemplateLearnService


def _preprocess_batch_worker(lines_batch):
    """Worker: preprocess batch of raw log lines."""
    from utils.preprocessing import preprocess_log_line
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

        self.parsing_agent = ParsingAgent(
            api_client=self.api_client,
        )
        self.reporter = ConsoleStatusReporter()
        self.learn_service = TemplateLearnService(
            parsing_agent=self.parsing_agent,
            reporter=self.reporter,
        )
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
            added, learned_matches, still_unmatched = self.learn_service.learn_templates(
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
                    match = library.match(processed.transformed)
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
                    }
                )

        # Persist updated usage statistics once parsing completes.
        library.save_if_dirty()

        # Persist results
        structured_output = output_dir / f"{log_path.stem}.parsed.tsv"
        template_snapshot = output_dir / f"{log_path.stem}.templates.json"

        self.reporter.emit("progress", line_number=None, message="Writing results to disk...", detail=None)
        self._write_structured_output(structured_output, parsed_results)
        self._write_template_snapshot(template_snapshot, library)

        parsed_lines = sum(1 for item in parsed_results if item["template_id"])

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

    def _write_structured_output(self, path: Path, rows: List[Dict]) -> None:
        with path.open("w", encoding="utf-8", newline='') as handle:
            handle.write("line\ttemplate_id\traw\n")
            for row in tqdm(sorted(rows, key=lambda x: x['line_number']), desc="Writing TSV", unit="line", disable=len(rows) < 10000):
                line_num = row['line_number']
                template_id = row.get('template_id') or ''
                raw = row.get('raw', '').replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
                handle.write(f"{line_num}\t{template_id}\t{raw}\n")

    def _write_template_snapshot(self, path: Path, library) -> None:
        data = {
            "metadata": library.metadata,
            "timestamp_spec": library.timestamp_spec,
            "templates": library.list_templates(),
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
