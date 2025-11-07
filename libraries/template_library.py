"""
Persistent storage for log parsing templates grouped by device and vendor.
"""

from dataclasses import dataclass, field, asdict
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.json_payloads import JsonPayload


@dataclass
class TemplateRecord:
    """Serializable representation of a template with metadata."""

    template_id: str
    template: str
    regex: str
    variables: List[Dict[str, str]]
    constants: List[str] = field(default_factory=list)
    usage_count: int = 0
    failure_count: int = 0
    timestamp_hint: Optional[Dict[str, str]] = None
    source: str = "llm"
    notes: str = ""
    version: str = "v1"
    is_active: bool = True
    json_placeholders: Dict[str, Dict[str, object]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        data = asdict(self)
        return data

    @staticmethod
    def from_dict(payload: Dict) -> "TemplateRecord":
        return TemplateRecord(
            template_id=payload["template_id"],
            template=payload.get("template", ""),
            regex=payload["regex"],
            variables=payload.get("variables", []),
            constants=payload.get("constants", []),
            usage_count=payload.get("usage_count", 0),
            failure_count=payload.get("failure_count", 0),
            timestamp_hint=payload.get("timestamp_hint"),
            source=payload.get("source", "llm"),
            notes=payload.get("notes", ""),
            version=payload.get("version", "v1"),
            is_active=payload.get("is_active", True),
            json_placeholders=payload.get("json_placeholders", {}),
        )


class TemplateLibrary:
    """
    Stores templates for a specific device + vendor combination.

    Templates are persisted as JSON alongside optional metadata. This class
    handles matching, updates, and cleanup.
    """

    def __init__(
        self,
        *,
        device_type: str,
        vendor: str,
        storage_dir: Path,
        auto_flush: bool = True,
    ) -> None:
        self.device_type = device_type
        self.vendor = vendor
        self._device_key = self._normalize(device_type)
        self._vendor_key = self._normalize(vendor)
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.auto_flush = auto_flush
        self.path = self.storage_dir / f"{device_type}__{vendor}.json"

        self.metadata: Dict[str, str] = {
            "device_type": device_type,
            "vendor": vendor,
            "version": "v1",
        }
        self.templates: Dict[str, TemplateRecord] = {}
        self.timestamp_spec: Optional[Dict[str, str]] = None
        self._compiled: Dict[str, re.Pattern] = {}
        self._dirty_since_save: bool = False
        self._sequence: int = 1

        if self.path.exists():
            self._load()
        self._initialize_sequence()

    # ------------------------------------------------------------------ #
    # Matching / management
    # ------------------------------------------------------------------ #
    def match(
        self,
        line: str,
        *,
        payloads: Optional[List[JsonPayload]] = None,
    ) -> Optional[Tuple[TemplateRecord, Dict[str, str]]]:
        """Try to match a log line against known templates."""
        for template_id, record in self.templates.items():
            if not record.is_active:
                continue
            compiled = self._compiled.get(template_id)
            if not compiled:
                try:
                    compiled = re.compile(record.regex)
                except re.error:
                    continue
                self._compiled[template_id] = compiled
            match = compiled.fullmatch(line)
            if match:
                record.usage_count += 1
                self._dirty_since_save = True
                if payloads:
                    updated = False
                    for payload in payloads:
                        info = record.json_placeholders.setdefault(
                            payload.placeholder,
                            {
                                "owner": payload.owner,
                                "keys": list(payload.key_paths),
                            },
                        )
                        if payload.owner and not info.get("owner"):
                            info["owner"] = payload.owner
                            updated = True
                        if payload.key_paths:
                            existing_keys = set(info.get("keys", []))
                            new_keys = set(payload.key_paths)
                            if new_keys - existing_keys:
                                info["keys"] = sorted(existing_keys | new_keys)
                                updated = True
                    if updated:
                        self._dirty_since_save = True
                return record, match.groupdict()
        return None


    def add_template(self, record: TemplateRecord, *, validate: bool = True) -> bool:
        """
        Register a new template in the library.

        Returns:
            bool: True if the template was stored, False when validation failed.
        """
        try:
            compiled = re.compile(record.regex)
        except re.error as exc:
            print(
                f"[TemplateLibrary] Failed to store '{record.template_id}': invalid regex ({exc})."
            )
            return False

        existing_id = record.template_id
        if existing_id and existing_id in self.templates:
            template_id = existing_id
        else:
            template_id = self._allocate_template_id()
            record.template_id = template_id

        self.templates[template_id] = record
        self._compiled[template_id] = compiled
        self.metadata["next_sequence"] = str(self._sequence)
        self._dirty_since_save = True
        if self.auto_flush:
            self.save()
        return True

    def mark_failure(self, template_id: str) -> None:
        """Record a failure event for the template."""
        if template_id in self.templates:
            record = self.templates[template_id]
            record.failure_count += 1
            self._dirty_since_save = True
            if self.auto_flush:
                self.save()

    def update_timestamp_spec(self, spec: Dict[str, str]) -> None:
        """Store the timestamp extraction specification."""
        self.timestamp_spec = spec
        self._dirty_since_save = True
        if self.auto_flush:
            self.save()

    def cleanup(
        self,
        *,
        min_usage: int = 1,
        max_failure_ratio: float = 0.6,
        dry_run: bool = False,
    ) -> List[str]:
        """
        Clean up underperforming templates.

        Returns list of template_ids put on hold or removed.
        """
        removed: List[str] = []
        for template_id, record in list(self.templates.items()):
            total_events = record.usage_count + record.failure_count
            failure_ratio = (
                record.failure_count / total_events if total_events else 0.0
            )
            should_remove = (
                record.usage_count < min_usage
                or failure_ratio > max_failure_ratio
            )
            if should_remove:
                record.is_active = False
                removed.append(template_id)
        if removed and not dry_run:
            self._dirty_since_save = True
            if self.auto_flush:
                self.save()
        return removed

    def list_templates(self) -> List[Dict]:
        """Return serialized template definitions."""
        return [record.to_dict() for record in self.templates.values()]

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as exc:
            backup_path = self._quarantine_corrupted_file()
            print(
                f"[TemplateLibrary] Corrupted library '{self.path.name}' detected and "
                f"moved to '{backup_path.name}'. Starting with a fresh library. "
                f"Error details: {exc}"
            )
            self.metadata["corrupted_backup"] = str(backup_path.name)
            self.templates = {}
            self.timestamp_spec = None
            self._compiled.clear()
            return

        self.metadata.update(payload.get("metadata", {}))
        self.timestamp_spec = payload.get("timestamp_spec")
        self.templates = {
            record["template_id"]: TemplateRecord.from_dict(record)
            for record in payload.get("templates", [])
        }
        self._compiled.clear()
        self._dirty_since_save = False

    def _quarantine_corrupted_file(self) -> Path:
        suffix_counter = 0
        candidate = self.path.with_name(f"{self.path.name}.corrupted")
        while candidate.exists():
            suffix_counter += 1
            candidate = self.path.with_name(
                f"{self.path.name}.corrupted.{suffix_counter}"
            )
        self.path.rename(candidate)
        return candidate

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata["next_sequence"] = str(self._sequence)
        payload = {
            "metadata": self.metadata,
            "timestamp_spec": self.timestamp_spec,
            "templates": [record.to_dict() for record in self.templates.values()],
        }
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        self._dirty_since_save = False

    def save_if_dirty(self) -> None:
        """Persist in-memory usage stats if anything changed."""
        if self._dirty_since_save:
            self.save()

    def reload_template(self, template_id: str) -> Optional[TemplateRecord]:
        """Reload a template definition from disk and refresh the compiled cache."""
        if not self.path.exists():
            return self.templates.get(template_id)

        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError:
            return self.templates.get(template_id)

        for payload in data.get("templates", []):
            if payload.get("template_id") == template_id:
                record = TemplateRecord.from_dict(payload)
                self.templates[template_id] = record
                try:
                    self._compiled[template_id] = re.compile(record.regex)
                except re.error:
                    self._compiled.pop(template_id, None)
                    return None
                return record

        return self.templates.get(template_id)

    @staticmethod
    def _normalize(identifier: str) -> str:
        return identifier.replace(" ", "_")

    def _initialize_sequence(self) -> None:
        next_seq = self.metadata.get("next_sequence")
        try:
            self._sequence = int(next_seq) if next_seq is not None else 1
        except (TypeError, ValueError):
            self._sequence = 1

        prefix = f"{self._device_key}-{self._vendor_key}-"
        for template_id in self.templates.keys():
            if template_id.startswith(prefix):
                suffix = template_id[len(prefix) :]
                if suffix.isdigit():
                    candidate = int(suffix) + 1
                    if candidate > self._sequence:
                        self._sequence = candidate

        self.metadata["next_sequence"] = str(self._sequence)

    def _allocate_template_id(self) -> str:
        template_id = f"{self._device_key}-{self._vendor_key}-{self._sequence:04d}"
        self._sequence += 1
        return template_id


class TemplateLibraryManager:
    """Utility to lazily load template libraries per routing bucket."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._libraries: Dict[Tuple[str, str], TemplateLibrary] = {}

    def get_library(self, device_type: str, vendor: str) -> TemplateLibrary:
        key = (device_type, vendor)
        if key not in self._libraries:
            library = TemplateLibrary(
                device_type=device_type,
                vendor=vendor,
                storage_dir=self.base_dir,
            )
            self._libraries[key] = library
        return self._libraries[key]

    def list_known_libraries(self) -> List[Tuple[str, str]]:
        return list(self._libraries.keys())

    def cleanup_all(self, **kwargs) -> Dict[Tuple[str, str], List[str]]:
        summary: Dict[Tuple[str, str], List[str]] = {}
        for key, library in self._libraries.items():
            summary[key] = library.cleanup(**kwargs)
        return summary
