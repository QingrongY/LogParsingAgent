"""
Terminology library maintains canonical variable names across templates.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
import difflib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _normalize(identifier: str) -> str:
    return (
        identifier.strip()
        .replace("-", "_")
        .replace(" ", "_")
        .replace(".", "_")
        .lower()
    )


@dataclass
class TermEntry:
    canonical_name: str
    description: str = ""
    synonyms: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now)
    last_seen_at: str = field(default_factory=_utc_now)
    usage_count: int = 0
    tags: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(payload: Dict) -> "TermEntry":
        return TermEntry(
            canonical_name=payload["canonical_name"],
            description=payload.get("description", ""),
            synonyms=payload.get("synonyms", []),
            created_at=payload.get("created_at", _utc_now()),
            last_seen_at=payload.get("last_seen_at", _utc_now()),
            usage_count=payload.get("usage_count", 0),
            tags=payload.get("tags", []),
            examples=payload.get("examples", []),
        )


class TerminologyLibrary:
    """
    Maintains a shared vocabulary for template variables.

    Provides basic synonym resolution using string similarity and allows
    manual inspection via the persisted JSON file.
    """

    def __init__(self, storage_path: Path) -> None:
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: Dict[str, TermEntry] = {}
        if self.storage_path.exists():
            self._load()

    # ------------------------------------------------------------------ #
    # Resolution and registration
    # ------------------------------------------------------------------ #
    def resolve(
        self,
        candidate: str,
        *,
        description: str = "",
        context: Optional[str] = None,
        example: Optional[str] = None,
    ) -> TermEntry:
        """
        Resolve a candidate variable name to a canonical entry.

        If no suitable entry exists, a new canonical term is created.
        """
        normalized = _normalize(candidate)

        # Direct match by canonical name
        entry = self.entries.get(normalized)
        if entry:
            self._update_usage(entry, example=example)
            return entry

        # Match by synonym
        for stored in self.entries.values():
            if normalized in [_normalize(name) for name in stored.synonyms]:
                self._update_usage(stored, synonym=normalized, example=example)
                return stored

        # Approximate match
        candidate_names = list(self.entries.keys())
        if candidate_names:
            closest, score = self._fuzzy_match(normalized, candidate_names)
            if score >= 0.86:
                entry = self.entries[closest]
                self._update_usage(entry, synonym=normalized, example=example)
                if normalized not in [_normalize(name) for name in entry.synonyms]:
                    entry.synonyms.append(candidate)
                self.save()
                return entry

        # Create new entry
        canonical = normalized
        entry = TermEntry(
            canonical_name=canonical,
            description=description or context or "",
            synonyms=[candidate] if candidate != canonical else [],
        )
        entry.examples.append(example) if example else None
        self.entries[canonical] = entry
        self.save()
        return entry

    def add_synonym(self, canonical: str, synonym: str) -> None:
        entry = self.entries.get(_normalize(canonical))
        if not entry:
            raise ValueError(f"Canonical term '{canonical}' not found")
        normalized_synonyms = {_normalize(name) for name in entry.synonyms}
        if _normalize(synonym) not in normalized_synonyms:
            entry.synonyms.append(synonym)
            entry.last_seen_at = _utc_now()
            self.save()

    def register_manual(
        self,
        canonical_name: str,
        *,
        description: str = "",
        synonyms: Optional[Iterable[str]] = None,
    ) -> TermEntry:
        entry = TermEntry(
            canonical_name=_normalize(canonical_name),
            description=description,
            synonyms=list(synonyms or []),
        )
        self.entries[entry.canonical_name] = entry
        self.save()
        return entry

    def iter_entries(self) -> List[TermEntry]:
        return list(self.entries.values())

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _update_usage(
        self,
        entry: TermEntry,
        *,
        synonym: Optional[str] = None,
        example: Optional[str] = None,
    ) -> None:
        entry.usage_count += 1
        entry.last_seen_at = _utc_now()
        if example and example not in entry.examples:
            entry.examples.append(example)
        if synonym:
            normalized_synonyms = {_normalize(name) for name in entry.synonyms}
            if synonym not in normalized_synonyms:
                entry.synonyms.append(synonym)

    @staticmethod
    def _fuzzy_match(candidate: str, choices: List[str]) -> (str, float):
        match = difflib.get_close_matches(candidate, choices, n=1, cutoff=0.0)
        if match:
            closest = match[0]
            ratio = difflib.SequenceMatcher(None, candidate, closest).ratio()
            return closest, ratio
        return "", 0.0

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        with self.storage_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.entries = {
            entry["canonical_name"]: TermEntry.from_dict(entry)
            for entry in payload.get("entries", [])
        }

    def save(self) -> None:
        payload = {
            "metadata": {"updated_at": _utc_now(), "count": len(self.entries)},
            "entries": [entry.to_dict() for entry in self.entries.values()],
        }
        with self.storage_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
