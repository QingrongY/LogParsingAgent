"""
Routing agent that maps raw log samples to device and vendor categories.
"""

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class RoutingResult:
    """Structured result returned by the router."""

    device_type: str
    vendor: str
    source: str
    reasoning: str = ""

    def is_unknown(self) -> bool:
        return self.device_type == "unknown" or self.vendor == "unknown"


class RouterAgent:
    """
    Router agent that performs multi-layer routing (device type + vendor).

    The agent combines deterministic pattern rules with optional LLM-based
    fallback classification. Pattern rules are persisted so learned routing
    behaviour survives across runs.
    """

    CACHE_WINDOW = 6

    def __init__(
        self,
        api_client=None,
        patterns_path: Path = Path("agents/routing_patterns.json"),
    ) -> None:
        self.api_client = api_client
        self.patterns_path = patterns_path
        self.pattern_rules = self._load_patterns()
        self._classification_cache: Dict[int, RoutingResult] = {}

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def identify_log_type(
        self,
        log_samples: Iterable[str],
        *,
        allow_llm: bool = True,
        bypass_cache: bool = False,
    ) -> RoutingResult:
        """
        Identify the device type and vendor using the available routing signals.

        Args:
            log_samples: Iterable of sample log lines (ideally 5-10 lines).
            allow_llm: Permit LLM fallback if patterns are inconclusive.
            bypass_cache: Force re-evaluation even if samples were seen before.
        """
        normalized_samples = [
            sample.strip() for sample in log_samples if sample and sample.strip()
        ]
        if not normalized_samples:
            return RoutingResult(
                "unknown",
                "unknown",
                "pattern",
                "No log lines",
            )

        cache_key = hash(tuple(normalized_samples[: self.CACHE_WINDOW]))
        if not bypass_cache and cache_key in self._classification_cache:
            return self._classification_cache[cache_key]

        pattern_result = self._pattern_classify(normalized_samples)

        if pattern_result:
            result = pattern_result
        elif allow_llm and self.api_client:
            result = self._llm_classify(normalized_samples, pattern_result) or RoutingResult("unknown", "unknown", "llm", "LLM failed")
        else:
            result = RoutingResult("unknown", "unknown", "pattern", "No rule match")

        self._classification_cache[cache_key] = result
        return result

    def add_pattern_rule(
        self,
        *,
        device_type: str,
        vendor: str,
        patterns: List[str],
        min_matches: int = 2,
        notes: str = "",
    ) -> None:
        """Add a new deterministic routing rule."""
        entry = {
            "device_type": device_type,
            "vendor": vendor,
            "patterns": patterns,
            "min_matches": min_matches,
            "notes": notes,
        }
        self.pattern_rules.setdefault("rules", []).append(entry)
        self._persist_patterns()

    def remove_pattern_rule(self, device_type: str, vendor: str) -> None:
        """Remove pattern rules for a device/vendor pair."""
        before = len(self.pattern_rules.get("rules", []))
        self.pattern_rules["rules"] = [
            rule
            for rule in self.pattern_rules.get("rules", [])
            if not (
                rule.get("device_type") == device_type
                and rule.get("vendor") == vendor
            )
        ]
        after = len(self.pattern_rules.get("rules", []))
        if before != after:
            self._persist_patterns()

    def get_pattern_overview(self) -> List[Dict[str, str]]:
        """Return human-readable overview of routing rules."""
        overview = []
        for rule in self.pattern_rules.get("rules", []):
            overview.append(
                {
                    "device_type": rule.get("device_type", "unknown"),
                    "vendor": rule.get("vendor", "unknown"),
                    "pattern_count": str(len(rule.get("patterns", []))),
                    "min_matches": str(rule.get("min_matches", 0)),
                    "notes": rule.get("notes", ""),
                }
            )
        return overview

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _pattern_classify(self, samples: List[str]) -> Optional[RoutingResult]:
        rules = self.pattern_rules.get("rules", [])
        if not rules:
            return None

        best_match = 0
        best_result = None
        for rule in rules:
            matches = self._count_rule_matches(samples, rule)
            if matches >= rule.get("min_matches", 1) and matches > best_match:
                best_match = matches
                reasoning = f"Matched {matches} of {len(samples)} samples using pattern rule"
                best_result = RoutingResult(
                    rule.get("device_type", "unknown"),
                    rule.get("vendor", "unknown"),
                    "pattern",
                    reasoning,
                )

        return best_result

    def _count_rule_matches(self, samples: List[str], rule: Dict) -> int:
        patterns = rule.get("patterns", [])
        compiled = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        matches = 0
        for sample in samples:
            for pattern in compiled:
                if pattern.search(sample):
                    matches += 1
                    break
        return matches

    def _llm_classify(
        self,
        samples: List[str],
        pattern_result: Optional[RoutingResult],
    ) -> Optional[RoutingResult]:
        if not self.api_client:
            return None
        sample_blob = "\n".join(samples[:12])

        prior = ""
        if pattern_result:
            prior = (
                f"Pattern matching suggested device='{pattern_result.device_type}' "
                f"and vendor='{pattern_result.vendor}'. You may confirm or override."
            )

        prompt = (
            "You are a log routing specialist. "
            "Classify the following log samples and respond with strict JSON:\n"
            "{\n"
            '  "device_type": "<category>",\n'
            '  "vendor": "<vendor>",\n'
            '  "reasoning": "<brief reasoning>"\n'
            "}\n"
            "Allowed device_type values: wifi_router, wifi_network, firewall, "
            "switch, application, mobile_device, server, storage, security, unknown.\n"
            "Vendor examples: aruba, ubiquiti, cisco, meraki, palo_alto, apple, android, generic, unknown.\n"
            + (f"\n{prior}" if prior else "")
            + "\n\nLog samples:\n"
            + sample_blob
        )

        messages = [
            {
                "role": "system",
                "content": "You are a precise classifier. Respond with JSON only.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.api_client.chat(messages)
        except Exception:  # pragma: no cover - network failures
            return None

        extracted = self._extract_json(response)
        if not extracted:
            return None

        device = extracted.get("device_type", "unknown").strip() or "unknown"
        vendor = extracted.get("vendor", "unknown").strip() or "unknown"
        reasoning = extracted.get("reasoning", "LLM classification")
        return RoutingResult(device, vendor, "llm", reasoning)

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict]:
        try:
            if text.strip().startswith("{"):
                return json.loads(text.strip())
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, TypeError):
            return None
        return None

    # --------------------------------------------------------------------- #
    # Persistence
    # --------------------------------------------------------------------- #
    def _load_patterns(self) -> Dict:
        if self.patterns_path.exists():
            try:
                with self.patterns_path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                    if isinstance(data, dict) and "rules" in data:
                        return data
            except json.JSONDecodeError:
                pass
        return self._default_patterns()

    def _persist_patterns(self) -> None:
        self.patterns_path.parent.mkdir(parents=True, exist_ok=True)
        with self.patterns_path.open("w", encoding="utf-8") as handle:
            json.dump(self.pattern_rules, handle, ensure_ascii=False)

    @staticmethod
    def _default_patterns() -> Dict:
        # Curated heuristics for the provided datasets plus common Wi-Fi vendors.
        return {
            "rules": [
                {
                    "device_type": "wifi_network",
                    "vendor": "cisco_meraki",
                    "patterns": [
                        r"\bclient-authentication\b",
                        r"\bclient-disconnected\b",
                        r"\breassoc-(req|resp)\b",
                        r"\bssid\b",
                        r"\bAP sent\b",
                    ],
                    "min_matches": 2,
                    "notes": "Meraki CSV-style event logs",
                },
                {
                    "device_type": "wifi_router",
                    "vendor": "aruba",
                    "patterns": [
                        r"RUP\d{4}-\d+-A\d+-\d+",
                        r"\bmdns\[\d+\]",
                        r"\bfpapps\[\d+\]",
                        r"<(INFO|WARN|NOTI|ERR)>",
                    ],
                    "min_matches": 2,
                    "notes": "Aruba controller syslog messages",
                },
                {
                    "device_type": "wifi_router",
                    "vendor": "ubiquiti",
                    "patterns": [
                        r"\bU7(MP|LT)\b",
                        r"\bstahtd\[\d+\]",
                        r"\bSTA-TRACKER\b",
                        r"\bdev_\d+\b",
                    ],
                    "min_matches": 2,
                    "notes": "Ubiquiti UniFi events",
                },
                {
                    "device_type": "wifi_router",
                    "vendor": "generic",
                    "patterns": [
                        r"\bAP\b",
                        r"\bRSSI\b",
                        r"\bmac\b",
                        r"\bssid\b",
                        r"\bwifi\b",
                    ],
                    "min_matches": 1,
                    "notes": "Fallback Wi-Fi router",
                },
            ]
        }
