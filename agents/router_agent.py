"""
Routing agent that maps raw log samples to device and vendor categories using LLM.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List

from agents.base_agent import BaseAgent


@dataclass(frozen=True)
class RoutingResult:
    """Structured result returned by the router."""

    device_type: str
    vendor: str
    reasoning: str = ""

    def is_unknown(self) -> bool:
        return self.device_type == "unknown" or self.vendor == "unknown"


class RouterAgent(BaseAgent):
    """Router agent that identifies device type and vendor using LLM."""

    CACHE_WINDOW = 6

    def __init__(self, api_client) -> None:
        super().__init__(api_client)
        self._cache: Dict[int, RoutingResult] = {}

    def identify_log_type(
        self, log_samples: Iterable[str], bypass_cache: bool = False
    ) -> RoutingResult:
        """Identify the device type and vendor using LLM."""
        samples = [s.strip() for s in log_samples if s and s.strip()]
        if not samples:
            return RoutingResult("unknown", "unknown", "No log lines")

        cache_key = hash(tuple(samples[: self.CACHE_WINDOW]))
        if not bypass_cache and cache_key in self._cache:
            return self._cache[cache_key]

        system_prompt = "You are a precise classifier. Respond with JSON only."
        user_prompt = (
            "You are a log routing specialist. "
            "Classify the following log samples and respond with strict JSON:\n"
            '{"device_type": "<category>", "vendor": "<vendor>", "reasoning": "<brief reasoning>"}\n'
            "Allowed device_type values: wifi_router, wifi_network, firewall, "
            "switch, application, mobile_device, server, storage, security, unknown.\n"
            "Vendor examples: aruba, ubiquiti, cisco, meraki, palo_alto, apple, android, generic, unknown.\n"
            "\n\nLog samples:\n" + "\n".join(samples[:12])
        )

        extracted = self._call_llm(system_prompt, user_prompt)
        if extracted:
            device = extracted.get("device_type", "unknown").strip() or "unknown"
            vendor = extracted.get("vendor", "unknown").strip() or "unknown"
            reasoning = extracted.get("reasoning", "LLM classification")
            result = RoutingResult(device, vendor, reasoning)
        else:
            result = RoutingResult("unknown", "unknown", "LLM failed")

        self._cache[cache_key] = result
        return result
