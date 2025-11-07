"""
Parsing agent that derives structured templates from raw log lines.

This agent distinguishes between:
- Business Data Variables (fruits): Instance-specific, unbounded values (IP, MAC, timestamps, IDs)
- Log Structure Elements (branches): System-predefined, finite value sets (event types, log levels)
"""

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Dict, Iterable, List, Optional, Set

from agents.router_agent import RoutingResult
from libraries.template_library import TemplateRecord
from libraries.terminology import TerminologyLibrary
from agents.timestamp_agent import TimestampSpec
from libraries.enumeration_library import EnumerationLibrary
from utils.json_payloads import ProcessedLogLine


@dataclass
class ParsingOutcome:
    template_record: TemplateRecord
    variables: Dict[str, Dict]
    reasoning: str
    new_terms: List[str]
    raw_response: str = ""


class ParsingAgent:
    """
    Responsible for deriving parsing templates via LLM or heuristic hints.

    The agent enforces terminology consistency by funnelling all variable names
    through the shared terminology library, and distinguishes between:
    - Business data variables (open domain)
    - Structural enumerations (closed domain)
    """

    def __init__(
        self,
        api_client,
        terminology_library: TerminologyLibrary,
        enumeration_library: EnumerationLibrary,
    ) -> None:
        self.api_client = api_client
        self.terminology_library = terminology_library
        self.enumeration_library = enumeration_library
        self.last_raw_response: str = ""
        self.last_error: str = ""

    def derive_template(
        self,
        log_samples: Iterable[ProcessedLogLine],
        *,
        routing: RoutingResult,
        timestamp_spec: Optional[TimestampSpec] = None,
        feedback: Optional[str] = None,
    ) -> Optional[ParsingOutcome]:
        """
        Derive a template and regex for the provided log samples.
        """
        samples = [sample for sample in log_samples if sample.transformed]
        if not samples:
            self.last_error = "no valid samples"
            return None
        sample_texts = [sample.transformed for sample in samples]
        self.last_error = ""
        payload = self._call_llm(sample_texts, routing, timestamp_spec, feedback=feedback)
        raw_response = self.last_raw_response
        if not payload:
            if not self.last_error:
                self.last_error = "llm returned no usable json"
            return None
        return self._build_outcome(
            payload, samples, routing, timestamp_spec, raw_response
        )

    def build_outcome_from_regex(
        self,
        *,
        regex: str,
        sample: ProcessedLogLine,
        routing: RoutingResult,
        timestamp_spec: Optional[TimestampSpec] = None,
        reasoning: str = "",
        raw_response: str = "",
    ) -> Optional[ParsingOutcome]:
        """
        Construct a parsing outcome from an externally provided regex definition.
        """
        payload = {"regex": regex, "reasoning": reasoning}
        return self._build_outcome(
            payload, [sample], routing, timestamp_spec, raw_response
        )

    # ------------------------------------------------------------------ #
    # LLM interaction
    # ------------------------------------------------------------------ #
    def _call_llm(
        self,
        samples: List[str],
        routing: RoutingResult,
        timestamp_spec: Optional[TimestampSpec],
        *,
        feedback: Optional[str] = None,
    ) -> Optional[Dict]:
        if not self.api_client:
            self.last_error = "no api client"
            return None
        context = f"device_type={routing.device_type}, vendor={routing.vendor}"
        timestamp_hint = ""
        if timestamp_spec:
            timestamp_hint = (
                f"Known timestamp regex: {timestamp_spec.regex} "
                f"and format {timestamp_spec.format}. "
            )

        # Get known enumerations for context
        known_enums = self.enumeration_library.get_all_closed_enumerations()
        enum_hint = ""
        if known_enums:
            enum_examples = ", ".join(
                f"{k}={{{','.join(list(v)[:3])}}}"
                for k, v in list(known_enums.items())[:5]
            )
            enum_hint = f"Known enumerations: {enum_examples}. "

        instructions = (
            "You are a log parsing expert. Distinguish between:\n\n"
            "TYPE 1 - BUSINESS DATA (Variables): Instance-specific, unbounded values\n"
            "  • Timestamps, IPs, MACs, usernames, device names, IDs, metrics, paths, messages\n"
            "  • Capture as: (?P<name>.*?)\n"
            "  • Criteria: Unbounded domain, externally determined, substitution doesn't change event type\n\n"
            "TYPE 2 - STRUCTURE (Constants): System-defined, finite value sets\n"
            "  • Event skeletons: 'connected', 'timeout', 'sent deauth', 'failed to connect'\n"
            "  • Log levels: INFO, WARN, ERROR, DEBUG\n"
            "  • Protocol keywords: GET, POST, TCP, UDP, deauth, disassoc\n"
            "  • Module names: kernel, sshd, mdns (Note: PIDs are variables)\n"
            "  • Syntactic markers: from, to, by, at, delimiters like :, |, =\n"
            "  • Keep as literal text in regex\n"
            "  • Criteria: Finite set, system-defined, changing it changes event semantics\n\n"
            "Decision Rule:\n"
            "  1. Can enumerate all values? YES→constant, NO→variable\n"
            "  2. Does changing it alter event type? YES→constant, NO→variable\n"
            "  3. When uncertain: prefer constant (short phrases ≤3 words are usually structural)\n\n"
            "Examples:\n"
            "  Input: '2024-01-01 10:00:00 User alice connected from 192.168.1.100'\n"
            "  Output: '(?P<ts>.*?) User (?P<user>.*?) connected from (?P<ip>.*?)'\n"
            "  Constants: User, connected, from | Variables: ts, user, ip\n\n"
            "  Input: 'Oct 29 13:04:23 AP-403 mdns[1234]: AP sent deauth to sta 00:1A:2B:3C:4D:5E'\n"
            "  Output: '(?P<ts>.*?) (?P<ap>.*?) mdns\\[(?P<pid>.*?)\\]: AP sent deauth to sta (?P<mac>.*?)'\n"
            "  Constants: mdns, AP sent deauth to sta | Variables: ts, ap, pid, mac\n\n"
            "Return JSON:\n"
            "{\n"
            '  "reasoning": "<brief explanation>",\n'
            '  "regex": "<regex with (?P<name>.*?) for ALL variables>"\n'
            "}\n\n"
            "Requirements:\n"
            "  • Use ONLY (?P<name>.*?) for variables, no \\d+, \\w+, [0-9]\n"
            "  • Escape regex special chars in constants: [ ] ( ) { } . * + ? ^ $ \\ |\n"
            "  • Return only valid JSON, no markdown\n"
        )
        # Build context information
        context_parts = [f"Context: {context}"]
        if timestamp_hint:
            context_parts.append(f"Timestamp hint: {timestamp_hint}")
        if enum_hint:
            context_parts.append(enum_hint)

        context_str = "\n".join(context_parts) if context_parts else ""

        prompt_sections: List[str] = []
        if context_str:
            prompt_sections.append(context_str)
        if feedback:
            prompt_sections.append(
                "Feedback: Previous attempt produced an invalid template. "
                f"{feedback.strip()}"
            )
        prompt_sections.append(
            "Generate a regex template for this log line:\n\n"
            f"Log line: {samples[0]}\n\n"
            "Remember: Distinguish between business data variables (fruits) and "
            "structural elements (branches). Return ONLY the JSON object."
        )

        messages = [
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": "\n\n".join(section for section in prompt_sections if section),
            },
        ]

        try:
            raw = self.api_client.chat(messages)
            self.last_raw_response = raw
        except Exception as exc:
            self.last_raw_response = f"[LLM ERROR] {exc}"
            self.last_error = f"llm error: {exc}"
            return None
        return self._extract_json(raw)

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict]:
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

    # ------------------------------------------------------------------ #
    # Outcome builder
    # ------------------------------------------------------------------ #
    def _build_outcome(
        self,
        payload: Dict,
        samples: List[ProcessedLogLine],
        routing: RoutingResult,
        timestamp_spec: Optional[TimestampSpec],
        raw_response: str,
    ) -> Optional[ParsingOutcome]:
        regex = payload.get("regex")
        if not isinstance(regex, str) or not regex.strip():
            self.last_error = "llm response missing regex"
            return None

        regex = regex.strip()
        reasoning = payload.get("reasoning", "")
        try:
            compiled = re.compile(regex)
        except re.error as exc:
            self.last_error = f"invalid regex ({exc})"
            return None

        primary = samples[0]
        match = compiled.fullmatch(primary.transformed)
        if not match:
            self.last_error = "regex did not match sample"
            return None

        # Derive template string by replacing matched groups with <group_name>
        spans = []
        for name, index in sorted(compiled.groupindex.items(), key=lambda x: x[1]):
            span = match.span(name)
            if span == (-1, -1):
                continue
            spans.append((span[0], span[1], name))

        line = primary.transformed
        template_parts: List[str] = []
        cursor = 0
        for start, end, name in spans:
            template_parts.append(line[cursor:start])
            template_parts.append(f"<{name}>")
            cursor = end
        template_parts.append(line[cursor:])
        template_text = "".join(template_parts)

        variable_details: Dict[str, Dict] = {}
        new_terms: List[str] = []
        variables = []

        for _, _, name in spans:
            example = match.group(name)
            candidate = name

            # Resolve canonical name through terminology library
            entry = self.terminology_library.resolve(
                candidate,
                description="",
                example=example,
                context="",
            )
            canonical_name = entry.canonical_name
            if canonical_name != candidate:
                template_text = template_text.replace(f"<{candidate}>", f"<{canonical_name}>")

            # Observe variable in enumeration library for LLM-based classification
            self.enumeration_library.observe_variable(canonical_name, example)

            is_enum = self.enumeration_library.is_closed_enumeration(canonical_name)
            variables.append(
                {
                    "name": canonical_name,
                    "description": "",
                    "example": example,
                    "is_enumeration": is_enum,
                }
            )
            variable_details[canonical_name] = {
                "description": "",
                "example": example,
                "is_enumeration": is_enum,
            }
            if canonical_name != candidate:
                new_terms.append(canonical_name)

        # Extract constants from template by finding text between variables
        constants = self._extract_constants(template_text)

        template_id = self._make_template_id(template_text, routing)
        payload_summary = self._summarize_payloads(samples)

        record = TemplateRecord(
            template_id=template_id,
            template=template_text,
            regex=regex,
            variables=variables,
            constants=constants,
            timestamp_hint=timestamp_spec.to_dict() if timestamp_spec else None,
            source="llm",
            notes=reasoning[:200] if reasoning else "",  # Store reasoning in notes
            json_placeholders=payload_summary,
        )

        self.last_error = ""
        return ParsingOutcome(
            template_record=record,
            variables=variable_details,
            reasoning=reasoning,
            new_terms=new_terms,
            raw_response=raw_response,
        )

    def _summarize_payloads(
        self, samples: List[ProcessedLogLine]
    ) -> Dict[str, Dict[str, object]]:
        summary: Dict[str, Dict[str, object]] = {}
        key_sets: Dict[str, Set[str]] = {}

        for sample in samples:
            for payload in sample.payloads:
                info = summary.setdefault(
                    payload.placeholder,
                    {"owner": payload.owner, "keys": []},
                )
                if payload.owner and not info.get("owner"):
                    info["owner"] = payload.owner
                if payload.key_paths:
                    key_set = key_sets.setdefault(
                        payload.placeholder,
                        set(info.get("keys", [])),
                    )
                    previous_size = len(key_set)
                    key_set.update(payload.key_paths)
                    if len(key_set) != previous_size or not info.get("keys"):
                        info["keys"] = sorted(key_set)

        for placeholder, info in summary.items():
            if placeholder in key_sets:
                info["keys"] = sorted(key_sets[placeholder])
            elif "keys" not in info or info["keys"] is None:
                info["keys"] = []
        return summary

    def _extract_constants(self, template: str) -> List[str]:
        """Extract constant text segments from template."""
        # Split by variable markers <...>
        parts = re.split(r'<[^>]+>', template)
        # Filter out empty strings and whitespace-only strings
        constants = [part.strip() for part in parts if part.strip()]
        return constants

    @staticmethod
    def _make_template_id(template: str, routing: RoutingResult) -> str:
        digest = hashlib.sha1(template.encode("utf-8")).hexdigest()[:8]
        device = routing.device_type.replace(" ", "_")
        vendor = routing.vendor.replace(" ", "_")
        return f"{device}-{vendor}-{digest}"
