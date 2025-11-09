"""
Base class for all LLM-powered agents.
"""

import json
import re
from typing import Dict, List, Optional, Union


class BaseAgent:
    """Base class providing common functionality for LLM agents."""

    def __init__(self, api_client) -> None:
        if api_client is None:
            raise ValueError(f"{self.__class__.__name__} requires an api_client")
        self.api_client = api_client

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        extract_json: bool = True,
        save_raw: bool = False,
    ) -> Optional[Union[str, Dict]]:
        """
        Call LLM with system and user prompts.

        Args:
            system_prompt: System role prompt
            user_prompt: User message content
            extract_json: If True, extract and parse JSON from response
            save_raw: If True, save raw response to self.last_raw_response

        Returns:
            Parsed JSON dict if extract_json=True, raw string otherwise
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response = self.api_client.chat(messages)
            if save_raw and hasattr(self, 'last_raw_response'):
                self.last_raw_response = response
            if extract_json:
                return self._extract_json(response)
            return response
        except Exception as exc:
            if save_raw and hasattr(self, 'last_raw_response'):
                self.last_raw_response = f"[LLM ERROR] {exc}"
            return None

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict]:
        """Extract JSON from LLM response text."""
        try:
            cleaned = text.strip()
            if cleaned.startswith("{"):
                return json.loads(cleaned)
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group())
        except json.JSONDecodeError:
            pass
        return None
