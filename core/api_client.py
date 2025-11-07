"""
AIML API client wrapper that issues HTTP requests via requests.
"""

from typing import Any, Dict, List

import requests


class AIMLAPIClient:
    """
    Lightweight client for the AIML API chat completions endpoint.
    """

    def __init__(
        self,
        *,
        model: str,
        config: Dict[str, Any],
        base_url: str = "https://api.aimlapi.com/v1",
        timeout: int = 60,
    ) -> None:
        self.model = model
        self.api_key = config.get("AIML_API_KEY")
        if not self.api_key:
            raise ValueError("AIML_API_KEY is required in config.")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Send chat messages to the AIML API and return the assistant response.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        url = f"{self.base_url}/chat/completions"
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:  # pragma: no cover - network failure
            raise RuntimeError(f"AIML API request failed: {exc} -> {response.text}") from exc

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"AIML API response missing content: {data}") from exc
        if not isinstance(content, str):
            raise RuntimeError(f"AIML API content is not string: {content!r}")
        return content.strip()
