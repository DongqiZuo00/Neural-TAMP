import os
import json
import re
import openai
from typing import Dict, Any, Optional


class LLMInterface:
    """OpenAI-compatible client wrapper.

    This works with:
      - OpenAI API (if you set OPENAI_BASE_URL appropriately)
      - Local vLLM OpenAI server (recommended defaults below)

    Notes:
      - vLLM + Qwen does not reliably support OpenAI "response_format=json_object".
        We enforce JSON via prompt and then parse.
    """

    def __init__(
        self,
        model: str = "Qwen2.5-32B",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        # Local vLLM doesn't require a real key; keep a placeholder to satisfy SDK.
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
        # Default to local vLLM OpenAI endpoint.
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or "http://127.0.0.1:8000/v1"
        self.model = model

        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    @staticmethod
    def _extract_json(text: str) -> str:
        """Best-effort extraction of the first JSON object from a string."""
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            return text
        # Try to find the first {...} block.
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            return m.group(0)
        return text

    def _parse_json(self, content: str) -> Dict[str, Any]:
        extracted = self._extract_json(content)
        return json.loads(extracted)

    def predict(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Send request and return parsed JSON dict (with basic error handling)."""
        try:
            # Enforce JSON via prompt to be compatible across providers.
            json_guard = "\n\nReturn ONLY a valid JSON object. No extra text, no markdown."
            messages = [
                {"role": "system", "content": system_prompt + json_guard},
                {"role": "user", "content": user_prompt},
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from LLM")

            try:
                return self._parse_json(content)
            except json.JSONDecodeError:
                print("❌ [LLM Error] Failed to parse JSON response.")
                print("Raw output:", content)

            repair_messages = [
                {
                    "role": "system",
                    "content": "You fix malformed JSON. Return ONLY a valid JSON object.",
                },
                {
                    "role": "user",
                    "content": f"Fix this into valid JSON only:\n{content}",
                },
            ]
            repair_response = self.client.chat.completions.create(
                model=self.model,
                messages=repair_messages,
                temperature=0.0,
                max_tokens=2048,
            )
            repaired = repair_response.choices[0].message.content
            if not repaired:
                raise ValueError("Empty repair response from LLM")
            return self._parse_json(repaired)

        except Exception as e:
            print(f"❌ [LLM Error] API Call failed: {e}")
            return {"error": str(e), "plan": []}
