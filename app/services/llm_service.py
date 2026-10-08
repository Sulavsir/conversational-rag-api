import logging
from typing import Protocol

from app.core.exceptions import ServiceUnavailableError

logger = logging.getLogger(__name__)


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
            if text:
                parts.append(str(text))
        return "\n".join(parts).strip()
    if content is None:
        return ""
    return str(content).strip()


class LLMClient(Protocol):
    async def complete(self, *, model: str, messages: list[dict[str, str]]) -> str: ...


class OpenAICompatibleClient:
    """Chat Completions client. Set LLM_BASE_URL to use another compatible provider."""

    def __init__(self, *, api_key: str, base_url: str | None, timeout_seconds: float) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._timeout_seconds = timeout_seconds
        self._client: object | None = None

    async def complete(self, *, model: str, messages: list[dict[str, str]]) -> str:
        if not self._api_key:
            raise ServiceUnavailableError("LLM_API_KEY is not configured.")
        try:
            response = await self._get_client().chat.completions.create(model=model, messages=messages)
        except ServiceUnavailableError:
            raise
        except Exception as exc:
            logger.exception("LLM request failed")
            raise ServiceUnavailableError("The language model is unavailable.") from exc

        content = response.choices[0].message.content if response.choices else None
        text = _message_text(content)
        if not text:
            raise ServiceUnavailableError("The language model returned an empty response.")
        return text

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            kwargs: dict[str, object] = {"api_key": self._api_key, "timeout": self._timeout_seconds}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = AsyncOpenAI(**kwargs)
        return self._client


class LLMService:
    """Application-facing LLM port. The concrete client is injected so tests and providers stay swappable."""

    def __init__(self, client: LLMClient, model: str) -> None:
        self._client = client
        self._model = model

    async def generate(self, system_prompt: str, messages: list[dict[str, str]]) -> str:
        payload = [{"role": "system", "content": system_prompt}, *messages]
        try:
            return await self._client.complete(model=self._model, messages=payload)
        except ServiceUnavailableError:
            raise
        except Exception as exc:
            logger.exception("LLM request failed")
            raise ServiceUnavailableError("The language model is unavailable.") from exc
