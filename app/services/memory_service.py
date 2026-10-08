import json
import logging
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from app.core.exceptions import ServiceUnavailableError

logger = logging.getLogger(__name__)

_HISTORY_PREFIX = "chat:"


class SupportsRedis(Protocol):
    async def get(self, key: str) -> str | bytes | None: ...

    async def set(self, key: str, value: str, ex: int | None = None) -> object: ...


class ConversationMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["user", "assistant"]
    content: str


class MemoryService:
    """Conversation history stored in Redis and capped so a session cannot grow forever."""

    def __init__(self, redis: SupportsRedis, *, max_messages: int, ttl_seconds: int) -> None:
        self._redis = redis
        self._max_messages = max_messages
        self._ttl_seconds = ttl_seconds

    async def get_history(self, session_id: str) -> list[ConversationMessage]:
        try:
            raw = await self._redis.get(_history_key(session_id))
        except ServiceUnavailableError:
            raise
        except Exception as exc:
            logger.exception("Failed to read conversation history")
            raise ServiceUnavailableError("Redis is unavailable.") from exc
        if raw is None:
            return []
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            payload = json.loads(raw)
            return [ConversationMessage.model_validate(item) for item in payload]
        except (json.JSONDecodeError, ValidationError, TypeError):
            logger.warning("Discarding unreadable conversation history for session %s", session_id)
            return []

    async def append(self, session_id: str, user_message: str, assistant_message: str) -> None:
        history = await self.get_history(session_id)
        history.append(ConversationMessage(role="user", content=user_message))
        history.append(ConversationMessage(role="assistant", content=assistant_message))
        trimmed = history[-self._max_messages :]
        payload = json.dumps([item.model_dump() for item in trimmed])
        try:
            await self._redis.set(_history_key(session_id), payload, ex=self._ttl_seconds)
        except Exception as exc:
            logger.exception("Failed to write conversation history")
            raise ServiceUnavailableError("Redis is unavailable.") from exc


def _history_key(session_id: str) -> str:
    return f"{_HISTORY_PREFIX}{session_id}"
