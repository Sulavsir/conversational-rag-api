from qdrant_client import AsyncQdrantClient

_client: AsyncQdrantClient | None = None


def init_qdrant(url: str, timeout_seconds: float) -> AsyncQdrantClient:
    global _client
    _client = AsyncQdrantClient(url=url, timeout=int(timeout_seconds))
    return _client


def get_qdrant() -> AsyncQdrantClient:
    if _client is None:
        raise RuntimeError("Qdrant client is not initialized")
    return _client


async def close_qdrant() -> None:
    global _client
    if _client is not None:
        await _client.close()
    _client = None
