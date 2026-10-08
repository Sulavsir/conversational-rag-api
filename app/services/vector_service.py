import logging
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.core.exceptions import ServiceUnavailableError

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class VectorRecord:
    id: str
    vector: list[float]
    document_id: str
    filename: str
    chunk_index: int
    chunking_strategy: str
    text: str


@dataclass(slots=True)
class ScoredChunk:
    document_id: UUID
    filename: str
    chunk_index: int
    text: str
    score: float


class VectorService:
    """Qdrant collection management, upsert, and similarity search."""

    def __init__(self, client: AsyncQdrantClient, collection_name: str) -> None:
        self._client = client
        self._collection = collection_name
        self._ready_dimension: int | None = None

    async def ensure_collection(self, dimension: int) -> None:
        if dimension <= 0:
            raise ServiceUnavailableError("Embedding dimension must be positive.")
        if self._ready_dimension == dimension:
            return
        try:
            exists = await self._client.collection_exists(self._collection)
            if not exists:
                await self._create_collection(dimension)
            else:
                info = await self._client.get_collection(self._collection)
                existing = _vector_size(info)
                if existing is None:
                    raise ServiceUnavailableError("The Qdrant collection vector size could not be read.")
                if existing != dimension:
                    raise ServiceUnavailableError(
                        "The Qdrant collection dimension does not match the embedding model."
                    )
        except ServiceUnavailableError:
            raise
        except Exception as exc:
            logger.exception("Qdrant collection setup failed")
            raise ServiceUnavailableError("Qdrant is unavailable.") from exc
        self._ready_dimension = dimension

    async def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        points = [
            PointStruct(
                id=record.id,
                vector=record.vector,
                payload={
                    "document_id": record.document_id,
                    "filename": record.filename,
                    "chunk_id": record.chunk_index,
                    "chunk_index": record.chunk_index,
                    "chunking_strategy": record.chunking_strategy,
                    "text": record.text,
                },
            )
            for record in records
        ]
        try:
            await self._client.upsert(collection_name=self._collection, points=points)
        except Exception as exc:
            logger.exception("Qdrant upsert failed")
            raise ServiceUnavailableError("Qdrant is unavailable.") from exc

    async def search(self, vector: list[float], top_k: int) -> list[ScoredChunk]:
        try:
            response = await self._client.query_points(
                collection_name=self._collection,
                query=vector,
                limit=top_k,
                with_payload=True,
            )
        except Exception as exc:
            logger.exception("Qdrant search failed")
            raise ServiceUnavailableError("Qdrant is unavailable.") from exc

        chunks: list[ScoredChunk] = []
        for point in response.points:
            chunk = _chunk_from_payload(point.payload, float(point.score or 0.0))
            if chunk is not None:
                chunks.append(chunk)
        return chunks

    async def delete_by_document_id(self, document_id: str) -> None:
        await self._client.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )

    async def _create_collection(self, dimension: int) -> None:
        try:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
        except Exception as exc:
            if "already exists" not in str(exc).lower():
                raise
            logger.info("Qdrant collection %s already exists", self._collection)


def _vector_size(collection: Any) -> int | None:
    vectors = collection.config.params.vectors
    size = getattr(vectors, "size", None)
    if isinstance(size, int):
        return size
    if isinstance(vectors, dict):
        for value in vectors.values():
            nested = getattr(value, "size", None)
            if isinstance(nested, int):
                return nested
    return None


def _chunk_from_payload(payload: Any, score: float) -> ScoredChunk | None:
    if not isinstance(payload, dict):
        return None
    try:
        document_id = UUID(str(payload["document_id"]))
        chunk_index = int(payload.get("chunk_index", payload["chunk_id"]))
        filename = str(payload["filename"])
        text = str(payload["text"])
    except (KeyError, TypeError, ValueError):
        logger.warning("Skipping Qdrant point with incomplete payload")
        return None
    return ScoredChunk(
        document_id=document_id,
        filename=filename,
        chunk_index=chunk_index,
        text=text,
        score=score,
    )
