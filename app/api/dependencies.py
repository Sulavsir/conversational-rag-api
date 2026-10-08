from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.postgres import get_db_session
from app.db.qdrant import get_qdrant
from app.db.redis import get_redis
from app.repositories.booking_repository import BookingRepository
from app.repositories.document_repository import DocumentRepository
from app.services.booking_service import BookingService
from app.services.chat_service import ChatService
from app.services.embedding_service import EmbeddingService
from app.services.ingestion_service import IngestionService
from app.services.llm_service import LLMService, OpenAICompatibleClient
from app.services.memory_service import MemoryService
from app.services.rag_service import RAGService
from app.services.vector_service import VectorService

_embedding_service: EmbeddingService | None = None
_vector_service: VectorService | None = None
_llm_service: LLMService | None = None


def get_embedding_service() -> EmbeddingService:
    """Reuse one embedding model for the process. Loading it per request would be far too slow."""
    global _embedding_service
    if _embedding_service is None:
        settings = get_settings()
        _embedding_service = EmbeddingService(settings.embedding_model)
    return _embedding_service


def get_vector_service() -> VectorService:
    global _vector_service
    if _vector_service is None:
        settings = get_settings()
        _vector_service = VectorService(get_qdrant(), settings.qdrant_collection)
    return _vector_service


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        settings = get_settings()
        client = OpenAICompatibleClient(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        _llm_service = LLMService(client=client, model=settings.llm_model)
    return _llm_service


def get_ingestion_service(session: AsyncSession = Depends(get_db_session)) -> IngestionService:
    settings = get_settings()
    return IngestionService(
        repository=DocumentRepository(session),
        embeddings=get_embedding_service(),
        vectors=get_vector_service(),
        session=session,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_upload_bytes=settings.max_upload_bytes,
    )


def get_chat_service(session: AsyncSession = Depends(get_db_session)) -> ChatService:
    settings = get_settings()
    redis = get_redis()
    memory = MemoryService(
        redis,
        max_messages=settings.max_conversation_messages,
        ttl_seconds=settings.conversation_ttl_seconds,
    )
    booking = BookingService(
        redis=redis,
        repository=BookingRepository(session),
        session=session,
        llm=get_llm_service(),
        ttl_seconds=settings.conversation_ttl_seconds,
    )
    rag = RAGService(
        embeddings=get_embedding_service(),
        vectors=get_vector_service(),
        llm=get_llm_service(),
        top_k=settings.rag_top_k,
    )
    return ChatService(memory=memory, booking=booking, rag=rag)
