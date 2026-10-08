import asyncio
import logging
import re

from app.schemas.chat import ChatResponse, Source
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.memory_service import ConversationMessage
from app.services.vector_service import ScoredChunk, VectorService

logger = logging.getLogger(__name__)

_FOLLOWUP_PATTERN = re.compile(r"\b(he|she|they|his|her|their|him|them|that|this|it|those)\b", re.IGNORECASE)


class RAGService:
    """Manual retrieval pipeline. This service does not use RetrievalQA or any prebuilt RAG chain."""

    def __init__(
        self,
        *,
        embeddings: EmbeddingService,
        vectors: VectorService,
        llm: LLMService,
        top_k: int,
    ) -> None:
        self._embeddings = embeddings
        self._vectors = vectors
        self._llm = llm
        self._top_k = top_k

    async def answer(
        self,
        *,
        session_id: str,
        message: str,
        history: list[ConversationMessage],
    ) -> ChatResponse:
        query = build_search_query(message, history)
        vector = await asyncio.to_thread(self._embeddings.embed_text, query)
        await self._vectors.ensure_collection(len(vector))
        matches = await self._vectors.search(vector, self._top_k)
        answer = await self._llm.generate(
            build_system_prompt(matches),
            [*_history_messages(history), {"role": "user", "content": message}],
        )
        logger.info("Answered session %s with %s source chunks", session_id, len(matches))
        return ChatResponse(session_id=session_id, answer=answer, sources=_sources(matches))


def build_search_query(message: str, history: list[ConversationMessage]) -> str:
    """Use the previous user turn when the new question refers to it with a pronoun."""
    if not _FOLLOWUP_PATTERN.search(message):
        return message
    previous = [item.content for item in history if item.role == "user"]
    if not previous:
        return message
    return f"{previous[-1]}\n{message}"


def build_system_prompt(chunks: list[ScoredChunk]) -> str:
    if not chunks:
        context = "No relevant document context was retrieved."
    else:
        blocks = [
            (
                f"[source document_id={chunk.document_id} filename={chunk.filename} "
                f"chunk_id={chunk.chunk_index}]\n{chunk.text}"
            )
            for chunk in chunks
        ]
        context = "\n\n".join(blocks)
    return (
        "You answer questions about uploaded candidate documents.\n"
        "Use only the document context and the conversation history.\n"
        "If the context does not contain the answer, say that you do not have that information.\n"
        "Do not invent experience, employers, dates, skills, or project details.\n\n"
        f"Document context:\n{context}"
    )


def _history_messages(history: list[ConversationMessage]) -> list[dict[str, str]]:
    return [{"role": item.role, "content": item.content} for item in history]


def _sources(chunks: list[ScoredChunk]) -> list[Source]:
    sources: list[Source] = []
    seen: set[tuple[str, int]] = set()
    for chunk in chunks:
        key = (str(chunk.document_id), chunk.chunk_index)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            Source(
                document_id=chunk.document_id,
                filename=chunk.filename,
                chunk_id=chunk.chunk_index,
            )
        )
    return sources
