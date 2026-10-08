import asyncio
import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AppError, EmptyDocumentError, PayloadTooLargeError, ServiceUnavailableError
from app.repositories.document_repository import DocumentRepository
from app.schemas.document import ChunkingStrategy, DocumentUploadResponse
from app.services.chunking_service import chunk_text
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorRecord, VectorService
from app.utils.file_parser import extract_text, safe_filename

logger = logging.getLogger(__name__)


class IngestionService:
    """Upload pipeline: validate, extract, chunk, embed, store vectors, store metadata."""

    def __init__(
        self,
        *,
        repository: DocumentRepository,
        embeddings: EmbeddingService,
        vectors: VectorService,
        session: AsyncSession,
        chunk_size: int,
        chunk_overlap: int,
        max_upload_bytes: int,
    ) -> None:
        self._repository = repository
        self._embeddings = embeddings
        self._vectors = vectors
        self._session = session
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self.max_upload_bytes = max_upload_bytes

    async def ingest(
        self,
        *,
        filename: str | None,
        content: bytes,
        chunking_strategy: ChunkingStrategy,
    ) -> DocumentUploadResponse:
        safe_name = safe_filename(filename)
        if len(content) > self.max_upload_bytes:
            raise PayloadTooLargeError(f"The file exceeds the {self.max_upload_bytes} byte upload limit.")

        file_type, text = await asyncio.to_thread(extract_text, safe_name, content)
        chunks = await asyncio.to_thread(
            chunk_text,
            text,
            chunking_strategy,
            self._chunk_size,
            self._chunk_overlap,
        )
        if not chunks:
            raise EmptyDocumentError()

        vectors = await asyncio.to_thread(self._embeddings.embed_texts, chunks)
        if len(vectors) != len(chunks) or not vectors[0]:
            raise ServiceUnavailableError("Failed to generate embeddings.")

        document_id = uuid.uuid4()
        rows: list[tuple[uuid.UUID, int, str]] = []
        records: list[VectorRecord] = []
        for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
            chunk_id = uuid.uuid4()
            rows.append((chunk_id, index, chunk))
            records.append(
                VectorRecord(
                    id=str(chunk_id),
                    vector=vector,
                    document_id=str(document_id),
                    filename=safe_name,
                    chunk_index=index,
                    chunking_strategy=chunking_strategy.value,
                    text=chunk,
                )
            )

        stored_vectors = False
        try:
            await self._vectors.ensure_collection(len(vectors[0]))
            await self._repository.add_document_with_chunks(
                document_id=document_id,
                filename=safe_name,
                file_type=file_type,
                chunking_strategy=chunking_strategy.value,
                chunks=rows,
            )
            await self._vectors.upsert(records)
            stored_vectors = True
            await self._session.commit()
        except AppError:
            await self._rollback(stored_vectors, document_id)
            raise
        except Exception as exc:
            await self._rollback(stored_vectors, document_id)
            logger.exception("Document ingestion failed")
            raise ServiceUnavailableError("The document could not be stored.") from exc

        logger.info(
            "Ingested document %s filename=%s strategy=%s chunks=%s",
            document_id,
            safe_name,
            chunking_strategy.value,
            len(chunks),
        )
        return DocumentUploadResponse(
            document_id=document_id,
            filename=safe_name,
            file_type=file_type,
            chunking_strategy=chunking_strategy,
            chunk_count=len(chunks),
        )

    async def _rollback(self, stored_vectors: bool, document_id: uuid.UUID) -> None:
        await self._session.rollback()
        if not stored_vectors:
            return
        try:
            await self._vectors.delete_by_document_id(str(document_id))
        except Exception:
            logger.exception("Failed to roll back Qdrant points for document %s", document_id)
