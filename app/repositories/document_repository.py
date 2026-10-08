import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentChunk


class DocumentRepository:
    """PostgreSQL access for document and chunk metadata. Embeddings are not stored here."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add_document_with_chunks(
        self,
        *,
        document_id: uuid.UUID,
        filename: str,
        file_type: str,
        chunking_strategy: str,
        chunks: list[tuple[uuid.UUID, int, str]],
    ) -> Document:
        document = Document(
            id=document_id,
            filename=filename,
            file_type=file_type,
            chunking_strategy=chunking_strategy,
        )
        self._session.add(document)
        for chunk_id, chunk_index, text in chunks:
            self._session.add(
                DocumentChunk(
                    id=chunk_id,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    text=text,
                )
            )
        await self._session.flush()
        return document
