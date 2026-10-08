from uuid import UUID

import pymupdf
import pytest
from fastapi.testclient import TestClient
from qdrant_client.models import Distance
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from app.api.dependencies import get_ingestion_service
from app.core.exceptions import EmptyDocumentError, InvalidFileError, PayloadTooLargeError, ServiceUnavailableError
from app.models import Booking, Document, DocumentChunk
from app.schemas.document import ChunkingStrategy
from app.services.chunking_service import chunk_text
from app.services.embedding_service import EmbeddingService
from app.services.ingestion_service import IngestionService
from app.services.vector_service import VectorRecord, VectorService
from app.utils.file_parser import extract_text
from tests.fakes import FakeDocumentRepository, FakeEmbeddings, FakeSession, FakeVectors


def test_fixed_chunking_uses_overlap_windows() -> None:
    chunks = chunk_text("abcdefghij", ChunkingStrategy.FIXED, chunk_size=6, chunk_overlap=2)

    assert chunks == ["abcdef", "efghij"]


def test_recursive_chunking_keeps_paragraphs_and_applies_overlap() -> None:
    text = "Alpha paragraph stays whole.\n\nBeta paragraph stays whole."

    together = chunk_text(text, ChunkingStrategy.RECURSIVE, chunk_size=80, chunk_overlap=10)
    assert len(together) == 1
    assert "Alpha paragraph stays whole." in together[0]
    assert "Beta paragraph stays whole." in together[0]

    split = chunk_text(text, ChunkingStrategy.RECURSIVE, chunk_size=40, chunk_overlap=0)
    assert split == ["Alpha paragraph stays whole.", "Beta paragraph stays whole."]

    overlapped = chunk_text(text, ChunkingStrategy.RECURSIVE, chunk_size=40, chunk_overlap=8)
    assert overlapped[0] == "Alpha paragraph stays whole."
    assert overlapped[1].endswith("Beta paragraph stays whole.")
    assert overlapped[0][-8:].strip() in overlapped[1]
    assert all(len(chunk) <= 40 for chunk in overlapped)


def test_fixed_and_recursive_strategies_differ() -> None:
    text = "Alpha paragraph stays whole.\n\nBeta paragraph stays whole."

    fixed = chunk_text(text, ChunkingStrategy.FIXED, chunk_size=40, chunk_overlap=0)
    recursive = chunk_text(text, ChunkingStrategy.RECURSIVE, chunk_size=40, chunk_overlap=0)

    assert fixed != recursive


def test_embedding_service_batches_and_reuses_one_model() -> None:
    class Model:
        def __init__(self) -> None:
            self.calls = 0

        def encode(self, sentences: list[str], *, normalize_embeddings: bool = True) -> list[list[float]]:
            self.calls += 1
            assert normalize_embeddings is True
            return [[float(len(sentence)), 1.0] for sentence in sentences]

        def get_embedding_dimension(self) -> int:
            return 2

    model = Model()
    service = EmbeddingService("test-model", model)

    assert service.embed_texts(["ab", "cdef"]) == [[2.0, 1.0], [4.0, 1.0]]
    assert service.embed_text("abc") == [3.0, 1.0]
    assert service.dimension == 2
    assert model.calls == 2


def test_chunking_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_text("hello", ChunkingStrategy.FIXED, chunk_size=10, chunk_overlap=10)


def test_txt_extraction_and_empty_or_invalid_files() -> None:
    file_type, text = extract_text("notes.txt", "Candidate knows retrieval.\n".encode())
    assert file_type == "txt"
    assert text == "Candidate knows retrieval."

    with pytest.raises(InvalidFileError):
        extract_text("resume.docx", b"hello")
    with pytest.raises(EmptyDocumentError):
        extract_text("notes.txt", b"")
    with pytest.raises(EmptyDocumentError):
        extract_text("notes.txt", b" \n\t")
    with pytest.raises(InvalidFileError):
        extract_text("notes.txt", b"\xff\xfehello")


def test_pdf_extraction_and_corrupt_pdf() -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Candidate built a search system.")
    content = document.tobytes()
    document.close()

    file_type, text = extract_text("resume.pdf", content)

    assert file_type == "pdf"
    assert "search system" in text
    with pytest.raises(InvalidFileError):
        extract_text("resume.pdf", b"this is not a pdf")


def test_postgres_models_compile_to_expected_tables() -> None:
    dialect = postgresql.dialect()
    statements = {
        table.name: str(CreateTable(table).compile(dialect=dialect))
        for table in (Document.__table__, DocumentChunk.__table__, Booking.__table__)
    }

    assert "filename" in statements["documents"]
    assert "chunking_strategy" in statements["documents"]
    assert "document_chunks" in statements
    assert "chunk_index" in statements["document_chunks"]
    assert "interview_date" in statements["bookings"]
    assert "interview_time" in statements["bookings"]


@pytest.mark.asyncio
async def test_vector_collection_uses_model_dimension_and_cosine() -> None:
    class FakeQdrant:
        def __init__(self) -> None:
            self.vectors_config = None
            self.points = []

        async def collection_exists(self, name: str) -> bool:
            return False

        async def create_collection(self, collection_name: str, vectors_config) -> None:
            self.vectors_config = vectors_config

        async def upsert(self, collection_name: str, points) -> None:
            self.points.extend(points)

        async def query_points(self, **kwargs):
            class Point:
                payload = {
                    "document_id": "2d9c8e3a-1b2c-4d5e-8f90-123456789abc",
                    "filename": "resume.pdf",
                    "chunk_id": 3,
                    "chunk_index": 3,
                    "chunking_strategy": "recursive",
                    "text": "Built a retrieval system.",
                }
                score = 0.8

            class Response:
                points = [Point()]

            return Response()

        async def delete(self, collection_name: str, points_selector) -> None:
            return None

    client = FakeQdrant()
    service = VectorService(client, "documents")  # type: ignore[arg-type]
    await service.ensure_collection(384)
    await service.upsert(
        [
            VectorRecord(
                id="5c56c793-69f3-4fbf-87e6-c4bf54c28c26",
                vector=[0.1, 0.2],
                document_id="2d9c8e3a-1b2c-4d5e-8f90-123456789abc",
                filename="resume.pdf",
                chunk_index=3,
                chunking_strategy="recursive",
                text="Built a retrieval system.",
            )
        ]
    )
    found = await service.search([0.1, 0.2], top_k=5)

    assert client.vectors_config.size == 384
    assert client.vectors_config.distance == Distance.COSINE
    payload = client.points[0].payload
    assert payload["chunk_id"] == 3
    assert payload["chunking_strategy"] == "recursive"
    assert found[0].document_id == UUID("2d9c8e3a-1b2c-4d5e-8f90-123456789abc")
    assert found[0].chunk_index == 3


@pytest.mark.asyncio
async def test_existing_collection_must_match_the_embedding_dimension() -> None:
    class Vectors:
        size = 8

    class Params:
        vectors = Vectors()

    class Config:
        params = Params()

    class Info:
        config = Config()

    class Client:
        async def collection_exists(self, name: str) -> bool:
            return True

        async def get_collection(self, name: str) -> Info:
            return Info()

    service = VectorService(Client(), "documents")  # type: ignore[arg-type]
    with pytest.raises(ServiceUnavailableError):
        await service.ensure_collection(384)


def _service(
    *,
    chunk_size: int = 40,
    chunk_overlap: int = 0,
    max_upload_bytes: int = 100,
    vectors: FakeVectors | None = None,
) -> tuple[IngestionService, FakeDocumentRepository, FakeVectors, FakeSession]:
    session = FakeSession()
    repository = FakeDocumentRepository(session)
    vector_store = vectors or FakeVectors()
    service = IngestionService(
        repository=repository,  # type: ignore[arg-type]
        embeddings=FakeEmbeddings(),  # type: ignore[arg-type]
        vectors=vector_store,  # type: ignore[arg-type]
        session=session,  # type: ignore[arg-type]
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_upload_bytes=max_upload_bytes,
    )
    return service, repository, vector_store, session


@pytest.mark.asyncio
async def test_txt_upload_stores_metadata_and_vectors() -> None:
    service, repository, vectors, session = _service()
    content = b"Alpha paragraph stays whole.\n\nBeta paragraph stays whole."

    result = await service.ingest(filename="notes.txt", content=content, chunking_strategy=ChunkingStrategy.RECURSIVE)

    assert result.filename == "notes.txt"
    assert result.file_type == "txt"
    assert result.chunking_strategy == ChunkingStrategy.RECURSIVE
    assert result.chunk_count == 2
    assert session.committed is True
    stored = repository.documents[0]
    assert stored["chunking_strategy"] == "recursive"
    assert stored["chunks"][0][1] == 0
    assert "Alpha paragraph stays whole." in stored["chunks"][0][2]
    assert vectors.dimension == 4
    assert len(vectors.records) == 2
    assert vectors.records[0].chunking_strategy == "recursive"


@pytest.mark.asyncio
async def test_upload_rejects_invalid_extension_empty_file_and_oversize() -> None:
    service, repository, _, session = _service(max_upload_bytes=8)

    with pytest.raises(InvalidFileError):
        await service.ingest(filename="resume.docx", content=b"hello", chunking_strategy=ChunkingStrategy.FIXED)
    with pytest.raises(EmptyDocumentError):
        await service.ingest(filename="notes.txt", content=b"   ", chunking_strategy=ChunkingStrategy.FIXED)
    with pytest.raises(PayloadTooLargeError):
        await service.ingest(filename="notes.txt", content=b"0123456789", chunking_strategy=ChunkingStrategy.FIXED)

    assert repository.documents == []
    assert session.committed is False


@pytest.mark.asyncio
async def test_qdrant_failure_rolls_back_postgres_work() -> None:
    vectors = FakeVectors()
    vectors.fail_upsert = True
    service, repository, _, session = _service(vectors=vectors)

    with pytest.raises(ServiceUnavailableError):
        await service.ingest(
            filename="notes.txt",
            content=b"Candidate knows retrieval.",
            chunking_strategy=ChunkingStrategy.FIXED,
        )

    assert session.committed is False
    assert session.rolled_back is True
    assert repository.documents == []


def test_upload_route(app) -> None:
    service, repository, _, _ = _service()
    app.dependency_overrides[get_ingestion_service] = lambda: service
    text = b"Alpha paragraph stays whole.\n\nBeta paragraph stays whole."

    with TestClient(app) as client:
        invalid = client.post(
            "/api/v1/documents/upload",
            files={"file": ("resume.docx", b"hello", "application/octet-stream")},
            data={"chunking_strategy": "fixed"},
        )
        empty = client.post(
            "/api/v1/documents/upload",
            files={"file": ("notes.txt", b"   ", "text/plain")},
            data={"chunking_strategy": "fixed"},
        )
        bad_strategy = client.post(
            "/api/v1/documents/upload",
            files={"file": ("notes.txt", text, "text/plain")},
            data={"chunking_strategy": "semantic"},
        )
        uploaded = client.post(
            "/api/v1/documents/upload",
            files={"file": ("notes.txt", text, "text/plain")},
            data={"chunking_strategy": "fixed"},
        )
        health = client.get("/health")
        docs = client.get("/docs")
        missing = client.get("/api/v1/does-not-exist")

    assert invalid.status_code == 400
    assert invalid.json()["code"] == "invalid_file"
    assert empty.status_code == 400
    assert empty.json()["code"] == "empty_document"
    assert bad_strategy.status_code == 422
    assert uploaded.status_code == 200
    assert uploaded.json()["chunking_strategy"] == "fixed"
    assert repository.documents[0]["chunking_strategy"] == "fixed"
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert docs.status_code == 200
    assert missing.status_code == 404
    assert "traceback" not in invalid.text.lower()
