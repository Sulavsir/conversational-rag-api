from datetime import date
from uuid import UUID

from app.schemas.chat import ChatResponse
from app.services.vector_service import ScoredChunk


class FakeSession:
    """Minimal stand-in for a database transaction."""

    def __init__(self) -> None:
        self.committed = False
        self.rolled_back = False
        self.pending: list[dict] = []
        self.committed_documents: list[dict] = []

    def add_pending(self, record: dict) -> None:
        self.pending.append(record)

    async def commit(self) -> None:
        self.committed = True
        self.committed_documents.extend(self.pending)
        self.pending.clear()

    async def rollback(self) -> None:
        self.rolled_back = True
        self.pending.clear()

    async def flush(self) -> None:
        return None


class FakeDocumentRepository:
    def __init__(self, session: FakeSession) -> None:
        self._session = session

    async def add_document_with_chunks(self, **kwargs):
        self._session.add_pending(kwargs)
        return kwargs

    @property
    def documents(self) -> list[dict]:
        return self._session.committed_documents


class FakeBookingRepository:
    def __init__(self) -> None:
        self.saved: list[dict] = []

    async def add(self, **kwargs):
        self.saved.append(kwargs)
        return kwargs


class FakeEmbeddings:
    def embed_text(self, text: str) -> list[float]:
        return [float(len(text)), 0.1, 0.2, 0.3]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


class FakeVectors:
    def __init__(self) -> None:
        self.records: list = []
        self.dimension: int | None = None
        self.deleted: list[str] = []
        self.hits: list[ScoredChunk] = []
        self.fail_upsert = False

    async def ensure_collection(self, dimension: int) -> None:
        self.dimension = dimension

    async def upsert(self, records) -> None:
        if self.fail_upsert:
            raise RuntimeError("qdrant down")
        self.records.extend(records)

    async def delete_by_document_id(self, document_id: str) -> None:
        self.deleted.append(document_id)
        self.records = [record for record in self.records if record.document_id != document_id]

    async def search(self, vector: list[float], top_k: int) -> list[ScoredChunk]:
        return self.hits[:top_k]


class FakeLLM:
    def __init__(self, replies: list[str] | None = None, answer: str = "The candidate has machine learning experience.") -> None:
        self.replies = list(replies or [])
        self.answer = answer
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    async def generate(self, system_prompt: str, messages: list[dict[str, str]]) -> str:
        self.calls.append((system_prompt, messages))
        if self.replies:
            return self.replies.pop(0)
        return self.answer


class ExplodingRAG:
    async def answer(self, **kwargs) -> ChatResponse:
        raise AssertionError("RAG should not run during interview booking")


def sample_hit() -> ScoredChunk:
    return ScoredChunk(
        document_id=UUID("2d9c8e3a-1b2c-4d5e-8f90-123456789abc"),
        filename="resume.pdf",
        chunk_index=3,
        text="Built a retrieval system using sentence embeddings and evaluated it on support tickets.",
        score=0.91,
    )


def booking_json(**overrides: object) -> str:
    payload = {
        "intent": "booking",
        "name": None,
        "email": None,
        "date": None,
        "time": None,
    }
    payload.update(overrides)
    import json

    return json.dumps(payload)


TODAY = date(2026, 10, 6)
