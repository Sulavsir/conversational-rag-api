import pytest
from fakeredis import FakeAsyncRedis
from fastapi.testclient import TestClient

from app.api.dependencies import get_chat_service
from app.services.chat_service import ChatService
from app.services.memory_service import ConversationMessage, MemoryService
from app.services.rag_service import RAGService, build_search_query
from tests.fakes import ExplodingRAG, FakeEmbeddings, FakeLLM, FakeVectors, booking_json, sample_hit


def test_follow_up_query_includes_the_previous_user_turn() -> None:
    history = [
        ConversationMessage(role="user", content="What experience does the candidate have?"),
        ConversationMessage(role="assistant", content="The candidate has backend experience."),
    ]

    query = build_search_query("What about his machine learning project?", history)

    assert "What experience does the candidate have?" in query
    assert "machine learning project" in query
    assert build_search_query("What programming languages are listed?", history) == "What programming languages are listed?"


@pytest.mark.asyncio
async def test_rag_returns_answer_and_sources_without_inventing_prompt() -> None:
    vectors = FakeVectors()
    vectors.hits = [sample_hit()]
    llm = FakeLLM(answer="The candidate built a retrieval system.")
    rag = RAGService(embeddings=FakeEmbeddings(), vectors=vectors, llm=llm, top_k=5)  # type: ignore[arg-type]

    response = await rag.answer(session_id="session_123", message="What did the candidate build?", history=[])

    assert response.answer == "The candidate built a retrieval system."
    assert response.sources[0].filename == "resume.pdf"
    assert response.sources[0].chunk_id == 3
    system_prompt, messages = llm.calls[0]
    assert "Do not invent" in system_prompt
    assert "retrieval system" in system_prompt
    assert messages[-1] == {"role": "user", "content": "What did the candidate build?"}


@pytest.mark.asyncio
async def test_multi_turn_conversation_is_stored_and_sent_to_the_model() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    memory = MemoryService(redis, max_messages=4, ttl_seconds=60)
    llm = FakeLLM()
    vectors = FakeVectors()
    vectors.hits = [sample_hit()]
    rag = RAGService(embeddings=FakeEmbeddings(), vectors=vectors, llm=llm, top_k=5)  # type: ignore[arg-type]
    chat = ChatService(memory=memory, booking=_rag_only_booking(), rag=rag)

    first = await chat.chat("session_123", "What experience does the candidate have?")
    second = await chat.chat("session_123", "What about his machine learning project?")

    assert first.sources[0].chunk_id == 3
    _, second_messages = llm.calls[1]
    assert second_messages[0]["content"] == "What experience does the candidate have?"
    assert second_messages[1]["content"] == first.answer
    assert "machine learning project" in second_messages[-1]["content"]
    history = await memory.get_history("session_123")
    assert [item.role for item in history] == ["user", "assistant", "user", "assistant"]


@pytest.mark.asyncio
async def test_conversation_history_is_capped() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    memory = MemoryService(redis, max_messages=2, ttl_seconds=60)

    await memory.append("session_123", "first", "one")
    await memory.append("session_123", "second", "two")
    history = await memory.get_history("session_123")

    assert [item.content for item in history] == ["second", "two"]


def test_chat_route_returns_rag_answer(app) -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    memory = MemoryService(redis, max_messages=10, ttl_seconds=60)
    llm = FakeLLM(answer="The candidate has machine learning experience.")
    vectors = FakeVectors()
    vectors.hits = [sample_hit()]
    rag = RAGService(embeddings=FakeEmbeddings(), vectors=vectors, llm=llm, top_k=5)  # type: ignore[arg-type]
    service = ChatService(memory=memory, booking=_rag_only_booking(), rag=rag)
    app.dependency_overrides[get_chat_service] = lambda: service

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={"session_id": "session_123", "message": "What machine learning experience does the candidate have?"},
        )
        missing = client.post("/api/v1/chat", json={"session_id": "session_123"})

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "session_123"
    assert body["answer"] == "The candidate has machine learning experience."
    assert body["sources"][0]["filename"] == "resume.pdf"
    assert body["sources"][0]["chunk_id"] == 3
    assert missing.status_code == 422
    assert missing.json()["code"] == "validation_error"


@pytest.mark.asyncio
async def test_booking_intent_does_not_call_rag() -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    memory = MemoryService(redis, max_messages=10, ttl_seconds=60)
    llm = FakeLLM(replies=[booking_json()])
    from app.services.booking_service import BookingService
    from tests.fakes import FakeBookingRepository, FakeSession

    booking = BookingService(
        redis=redis,
        repository=FakeBookingRepository(),  # type: ignore[arg-type]
        session=FakeSession(),  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
        ttl_seconds=60,
    )
    chat = ChatService(memory=memory, booking=booking, rag=ExplodingRAG())  # type: ignore[arg-type]

    response = await chat.chat("session_123", "I want to book an interview.")

    assert response.answer == "Sure. What is your name?"
    assert response.sources == []
    assert "Document context" not in llm.calls[0][0]


def _rag_only_booking():
    class _Booking:
        async def handle_turn(self, **kwargs):
            return None

    return _Booking()
