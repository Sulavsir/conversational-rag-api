from datetime import date, time

import pytest
from fakeredis import FakeAsyncRedis

from app.services.booking_service import BookingService, parse_date, parse_email, parse_time
from tests.fakes import TODAY, FakeBookingRepository, FakeLLM, FakeSession, booking_json


def _service(
    replies: list[str] | None = None,
) -> tuple[BookingService, FakeBookingRepository, FakeSession]:
    repository = FakeBookingRepository()
    session = FakeSession()
    service = BookingService(
        redis=FakeAsyncRedis(decode_responses=True),
        repository=repository,  # type: ignore[arg-type]
        session=session,  # type: ignore[arg-type]
        llm=FakeLLM(replies=replies),  # type: ignore[arg-type]
        ttl_seconds=60,
        clock=lambda: TODAY,
    )
    return service, repository, session


@pytest.mark.asyncio
async def test_booking_is_collected_across_turns_and_saved_once() -> None:
    service, repository, session = _service(replies=[booking_json()] * 5)
    turns = [
        ("I want to book an interview.", "Sure. What is your name?"),
        ("Sulav Karki.", "What email should I use?"),
        ("sulav@example.com", "What date would you prefer?"),
        ("October 15.", "What time?"),
        ("2 PM.", "Your interview is booked for October 15 at 2 PM."),
    ]

    for message, expected in turns:
        response = await service.handle_turn(session_id="session_123", message=message, history=[])
        assert response is not None
        assert response.answer == expected

    assert session.committed is True
    assert repository.saved == [
        {
            "session_id": "session_123",
            "name": "Sulav Karki",
            "email": "sulav@example.com",
            "interview_date": date(2026, 10, 15),
            "interview_time": time(14, 0),
        }
    ]


@pytest.mark.asyncio
async def test_booking_accepts_a_complete_validated_extraction() -> None:
    service, repository, session = _service(
        replies=[
            booking_json(
                name="Sulav Karki",
                email="sulav@example.com",
                date="October 15",
                time="2 PM",
            )
        ]
    )

    response = await service.handle_turn(session_id="session_123", message="Please book the interview.", history=[])

    assert response is not None
    assert response.answer == "Your interview is booked for October 15 at 2 PM."
    assert session.committed is True
    assert len(repository.saved) == 1


@pytest.mark.asyncio
async def test_missing_booking_fields_are_requested_and_not_saved() -> None:
    service, repository, session = _service(replies=[booking_json(name="Sulav Karki")])

    response = await service.handle_turn(session_id="session_123", message="I want to book an interview.", history=[])

    assert response is not None
    assert response.answer == "What email should I use?"
    assert repository.saved == []
    assert session.committed is False


@pytest.mark.asyncio
async def test_invalid_email_is_rejected() -> None:
    service, repository, _ = _service(
        replies=[
            booking_json(name="Sulav Karki", email="not-an-email"),
            booking_json(),
        ]
    )

    invalid = await service.handle_turn(session_id="session_123", message="Book an interview.", history=[])
    corrected = await service.handle_turn(session_id="session_123", message="sulav@example.com", history=[])

    assert invalid is not None
    assert invalid.answer == "That email doesn't look valid. What email should I use?"
    assert corrected is not None
    assert corrected.answer == "What date would you prefer?"
    assert repository.saved == []


@pytest.mark.asyncio
async def test_invalid_date_and_time_are_rejected() -> None:
    service, repository, _ = _service(
        replies=[
            booking_json(name="Sulav Karki", email="sulav@example.com", date="January 1, 2020"),
            booking_json(),
            booking_json(),
        ]
    )

    past = await service.handle_turn(session_id="session_123", message="Book an interview.", history=[])
    assert past is not None
    assert past.answer == "That date is in the past. What date would you prefer?"

    asked_for_time = await service.handle_turn(session_id="session_123", message="October 15", history=[])
    assert asked_for_time is not None
    assert asked_for_time.answer == "What time?"

    bad_time = await service.handle_turn(session_id="session_123", message="25:99", history=[])
    assert bad_time is not None
    assert bad_time.answer == "That time doesn't look valid. What time?"
    assert repository.saved == []


@pytest.mark.asyncio
async def test_document_question_is_not_a_booking() -> None:
    service, repository, _ = _service(replies=['{"intent":"rag","name":null,"email":null,"date":null,"time":null}'])

    response = await service.handle_turn(
        session_id="session_123",
        message="What machine learning experience does the candidate have?",
        history=[],
    )

    assert response is None
    assert repository.saved == []


def test_field_parsers() -> None:
    assert parse_email("Email me at Sulav@Example.com.") == "sulav@example.com"
    assert parse_date("October 15", TODAY) == date(2026, 10, 15)
    assert parse_date("October 1", TODAY) == date(2027, 10, 1)
    assert parse_time("2 PM") == time(14, 0)
    assert parse_time("14:30") == time(14, 30)

    with pytest.raises(ValueError):
        parse_email("not-an-email")
    with pytest.raises(ValueError, match="past"):
        parse_date("January 1, 2020", TODAY)
    with pytest.raises(ValueError):
        parse_time("noonish")
