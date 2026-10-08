import json
import logging
import re
from collections.abc import Awaitable, Callable
from datetime import date, datetime, time, timedelta
from typing import Protocol

from dateutil.parser import ParserError
from dateutil.parser import parse as parse_datetime
from email_validator import EmailNotValidError, validate_email

from app.core.exceptions import ServiceUnavailableError
from app.schemas.booking import BookingDraft, BookingIntent
from app.schemas.chat import ChatResponse
from app.services.llm_service import LLMService
from app.services.memory_service import ConversationMessage

logger = logging.getLogger(__name__)

_DRAFT_PREFIX = "booking:"
_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z]+(?:[ '\-][A-Za-z][A-Za-z]+){0,3}$")
_BOOKING_REQUEST_PATTERN = re.compile(r"\b(book|booking|schedule|appointment)\b", re.IGNORECASE)
_CANCEL_PATTERN = re.compile(r"^(cancel|stop|never mind|nevermind)[.!]?$", re.IGNORECASE)
_TIME_PATTERN = re.compile(
    r"^(?:(?P<h12>\d{1,2})(?::(?P<m12>\d{2}))?(?::(?P<s12>\d{2}))?\s*(?P<ampm>am|pm)"
    r"|(?P<h24>[01]?\d|2[0-3]):(?P<m24>[0-5]\d)(?::(?P<s24>[0-5]\d))?"
    r"|(?P<word>noon|midnight))$",
    re.IGNORECASE,
)

_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
_MONTH_ALTERNATIVES = "|".join(sorted(_MONTHS, key=len, reverse=True))
_NATURAL_DATE_PATTERN = re.compile(
    rf"^(?P<month>{_MONTH_ALTERNATIVES})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s*(?P<year>\d{{4}}))?$",
    re.IGNORECASE,
)
_FIELD_ORDER = ("name", "email", "date", "time")
_FIRST_NAME_PROMPT = "Sure. What is your name?"
_PROMPTS = {
    "name": "What is your name?",
    "email": "What email should I use?",
    "date": "What date would you prefer?",
    "time": "What time?",
}
_INVALID = {
    "name": "I didn't catch a valid name. What is your name?",
    "email": "That email doesn't look valid. What email should I use?",
    "date": "That date doesn't look valid. What date would you prefer?",
    "past": "That date is in the past. What date would you prefer?",
    "time": "That time doesn't look valid. What time?",
}
_EXTRACTION_PROMPT = """You classify a user message and extract interview booking fields.
Return only a JSON object with this shape:
{"intent":"booking" or "rag","name":string or null,"email":string or null,"date":string or null,"time":string or null}

Use intent "booking" when the user wants to book, schedule, or reschedule an interview, or is supplying booking details.
Use intent "rag" for questions about documents or anything else.
Copy only values the user actually provided. Use null when a value was not provided.
Do not invent a name, email, date, or time.
"""


class BookingRedis(Protocol):
    async def get(self, key: str) -> str | bytes | None: ...

    async def set(self, key: str, value: str, ex: int | None = None) -> object: ...

    async def delete(self, key: str) -> object: ...


class BookingRepositoryPort(Protocol):
    def add(
        self,
        *,
        session_id: str,
        name: str,
        email: str,
        interview_date: date,
        interview_time: time,
    ) -> Awaitable[object]: ...


class SessionPort(Protocol):
    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...


class BookingService:
    """Collects interview details across turns, validates them, and stores a completed booking."""

    def __init__(
        self,
        *,
        redis: BookingRedis,
        repository: BookingRepositoryPort,
        session: SessionPort,
        llm: LLMService,
        ttl_seconds: int,
        clock: Callable[[], date] | None = None,
    ) -> None:
        self._redis = redis
        self._repository = repository
        self._session = session
        self._llm = llm
        self._ttl_seconds = ttl_seconds
        self._clock = clock or _today

    async def handle_turn(
        self,
        *,
        session_id: str,
        message: str,
        history: list[ConversationMessage],
    ) -> ChatResponse | None:
        draft = await self._load_draft(session_id)
        if draft.active and _is_cancel(message):
            await self._clear_draft(session_id)
            return _reply(session_id, "The interview booking has been cancelled.")

        intent = await self._extract(message, history, draft)
        if not draft.active and intent.intent != "booking":
            return None

        was_active = draft.active
        draft.active = True
        error = self._apply_fields(draft, intent, message, was_active=was_active)
        if error:
            await self._save_draft(session_id, draft)
            return _reply(session_id, error)

        missing = _next_missing(draft)
        if missing is not None:
            await self._save_draft(session_id, draft)
            prompt = _FIRST_NAME_PROMPT if missing == "name" and not was_active else _PROMPTS[missing]
            return _reply(session_id, prompt)

        await self._save_booking(session_id, draft)
        await self._clear_draft(session_id)
        interview_date = date.fromisoformat(draft.interview_date or "")
        interview_time = time.fromisoformat(draft.interview_time or "")
        confirmation = (
            f"Your interview is booked for {format_date(interview_date)} at {format_time(interview_time)}."
        )
        return _reply(session_id, confirmation)

    async def _extract(
        self,
        message: str,
        history: list[ConversationMessage],
        draft: BookingDraft,
    ) -> BookingIntent:
        system_prompt = _EXTRACTION_PROMPT
        if draft.active:
            system_prompt += (
                "\nCurrent draft:"
                f" name={draft.name or 'missing'},"
                f" email={draft.email or 'missing'},"
                f" date={draft.interview_date or 'missing'},"
                f" time={draft.interview_time or 'missing'}."
            )
        messages = [{"role": item.role, "content": item.content} for item in history[-8:]]
        messages.append({"role": "user", "content": message})
        try:
            raw = await self._llm.generate(system_prompt, messages)
            return parse_booking_intent(raw)
        except ServiceUnavailableError:
            if draft.active or _is_booking_request(message):
                logger.warning("Continuing booking collection without an LLM extraction")
                return BookingIntent(intent="booking")
            raise
        except Exception:
            logger.exception("Booking extraction could not be parsed")
            if draft.active or _is_booking_request(message):
                return BookingIntent(intent="booking")
            return BookingIntent(intent="rag")

    def _apply_fields(
        self,
        draft: BookingDraft,
        intent: BookingIntent,
        message: str,
        *,
        was_active: bool,
    ) -> str | None:
        proposals = {
            "name": _clean(intent.name),
            "email": _clean(intent.email),
            "date": _clean(intent.date),
            "time": _clean(intent.time),
        }
        missing = _next_missing(draft)
        if missing and proposals[missing] is None and _is_attempt(missing, message, was_active, self._clock()):
            proposals[missing] = message

        error: str | None = None
        for field in _FIELD_ORDER:
            raw = proposals[field]
            if raw is None:
                continue
            try:
                _assign(draft, field, raw, self._clock())
            except ValueError as exc:
                if field == "name" and _is_booking_request(message) and not parse_succeeds("name", message, self._clock()):
                    continue
                if error is None:
                    error = _INVALID["past"] if str(exc) == "past" else _INVALID[field]
        return error

    async def _save_booking(self, session_id: str, draft: BookingDraft) -> None:
        if not draft.name or not draft.email or not draft.interview_date or not draft.interview_time:
            raise ServiceUnavailableError("The booking could not be saved.")
        try:
            await self._repository.add(
                session_id=session_id,
                name=draft.name,
                email=draft.email,
                interview_date=date.fromisoformat(draft.interview_date),
                interview_time=time.fromisoformat(draft.interview_time),
            )
            await self._session.commit()
        except Exception as exc:
            await self._session.rollback()
            logger.exception("Failed to save interview booking")
            raise ServiceUnavailableError("The booking could not be saved.") from exc
        logger.info("Saved interview booking for session %s", session_id)

    async def _load_draft(self, session_id: str) -> BookingDraft:
        try:
            raw = await self._redis.get(f"{_DRAFT_PREFIX}{session_id}")
        except Exception as exc:
            logger.exception("Failed to read booking draft")
            raise ServiceUnavailableError("Redis is unavailable.") from exc
        if raw is None:
            return BookingDraft()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            return BookingDraft.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValueError):
            logger.warning("Discarding unreadable booking draft for session %s", session_id)
            return BookingDraft()

    async def _save_draft(self, session_id: str, draft: BookingDraft) -> None:
        try:
            await self._redis.set(
                f"{_DRAFT_PREFIX}{session_id}",
                draft.model_dump_json(),
                ex=self._ttl_seconds,
            )
        except Exception as exc:
            logger.exception("Failed to save booking draft")
            raise ServiceUnavailableError("Redis is unavailable.") from exc

    async def _clear_draft(self, session_id: str) -> None:
        try:
            await self._redis.delete(f"{_DRAFT_PREFIX}{session_id}")
        except Exception:
            logger.exception("Failed to clear booking draft for session %s", session_id)


def parse_booking_intent(raw: str) -> BookingIntent:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Booking extraction did not contain JSON")
    payload = json.loads(text[start : end + 1])
    return BookingIntent.model_validate(payload)


def parse_name(value: str) -> str:
    name = re.sub(r"\s+", " ", value).strip().strip(".").strip()
    name = re.sub(r"^(my name is|this is|i am|i'm)\s+", "", name, flags=re.IGNORECASE)
    if not _NAME_PATTERN.fullmatch(name):
        raise ValueError("invalid")
    return name


def parse_email(value: str) -> str:
    match = re.search(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", value)
    candidate = match.group(0) if match else value.strip()
    try:
        result = validate_email(candidate, check_deliverability=False)
    except EmailNotValidError as exc:
        raise ValueError("invalid") from exc
    return result.normalized.lower()


def parse_date(value: str, today: date) -> date:
    text = value.strip().rstrip(".").strip()
    lowered = text.lower()
    if lowered == "today":
        return today
    if lowered == "tomorrow":
        return today + timedelta(days=1)
    if lowered == "yesterday":
        raise ValueError("past")
    if _TIME_PATTERN.fullmatch(text):
        raise ValueError("invalid")

    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y"):
        try:
            return _reject_past(datetime.strptime(text, fmt).date(), today)
        except ValueError as exc:
            if str(exc) == "past":
                raise
            continue

    match = _NATURAL_DATE_PATTERN.fullmatch(text)
    if match:
        year_text = match.group("year")
        year = int(year_text) if year_text else None
        return _reject_past(_build_date(int(match.group("day")), _MONTHS[match.group("month").lower()], year, today), today)

    if not re.search(rf"\b(?:{_MONTH_ALTERNATIVES})\b", lowered) and not re.search(r"\d{1,4}[/-]\d{1,2}[/-]\d{1,4}", text):
        raise ValueError("invalid")
    try:
        parsed = parse_datetime(text, fuzzy=False, default=datetime(today.year, today.month, today.day)).date()
    except (ParserError, OverflowError, ValueError) as exc:
        raise ValueError("invalid") from exc
    if not re.search(r"\b(?:19|20)\d{2}\b", text):
        parsed = _roll_to_year(parsed, today)
    return _reject_past(parsed, today)


def parse_time(value: str) -> time:
    text = re.sub(r"\s+", " ", value).strip().rstrip(".")
    match = _TIME_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError("invalid")
    if match.group("word"):
        return time(0, 0) if match.group("word").lower() == "midnight" else time(12, 0)
    if match.group("ampm"):
        hour = int(match.group("h12"))
        minute = int(match.group("m12") or 0)
        if hour < 1 or hour > 12 or minute > 59:
            raise ValueError("invalid")
        suffix = match.group("ampm").lower()
        if suffix == "pm" and hour != 12:
            hour += 12
        if suffix == "am" and hour == 12:
            hour = 0
        return time(hour, minute)
    return time(int(match.group("h24")), int(match.group("m24")))


def format_date(value: date) -> str:
    return f"{value.strftime('%B')} {value.day}"


def format_time(value: time) -> str:
    hour = value.hour % 12 or 12
    suffix = "AM" if value.hour < 12 else "PM"
    if value.minute == 0:
        return f"{hour} {suffix}"
    return f"{hour}:{value.minute:02d} {suffix}"


def _assign(draft: BookingDraft, field: str, raw: str, today: date) -> None:
    if field == "name":
        draft.name = parse_name(raw)
    elif field == "email":
        draft.email = parse_email(raw)
    elif field == "date":
        draft.interview_date = parse_date(raw, today).isoformat()
    elif field == "time":
        draft.interview_time = parse_time(raw).isoformat()


def _next_missing(draft: BookingDraft) -> str | None:
    if not draft.name:
        return "name"
    if not draft.email:
        return "email"
    if not draft.interview_date:
        return "date"
    if not draft.interview_time:
        return "time"
    return None


def _is_attempt(field: str, message: str, was_active: bool, today: date) -> bool:
    if parse_succeeds(field, message, today):
        return True
    if not was_active:
        return False
    if field == "name":
        return not _is_booking_request(message)
    if field == "email":
        return "@" in message or len(message.split()) <= 4
    return len(message) <= 40


def parse_succeeds(field: str, message: str, today: date) -> bool:
    try:
        if field == "name":
            parse_name(message)
        elif field == "email":
            parse_email(message)
        elif field == "date":
            parse_date(message, today)
        else:
            parse_time(message)
    except ValueError:
        return False
    return True


def _build_date(day: int, month: int, year: int | None, today: date) -> date:
    chosen_year = today.year if year is None else year
    try:
        parsed = date(chosen_year, month, day)
    except ValueError as exc:
        raise ValueError("invalid") from exc
    if year is None and parsed < today:
        try:
            parsed = date(chosen_year + 1, month, day)
        except ValueError as exc:
            raise ValueError("invalid") from exc
    return parsed


def _roll_to_year(parsed: date, today: date) -> date:
    try:
        candidate = parsed.replace(year=today.year)
    except ValueError as exc:
        raise ValueError("invalid") from exc
    if candidate < today:
        try:
            candidate = candidate.replace(year=today.year + 1)
        except ValueError as exc:
            raise ValueError("invalid") from exc
    return candidate


def _reject_past(parsed: date, today: date) -> date:
    if parsed < today:
        raise ValueError("past")
    return parsed


def _is_booking_request(message: str) -> bool:
    return _BOOKING_REQUEST_PATTERN.search(message) is not None


def _is_cancel(message: str) -> bool:
    return _CANCEL_PATTERN.fullmatch(message.strip()) is not None


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped.lower() in {"null", "none", "n/a"}:
        return None
    return stripped


def _reply(session_id: str, answer: str) -> ChatResponse:
    return ChatResponse(session_id=session_id, answer=answer, sources=[])


def _today() -> date:
    return date.today()
