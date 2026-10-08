from app.core.exceptions import AppError
from app.schemas.chat import ChatResponse
from app.services.booking_service import BookingService
from app.services.memory_service import MemoryService
from app.services.rag_service import RAGService


class ChatService:
    """Decides whether a turn is interview booking or a document question, then records the turn."""

    def __init__(self, *, memory: MemoryService, booking: BookingService, rag: RAGService) -> None:
        self._memory = memory
        self._booking = booking
        self._rag = rag

    async def chat(self, session_id: str, message: str) -> ChatResponse:
        cleaned = message.strip()
        if not cleaned:
            raise AppError("Message cannot be empty.", status_code=422, code="validation_error")

        history = await self._memory.get_history(session_id)
        booking_response = await self._booking.handle_turn(
            session_id=session_id,
            message=cleaned,
            history=history,
        )
        response = booking_response
        if response is None:
            response = await self._rag.answer(session_id=session_id, message=cleaned, history=history)
        await self._memory.append(session_id, cleaned, response.answer)
        return response
