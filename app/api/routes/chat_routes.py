from fastapi import APIRouter, Depends

from app.api.dependencies import get_chat_service
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.common import ErrorResponse
from app.services.chat_service import ChatService

router = APIRouter(tags=["Chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a question or book an interview",
    description=(
        "Continues a conversation identified by session_id. Document questions use retrieval "
        "over ingested chunks. Interview booking is collected across turns and saved once "
        "name, email, date, and time are valid."
    ),
    responses={
        422: {"model": ErrorResponse, "description": "Invalid request body"},
        503: {"model": ErrorResponse, "description": "Redis, Qdrant, PostgreSQL, or the LLM is unavailable"},
    },
)
async def chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    return await service.chat(session_id=request.session_id, message=request.message)
