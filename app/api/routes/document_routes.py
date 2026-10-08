from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.dependencies import get_ingestion_service
from app.schemas.common import ErrorResponse
from app.schemas.document import ChunkingStrategy, DocumentUploadResponse
from app.services.ingestion_service import IngestionService

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    summary="Upload a PDF or text document",
    description=(
        "Extracts text, chunks it with the selected strategy, embeds the chunks, "
        "stores vectors in Qdrant, and stores document metadata in PostgreSQL."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Unsupported, empty, or unreadable file"},
        413: {"model": ErrorResponse, "description": "File exceeds the upload limit"},
        422: {"model": ErrorResponse, "description": "Invalid form fields"},
        503: {"model": ErrorResponse, "description": "Embedding, Qdrant, or PostgreSQL unavailable"},
    },
)
async def upload_document(
    file: Annotated[UploadFile, File(description="A .pdf or .txt document.")],
    chunking_strategy: Annotated[
        ChunkingStrategy,
        Form(description="How to split the document. Allowed values: recursive, fixed."),
    ],
    service: IngestionService = Depends(get_ingestion_service),
) -> DocumentUploadResponse:
    content = await file.read(service.max_upload_bytes + 1)
    return await service.ingest(
        filename=file.filename,
        content=content,
        chunking_strategy=chunking_strategy,
    )
