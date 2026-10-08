from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ChunkingStrategy(StrEnum):
    RECURSIVE = "recursive"
    FIXED = "fixed"


class DocumentUploadResponse(BaseModel):
    document_id: UUID
    filename: str
    file_type: str = Field(description="pdf or txt")
    chunking_strategy: ChunkingStrategy
    chunk_count: int = Field(ge=1)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "document_id": "2d9c8e3a-1b2c-4d5e-8f90-123456789abc",
                    "filename": "resume.pdf",
                    "file_type": "pdf",
                    "chunking_strategy": "recursive",
                    "chunk_count": 4,
                }
            ]
        }
    )
