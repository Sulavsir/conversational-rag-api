from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_\-:.]+$",
        examples=["session_123"],
    )
    message: str = Field(
        min_length=1,
        max_length=8000,
        examples=["What machine learning experience does the candidate have?"],
    )


class Source(BaseModel):
    document_id: UUID
    filename: str
    chunk_id: int = Field(description="Chunk index within the source document, starting at 0.")


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[Source]

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "session_id": "session_123",
                    "answer": "The candidate built a retrieval system with sentence embeddings.",
                    "sources": [
                        {
                            "document_id": "2d9c8e3a-1b2c-4d5e-8f90-123456789abc",
                            "filename": "resume.pdf",
                            "chunk_id": 3,
                        }
                    ],
                }
            ]
        }
    )
