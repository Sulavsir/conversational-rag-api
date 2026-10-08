from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    detail: str
    code: str
    errors: list[dict[str, str]] | None = None


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
