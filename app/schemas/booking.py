from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BookingIntent(BaseModel):
    """Structured booking extraction. Values are proposals and must be validated in code."""

    model_config = ConfigDict(extra="ignore")

    intent: Literal["booking", "rag"] = "rag"
    name: str | None = None
    email: str | None = None
    date: str | None = None
    time: str | None = None


class BookingDraft(BaseModel):
    """Partial interview booking kept in Redis until every field is valid."""

    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    email: str | None = None
    interview_date: str | None = Field(default=None, description="ISO calendar date, YYYY-MM-DD.")
    interview_time: str | None = Field(default=None, description="ISO time, HH:MM:SS.")
    active: bool = False
