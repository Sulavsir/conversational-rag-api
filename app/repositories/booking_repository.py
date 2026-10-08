import uuid
from datetime import date, time

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.booking import Booking


class BookingRepository:
    """PostgreSQL access for completed interview bookings."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(
        self,
        *,
        session_id: str,
        name: str,
        email: str,
        interview_date: date,
        interview_time: time,
    ) -> Booking:
        booking = Booking(
            id=uuid.uuid4(),
            session_id=session_id,
            name=name,
            email=email,
            interview_date=interview_date,
            interview_time=interview_time,
        )
        self._session.add(booking)
        await self._session.flush()
        return booking
