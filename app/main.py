import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.exception_handlers import register_exception_handlers
from app.api.routes.chat_routes import router as chat_router
from app.api.routes.document_routes import router as document_router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.db.postgres import create_tables, dispose_engine, init_engine
from app.db.qdrant import close_qdrant, get_qdrant, init_qdrant
from app.db.redis import close_redis, get_redis, init_redis
from app.schemas.common import HealthResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    if settings.skip_external_startup:
        yield
        return

    logger.info("Connecting to PostgreSQL, Redis, and Qdrant")
    try:
        init_engine(settings.database_url)
        await create_tables()
        init_redis(settings.redis_url)
        await get_redis().ping()
        init_qdrant(settings.qdrant_url, settings.qdrant_timeout_seconds)
        await get_qdrant().get_collections()
    except Exception:
        logger.exception("Startup dependency check failed")
        await _shutdown()
        raise

    try:
        yield
    finally:
        await _shutdown()


async def _shutdown() -> None:
    await close_redis()
    await close_qdrant()
    await dispose_engine()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        summary="Document ingestion and conversational retrieval for interview screening.",
        description=(
            "Upload PDF or text documents, then ask questions about them. "
            "The same chat endpoint can book an interview across multiple turns."
        ),
        lifespan=lifespan,
    )
    register_exception_handlers(app)
    app.include_router(document_router, prefix=settings.api_v1_prefix)
    app.include_router(chat_router, prefix=settings.api_v1_prefix)

    @app.get("/health", tags=["Health"], response_model=HealthResponse, summary="Liveness check")
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    return app


app = create_app()
