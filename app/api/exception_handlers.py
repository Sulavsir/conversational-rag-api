import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.exceptions import AppError

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.error("%s %s failed: %s", request.method, request.url.path, exc.message)
        else:
            logger.info("%s %s rejected: %s", request.method, request.url.path, exc.code)
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message, "code": exc.code},
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {
                "field": ".".join(str(part) for part in error.get("loc", []) if part not in {"body", "query"}),
                "message": str(error.get("msg", "Invalid value")),
            }
            for error in exc.errors()
        ]
        logger.info("%s %s validation failed", request.method, request.url.path)
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Request validation failed.",
                "code": "validation_error",
                "errors": errors,
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected error occurred.", "code": "internal_error"},
        )
