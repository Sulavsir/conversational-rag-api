class AppError(Exception):
    """An error whose message is safe to return to an API client."""

    def __init__(self, message: str, *, status_code: int = 500, code: str = "internal_error") -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


class InvalidFileError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=400, code="invalid_file")


class EmptyDocumentError(AppError):
    def __init__(self, message: str = "The uploaded document has no extractable text.") -> None:
        super().__init__(message, status_code=400, code="empty_document")


class PayloadTooLargeError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=413, code="payload_too_large")


class ServiceUnavailableError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=503, code="service_unavailable")
