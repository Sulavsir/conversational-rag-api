from pathlib import Path

import pymupdf

from app.core.exceptions import EmptyDocumentError, InvalidFileError

_SUPPORTED_EXTENSIONS = {".pdf": "pdf", ".txt": "txt"}


def safe_filename(filename: str | None) -> str:
    raw = (filename or "").replace("\\", "/").strip()
    name = raw.rsplit("/", 1)[-1].strip()
    if not name or name in {".", ".."}:
        raise InvalidFileError("A filename is required.")
    if len(name) > 255:
        raise InvalidFileError("The filename is too long.")
    return name


def file_type_for(filename: str) -> str:
    extension = Path(filename).suffix.lower()
    file_type = _SUPPORTED_EXTENSIONS.get(extension)
    if file_type is None:
        allowed = ", ".join(sorted(_SUPPORTED_EXTENSIONS))
        raise InvalidFileError(f"Unsupported file type. Upload a {allowed} file.")
    return file_type


def extract_text(filename: str, content: bytes) -> tuple[str, str]:
    """Return (file_type, text). Raises client-safe errors for bad uploads."""
    safe_name = safe_filename(filename)
    file_type = file_type_for(safe_name)
    if not content:
        raise EmptyDocumentError("The uploaded file is empty.")

    if file_type == "txt":
        text = _read_txt(content)
    else:
        text = _read_pdf(content)

    if not text.strip():
        raise EmptyDocumentError()
    return file_type, text.strip()


def _read_txt(content: bytes) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InvalidFileError("Text files must be valid UTF-8.") from exc


def _read_pdf(content: bytes) -> str:
    try:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            pages = [page.get_text("text") for page in document]
    except Exception as exc:
        raise InvalidFileError("The PDF could not be read. It may be corrupted.") from exc
    return "\n".join(pages)
