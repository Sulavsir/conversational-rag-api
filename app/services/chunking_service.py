from app.schemas.document import ChunkingStrategy

_SEPARATORS = ("\n\n", "\n", ". ", " ", "")


def chunk_text(
    text: str,
    strategy: ChunkingStrategy,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split text with the selected strategy.

    Recursive chunking prefers paragraph, line, sentence, and word boundaries.
    Fixed chunking uses character windows. Both keep overlap smaller than the window.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    if strategy is ChunkingStrategy.FIXED:
        return _fixed_chunks(normalized, chunk_size, chunk_overlap)
    if strategy is ChunkingStrategy.RECURSIVE:
        pieces = _split_into_pieces(normalized, chunk_size, _SEPARATORS)
        return _pack(pieces, chunk_size, chunk_overlap)
    raise ValueError(f"Unsupported chunking strategy: {strategy}")


def _fixed_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    start = 0
    while start < len(text):
        window = text[start : start + chunk_size]
        if window.strip():
            chunks.append(window)
        if start + chunk_size >= len(text):
            break
        start += step
    return chunks


def _split_into_pieces(text: str, chunk_size: int, separators: tuple[str, ...]) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    separator = ""
    next_separators: tuple[str, ...] = ()
    for index, candidate in enumerate(separators):
        if candidate == "":
            separator = ""
            next_separators = ()
            break
        if candidate in cleaned:
            separator = candidate
            next_separators = separators[index + 1 :]
            break

    if separator == "":
        return [
            cleaned[index : index + chunk_size].strip()
            for index in range(0, len(cleaned), chunk_size)
            if cleaned[index : index + chunk_size].strip()
        ]

    parts = [part.strip() for part in cleaned.split(separator) if part.strip()]
    pieces: list[str] = []
    for part in parts:
        if len(part) <= chunk_size:
            pieces.append(part)
        else:
            pieces.extend(_split_into_pieces(part, chunk_size, next_separators))
    return pieces


def _pack(pieces: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not piece:
            continue
        if not current:
            current = piece
            continue
        proposed = f"{current} {piece}"
        if len(proposed) <= chunk_size:
            current = proposed
            continue

        chunks.append(current)
        if chunk_overlap <= 0:
            current = piece
            continue

        tail = current[-chunk_overlap:]
        combined = f"{tail} {piece}".strip()
        current = combined if len(combined) <= chunk_size else piece

    if current:
        chunks.append(current)
    return chunks
