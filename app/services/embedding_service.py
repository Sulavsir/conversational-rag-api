import logging
from typing import Any, Protocol

from app.core.exceptions import ServiceUnavailableError

logger = logging.getLogger(__name__)


class EmbeddingModel(Protocol):
    def encode(self, sentences: list[str], *, normalize_embeddings: bool = True) -> Any: ...

    def get_embedding_dimension(self) -> int | None: ...


class EmbeddingService:
    """Loads one sentence-transformers model per process and reuses it.

    Pass a model instance in tests. Production loads the configured model on first use.
    """

    def __init__(self, model_name: str, model: EmbeddingModel | None = None) -> None:
        self._model_name = model_name
        self._model = model

    def embed_text(self, text: str) -> list[float]:
        vectors = self.embed_texts([text])
        return vectors[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            encoded = self._get_model().encode(list(texts), normalize_embeddings=True)
        except ServiceUnavailableError:
            raise
        except Exception as exc:
            logger.exception("Embedding generation failed")
            raise ServiceUnavailableError("Failed to generate embeddings.") from exc

        rows = _as_rows(encoded, expected=len(texts))
        return [[float(value) for value in row] for row in rows]

    @property
    def dimension(self) -> int:
        model = self._get_model()
        dimension = model.get_embedding_dimension()
        if not isinstance(dimension, int) or dimension <= 0:
            raise ServiceUnavailableError("The embedding model did not report a vector dimension.")
        return dimension

    def _get_model(self) -> EmbeddingModel:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name)
            except Exception as exc:
                logger.exception("Failed to load embedding model %s", self._model_name)
                raise ServiceUnavailableError("The embedding model could not be loaded.") from exc
        return self._model


def _as_rows(encoded: Any, *, expected: int) -> list[Any]:
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if expected == 1 and encoded and isinstance(encoded[0], (int, float)):
        return [encoded]
    if not isinstance(encoded, list) or len(encoded) != expected:
        raise ServiceUnavailableError("The embedding model returned an unexpected shape.")
    return encoded
