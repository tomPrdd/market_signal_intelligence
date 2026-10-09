from abc import ABC, abstractmethod


class LLMBackend(ABC):
    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 4096) -> str:
        """Return the text content of the LLM completion."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Identifier of the model, used for audit (stored on PTC.extracted_by)."""
