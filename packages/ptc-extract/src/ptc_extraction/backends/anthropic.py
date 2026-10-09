"""Anthropic Messages API backend. Requires ``ptc-extract[anthropic]``.

Use this when you want the library without an AWS account: it needs only an
``ANTHROPIC_API_KEY``. :class:`~ptc_extraction.backends.bedrock.BedrockBackend`
remains the reference implementation.
"""

import threading

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..exceptions import LLMBackendError, missing_extra
from .base import LLMBackend

try:
    import anthropic
except ImportError as _e:  # pragma: no cover - exercised in the no-extras CI job
    raise missing_extra("anthropic", "anthropic", _e) from _e

_RETRYABLE = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
    anthropic.RateLimitError,
)


class AnthropicBackend(LLMBackend):
    """Claude via the direct Anthropic API.

    The API key is read from ``ANTHROPIC_API_KEY`` unless passed explicitly.
    Temperature defaults to 0.0: extraction should be reproducible.
    """

    DEFAULT_MODEL = "claude-sonnet-4-5"

    def __init__(
        self,
        model_id: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout_seconds: int = 120,
        max_retries: int = 5,
    ):
        self._model_id = model_id or self.DEFAULT_MODEL
        self._temperature = temperature
        self._max_retries = max_retries
        # The SDK has its own retry loop; ours wraps it, so disable the inner one
        # to keep the backoff schedule in one place.
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout_seconds, max_retries=0)
        self._usage_lock = threading.Lock()
        self._input_tokens = 0
        self._output_tokens = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def usage(self) -> dict[str, int]:
        """Cumulative token usage across all calls on this backend instance."""
        with self._usage_lock:
            return {"input_tokens": self._input_tokens, "output_tokens": self._output_tokens}

    def _create(self, system: str, user: str, max_tokens: int):
        @retry(
            retry=retry_if_exception_type(_RETRYABLE),
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=2, min=2, max=60),
            reraise=True,
        )
        def _call():
            return self._client.messages.create(
                model=self._model_id,
                system=system,
                messages=[{"role": "user", "content": user}],
                max_tokens=max_tokens,
                temperature=self._temperature,
            )

        return _call()

    def complete(self, system: str, user: str, max_tokens: int = 4096) -> str:
        try:
            response = self._create(system, user, max_tokens)
        except anthropic.AnthropicError as e:
            raise LLMBackendError(f"Anthropic call failed: {e}") from e

        usage = getattr(response, "usage", None)
        if usage is not None:
            with self._usage_lock:
                self._input_tokens += getattr(usage, "input_tokens", 0) or 0
                self._output_tokens += getattr(usage, "output_tokens", 0) or 0

        try:
            return response.content[0].text
        except (AttributeError, IndexError) as e:
            raise LLMBackendError(f"Unexpected Anthropic response shape: {response!r}") from e
