"""Amazon Bedrock backend, via the Converse API. Requires ``ptc-extract[bedrock]``."""

import threading

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..exceptions import LLMBackendError, missing_extra
from .base import LLMBackend

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
except ImportError as _e:  # pragma: no cover - exercised in the no-extras CI job
    raise missing_extra("bedrock", "boto3", _e) from _e


class BedrockBackend(LLMBackend):
    """Bedrock LLM via Converse API. Default region: eu-west-3.

    Default model is Amazon Nova Pro — ~5x cheaper than Claude Sonnet on Bedrock
    with comparable structured-extraction quality for this task (validated
    manually against Sonnet on the PTC schema). Pass model_id="eu.anthropic.claude-sonnet-4-6"
    (or any other eu.* inference profile) to use a different model.
    """

    DEFAULT_MODEL = "eu.amazon.nova-pro-v1:0"
    DEFAULT_REGION = "eu-west-3"

    def __init__(
        self,
        model_id: str | None = None,
        region: str | None = None,
        temperature: float = 0.0,
        timeout_seconds: int = 120,
    ):
        self._model_id = model_id or self.DEFAULT_MODEL
        self._region = region or self.DEFAULT_REGION
        self._temperature = temperature
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=self._region,
            config=Config(read_timeout=timeout_seconds),
        )
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

    @retry(
        retry=retry_if_exception_type(ClientError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    def _converse(self, system: str, user: str, max_tokens: int) -> dict:
        return self._client.converse(
            modelId=self._model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={
                "maxTokens": max_tokens,
                "temperature": self._temperature,
            },
        )

    def complete(self, system: str, user: str, max_tokens: int = 4096) -> str:
        try:
            response = self._converse(system, user, max_tokens)
        except ClientError as e:
            raise LLMBackendError(f"Bedrock call failed: {e}") from e
        usage = response.get("usage", {})
        with self._usage_lock:
            self._input_tokens += usage.get("inputTokens", 0)
            self._output_tokens += usage.get("outputTokens", 0)
        try:
            return response["output"]["message"]["content"][0]["text"]
        except (KeyError, IndexError) as e:
            raise LLMBackendError(f"Unexpected Bedrock response shape: {response}") from e
