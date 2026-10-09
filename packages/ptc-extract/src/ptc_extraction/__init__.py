"""ptc-extract — typed, signed causal claim extraction from text with an LLM.

A PTC (Point To Correlate) is one atomic causal claim: a mechanism, the
direction of its effect in time, the sign of that effect, when it was published,
and the passage it came from.

Backends live behind optional extras. ``BedrockBackend`` and
``AnthropicBackend`` are re-exported here for convenience but import lazily, so
``import ptc_extraction`` works with no extras installed and only fails — with a
message naming the extra — when you actually construct one.
"""

from typing import TYPE_CHECKING

from .chunking import Chunk, split_document
from .exceptions import InvalidPTCError, LLMBackendError, PTCExtractionError
from .extractor import PTCExtractor
from .io import append_to_pool, dedup_pool, existing_source_ids, load_pool
from .prompts import (
    FINANCE_PROMPT_VERSION,
    FINANCE_SYSTEM_PROMPT_V1_1,
    FINANCE_USER_PROMPT_TEMPLATE,
)
from .schema import PTC, Direction, Polarity, SourceType
from .validation import filter_valid_ptcs, validate_raw_ptc_dict

if TYPE_CHECKING:  # pragma: no cover
    from .backends.anthropic import AnthropicBackend
    from .backends.base import LLMBackend
    from .backends.bedrock import BedrockBackend

__version__ = "0.1.0"

__all__ = [
    "FINANCE_PROMPT_VERSION",
    "FINANCE_SYSTEM_PROMPT_V1_1",
    "FINANCE_USER_PROMPT_TEMPLATE",
    "PTC",
    "AnthropicBackend",
    "BedrockBackend",
    "Chunk",
    "Direction",
    "InvalidPTCError",
    "LLMBackend",
    "LLMBackendError",
    "PTCExtractionError",
    "PTCExtractor",
    "Polarity",
    "SourceType",
    "append_to_pool",
    "dedup_pool",
    "existing_source_ids",
    "filter_valid_ptcs",
    "load_pool",
    "split_document",
    "validate_raw_ptc_dict",
]

_LAZY = {
    "AnthropicBackend": ".backends.anthropic",
    "BedrockBackend": ".backends.bedrock",
    "LLMBackend": ".backends.base",
}


def __getattr__(name: str):
    """Import backends on first access, so the base install needs no vendor SDK."""
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_path, __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
