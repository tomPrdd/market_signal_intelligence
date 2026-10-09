"""LLM backends.

Only :class:`LLMBackend` imports eagerly — it is pure stdlib. The concrete
backends pull in a vendor SDK, so they are imported on first access and raise a
clear :class:`ImportError` naming the extra to install.
"""

from typing import TYPE_CHECKING

from .base import LLMBackend

if TYPE_CHECKING:  # pragma: no cover
    from .anthropic import AnthropicBackend
    from .bedrock import BedrockBackend

__all__ = ["AnthropicBackend", "BedrockBackend", "LLMBackend"]

_LAZY = {"AnthropicBackend": ".anthropic", "BedrockBackend": ".bedrock"}


def __getattr__(name: str):
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_path, __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
