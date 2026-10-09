"""Exception types, and the helper that reports a missing optional dependency."""


class PTCExtractionError(Exception):
    """Base exception for the package."""


class LLMBackendError(PTCExtractionError):
    """Raised when the LLM backend fails after retries."""


class InvalidPTCError(PTCExtractionError):
    """Raised when validation cannot rescue a raw PTC dict."""


def missing_extra(extra: str, package: str, cause: Exception | None = None) -> ImportError:
    """Build the ImportError raised when an optional dependency is absent.

    Every optional code path uses this, so the message always tells the user
    exactly which extra to install rather than leaking a bare ModuleNotFoundError.
    """
    err = ImportError(
        f"{package!r} is required for this feature but is not installed. "
        f"Install it with:  pip install "
        f"'ptc-extract[{extra}] @ git+https://github.com/tomPrdd/ptc-extract'"
    )
    err.__cause__ = cause
    return err
