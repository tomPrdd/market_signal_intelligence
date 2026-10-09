"""Backend contracts, optional-dependency guards, and the lazy base import."""

import builtins
import importlib
import sys
from pathlib import Path

import pytest

import ptc_extraction
from ptc_extraction.backends.base import LLMBackend
from ptc_extraction.exceptions import LLMBackendError, missing_extra

anthropic = pytest.importorskip("anthropic", reason="anthropic extra not installed")
boto3 = pytest.importorskip("boto3", reason="bedrock extra not installed")

from ptc_extraction.backends.anthropic import AnthropicBackend  # noqa: E402
from ptc_extraction.backends.bedrock import BedrockBackend  # noqa: E402

# --- the base package must not need any extra ------------------------------------


def test_base_import_pulls_in_no_vendor_sdk():
    """Importing the package must not import boto3, anthropic or huggingface_hub.

    Runs in a subprocess because the parent process has already imported those
    SDKs for other tests. The package source is put on PYTHONPATH explicitly so
    the test measures import side effects rather than whether an editable
    install happens to be resolving — those are different things, and on macOS
    a .pth file that acquires UF_HIDDEN is silently skipped by Python 3.12+.
    """
    import os
    import subprocess

    code = (
        "import sys; import ptc_extraction; "
        "leaked = [m for m in ('boto3','botocore','anthropic','huggingface_hub','pyarrow') "
        "if m in sys.modules]; "
        "assert not leaked, leaked; print('clean')"
    )
    src = str(Path(__file__).resolve().parent.parent / "src")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([src, os.environ.get("PYTHONPATH", "")])}

    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False, env=env
    )
    assert out.returncode == 0, out.stderr
    assert "clean" in out.stdout


def test_backends_are_reachable_from_the_top_level_namespace():
    assert ptc_extraction.BedrockBackend is BedrockBackend
    assert ptc_extraction.AnthropicBackend is AnthropicBackend
    assert ptc_extraction.LLMBackend is LLMBackend


def test_unknown_top_level_attribute_still_raises_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'Nonsense'"):
        _ = ptc_extraction.Nonsense


def test_dir_lists_the_public_api():
    assert "BedrockBackend" in dir(ptc_extraction)
    assert "PTCExtractor" in dir(ptc_extraction)


def test_backends_subpackage_resolves_lazily_too():
    from ptc_extraction import backends

    assert backends.BedrockBackend is BedrockBackend
    assert backends.AnthropicBackend is AnthropicBackend
    assert backends.LLMBackend is LLMBackend
    assert dir(backends) == ["AnthropicBackend", "BedrockBackend", "LLMBackend"]


def test_backends_subpackage_rejects_unknown_attributes():
    from ptc_extraction import backends

    with pytest.raises(AttributeError, match="no attribute 'Nope'"):
        _ = backends.Nope


# --- missing-extra guards ---------------------------------------------------------


def test_missing_extra_message_names_the_install_command():
    err = missing_extra("hub", "huggingface_hub")
    assert "ptc-extract[hub]" in str(err)
    assert "huggingface_hub" in str(err)
    assert isinstance(err, ImportError)


@pytest.mark.parametrize(
    ("module", "blocked", "extra"),
    [
        ("ptc_extraction.backends.bedrock", "boto3", "bedrock"),
        ("ptc_extraction.backends.anthropic", "anthropic", "anthropic"),
    ],
)
def test_backend_import_without_its_sdk_names_the_extra(monkeypatch, module, blocked, extra):
    real_import = builtins.__import__

    def _blocking_import(name, *args, **kwargs):
        if name == blocked or name.startswith(f"{blocked}."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocking_import)
    for name in [m for m in sys.modules if m.startswith(("ptc_extraction.backends", blocked))]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    with pytest.raises(ImportError, match=rf"ptc-extract\[{extra}\]"):
        importlib.import_module(module)


def test_hub_module_imports_without_its_extras(monkeypatch):
    """hub.py must be importable bare; only its functions may raise."""
    from ptc_extraction import hub

    def _boom():
        raise missing_extra("hub", "huggingface_hub")

    monkeypatch.setattr(hub, "_require_hub", _boom)
    with pytest.raises(ImportError, match=r"ptc-extract\[hub\]"):
        hub.load_pool_from_hub("acme/pool")


# --- shared backend contract ------------------------------------------------------


@pytest.fixture(params=["bedrock", "anthropic"])
def backend_case(request, monkeypatch):
    """Each backend, with its network client replaced by a recording double."""
    if request.param == "bedrock":
        backend = BedrockBackend.__new__(BedrockBackend)
        backend._model_id = "eu.amazon.nova-pro-v1:0"
        backend._temperature = 0.0
        backend._usage_lock = __import__("threading").Lock()
        backend._input_tokens = backend._output_tokens = 0

        class _Client:
            def converse(self, **kwargs):
                self.kwargs = kwargs
                return {
                    "output": {"message": {"content": [{"text": "[]"}]}},
                    "usage": {"inputTokens": 120, "outputTokens": 34},
                }

        backend._client = _Client()
        return backend, lambda: backend._client.kwargs

    backend = AnthropicBackend.__new__(AnthropicBackend)
    backend._model_id = "claude-sonnet-4-5"
    backend._temperature = 0.0
    backend._max_retries = 5
    backend._usage_lock = __import__("threading").Lock()
    backend._input_tokens = backend._output_tokens = 0

    class _Messages:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return type(
                "R",
                (),
                {
                    "content": [type("B", (), {"text": "[]"})()],
                    "usage": type("U", (), {"input_tokens": 120, "output_tokens": 34})(),
                },
            )()

    class _Client:
        messages = _Messages()

    backend._client = _Client()
    return backend, lambda: backend._client.messages.kwargs


def test_complete_returns_the_text_content(backend_case):
    backend, _ = backend_case
    assert backend.complete("system", "user") == "[]"


def test_usage_accumulates_across_calls(backend_case):
    backend, _ = backend_case
    assert backend.usage == {"input_tokens": 0, "output_tokens": 0}
    backend.complete("system", "user")
    backend.complete("system", "user")
    assert backend.usage == {"input_tokens": 240, "output_tokens": 68}


def test_both_backends_satisfy_the_abc(backend_case):
    backend, _ = backend_case
    assert isinstance(backend, LLMBackend)
    assert isinstance(backend.model_id, str)


def test_temperature_is_zero_by_default(backend_case):
    backend, sent = backend_case
    backend.complete("system", "user")
    kwargs = sent()
    temp = kwargs.get("temperature", kwargs.get("inferenceConfig", {}).get("temperature"))
    assert temp == 0.0


def test_max_tokens_is_forwarded(backend_case):
    backend, sent = backend_case
    backend.complete("system", "user", max_tokens=777)
    kwargs = sent()
    got = kwargs.get("max_tokens", kwargs.get("inferenceConfig", {}).get("maxTokens"))
    assert got == 777


# --- error translation -------------------------------------------------------------


def test_anthropic_malformed_response_raises_llm_backend_error():
    backend = AnthropicBackend.__new__(AnthropicBackend)
    backend._model_id = "claude-sonnet-4-5"
    backend._temperature = 0.0
    backend._max_retries = 1
    backend._usage_lock = __import__("threading").Lock()
    backend._input_tokens = backend._output_tokens = 0

    class _Client:
        class messages:
            @staticmethod
            def create(**kwargs):
                return type("R", (), {"content": []})()

    backend._client = _Client()
    with pytest.raises(LLMBackendError, match="Unexpected Anthropic response shape"):
        backend.complete("system", "user")


def test_bedrock_malformed_response_raises_llm_backend_error():
    backend = BedrockBackend.__new__(BedrockBackend)
    backend._model_id = "eu.amazon.nova-pro-v1:0"
    backend._temperature = 0.0
    backend._usage_lock = __import__("threading").Lock()
    backend._input_tokens = backend._output_tokens = 0

    class _Client:
        @staticmethod
        def converse(**kwargs):
            return {"output": {}}

    backend._client = _Client()
    with pytest.raises(LLMBackendError, match="Unexpected Bedrock response shape"):
        backend.complete("system", "user")


def test_default_models_are_pinned():
    assert BedrockBackend.DEFAULT_MODEL == "eu.amazon.nova-pro-v1:0"
    assert AnthropicBackend.DEFAULT_MODEL == "claude-sonnet-4-5"
