import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from signal_intelligence.embedding import _embed_text, embed_ptcs


def _fake_ptc(
    *,
    ticker: str = "ML.PA",
    idx: int = 0,
    direction: str = "precursor",
    polarity: int = -1,
) -> dict:
    """Return the kwargs dict that matches what load_pool should yield."""
    return {
        "mechanism": f"Mechanism text number {idx} long enough to pass validation",
        "raw_text": f"Raw text {idx}",
        "direction": direction,
        "polarity": polarity,
        "source_date": "2024-01-30",
        "event_date": None,
        "source_type": "management",
        "source_id": f"{ticker}-management-2024-01-30-doc{idx}",
        "extracted_by": "test",
        "extraction_version": "v1.1",
        "confidence": 0.9,
    }


def _make_ptc_object(data: dict):
    from ptc_extraction.schema import PTC

    return PTC(**data)


def _mock_bedrock_client(dim: int = 1024) -> MagicMock:
    client = MagicMock()
    body_mock = MagicMock()
    body_mock.read.return_value = json.dumps(
        {"embedding": list(np.random.rand(dim).astype(float))}
    ).encode()
    client.invoke_model.return_value = {"body": body_mock}
    return client


# ---------------------------------------------------------------------------
# _embed_text unit test
# ---------------------------------------------------------------------------


def test_embed_text_helper():
    client = _mock_bedrock_client(1024)
    vec = _embed_text(client, "some text")
    assert vec.shape == (1024,)
    assert vec.dtype == np.float32
    client.invoke_model.assert_called_once()
    call_kwargs = client.invoke_model.call_args.kwargs
    assert call_kwargs["modelId"] == "amazon.titan-embed-text-v2:0"


# ---------------------------------------------------------------------------
# embed_ptcs integration tests (Bedrock mocked)
# ---------------------------------------------------------------------------


def _make_pool_with_ptcs(tmp_path: Path, ticker: str, count: int) -> Path:
    from ptc_extraction.io import append_to_pool

    pool_path = tmp_path / "ptc_pool.jsonl"
    ptcs = [_make_ptc_object(_fake_ptc(ticker=ticker, idx=i)) for i in range(count)]
    append_to_pool(pool_path, ptcs)
    return pool_path


@pytest.fixture()
def mock_boto3():
    client = _mock_bedrock_client()
    with patch("signal_intelligence.embedding.boto3.client", return_value=client) as p:
        yield p, client


def test_embed_ptcs_creates_npy_and_index(tmp_path, mock_boto3):
    _, bedrock_client = mock_boto3
    pool_path = _make_pool_with_ptcs(tmp_path, "ML.PA", 3)
    output_dir = tmp_path / "interim"

    vectors, index_df = embed_ptcs("ML.PA", pool_path=pool_path, output_dir=output_dir)

    assert (output_dir / "ML.PA_embeddings.npy").exists()
    assert (output_dir / "ML.PA_embedding_index.csv").exists()
    assert vectors.shape == (3, 1024)
    assert vectors.dtype == np.float32
    assert len(index_df) == 3
    assert set(index_df.columns) >= {
        "content_hash",
        "source_id",
        "source_type",
        "direction",
        "polarity",
        "source_date",
        "mechanism",
    }
    assert bedrock_client.invoke_model.call_count == 3


def test_embed_ptcs_resume_skips_existing(tmp_path, mock_boto3):
    _, bedrock_client = mock_boto3
    pool_path = _make_pool_with_ptcs(tmp_path, "ML.PA", 2)
    output_dir = tmp_path / "interim"

    # First run: embeds both
    embed_ptcs("ML.PA", pool_path=pool_path, output_dir=output_dir)
    first_call_count = bedrock_client.invoke_model.call_count
    assert first_call_count == 2

    # Add one more PTC to the pool
    from ptc_extraction.io import append_to_pool

    extra_ptc = _make_ptc_object(_fake_ptc(ticker="ML.PA", idx=99))
    append_to_pool(pool_path, [extra_ptc])

    embed_ptcs("ML.PA", pool_path=pool_path, output_dir=output_dir)
    # Only the new PTC should have been embedded
    assert bedrock_client.invoke_model.call_count == first_call_count + 1


def test_embed_ptcs_refresh_re_embeds_all(tmp_path, mock_boto3):
    _, bedrock_client = mock_boto3
    pool_path = _make_pool_with_ptcs(tmp_path, "ML.PA", 2)
    output_dir = tmp_path / "interim"

    embed_ptcs("ML.PA", pool_path=pool_path, output_dir=output_dir)
    assert bedrock_client.invoke_model.call_count == 2

    embed_ptcs("ML.PA", pool_path=pool_path, output_dir=output_dir, refresh=True)
    assert bedrock_client.invoke_model.call_count == 4


def test_embed_ptcs_skips_on_bedrock_error(tmp_path, mock_boto3):
    _, bedrock_client = mock_boto3
    pool_path = _make_pool_with_ptcs(tmp_path, "ML.PA", 3)
    output_dir = tmp_path / "interim"

    # Calls 2, 3, 4 are the 3 retry attempts for ptc index 1 — all should fail
    call_count = [0]

    def sometimes_fail(*args, **kwargs):
        call_count[0] += 1
        if 2 <= call_count[0] <= 4:
            raise RuntimeError("Bedrock error")
        body_mock = MagicMock()
        body_mock.read.return_value = json.dumps(
            {"embedding": list(np.random.rand(1024).astype(float))}
        ).encode()
        return {"body": body_mock}

    bedrock_client.invoke_model.side_effect = sometimes_fail

    vectors, index_df = embed_ptcs("ML.PA", pool_path=pool_path, output_dir=output_dir)
    # PTC at index 1 gets skipped; ptcs 0 and 2 succeed
    assert len(index_df) == 2
    assert vectors.shape[0] == 2
