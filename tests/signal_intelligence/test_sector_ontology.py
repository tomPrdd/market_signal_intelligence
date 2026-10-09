"""Tests for signal_intelligence.sector_ontology (Phase 0).

All Bedrock and Exa calls are mocked. No network access.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from signal_intelligence.sector_ontology import (
    build_sector_vectors,
    discover_peers,
    embed_peer_ptcs,
)

# ---------------------------------------------------------------------------
# discover_peers
# ---------------------------------------------------------------------------


def _mock_bedrock_converse(peers: list[str]) -> MagicMock:
    client = MagicMock()
    client.converse.return_value = {
        "output": {"message": {"content": [{"text": json.dumps(peers)}]}}
    }
    return client


def test_discover_peers_returns_list():
    expected = ["MT.AS", "CGEM.PA", "5108.T"]
    with patch(
        "signal_intelligence.sector_ontology.boto3.client",
        return_value=_mock_bedrock_converse(expected),
    ):
        result = discover_peers("ML.PA", "Michelin", "Tires", n_peers=3)
    assert result == expected


def test_discover_peers_bedrock_error_returns_empty():
    client = MagicMock()
    client.converse.side_effect = RuntimeError("Bedrock error")
    with patch("signal_intelligence.sector_ontology.boto3.client", return_value=client):
        result = discover_peers("ML.PA", "Michelin", "Tires")
    assert result == []


def test_discover_peers_invalid_json_returns_empty():
    client = MagicMock()
    client.converse.return_value = {
        "output": {"message": {"content": [{"text": "not valid json"}]}}
    }
    with patch("signal_intelligence.sector_ontology.boto3.client", return_value=client):
        result = discover_peers("ML.PA", "Michelin", "Tires")
    assert result == []


# ---------------------------------------------------------------------------
# build_sector_vectors
# ---------------------------------------------------------------------------


def _unit_vectors(n: int, dim: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1, norms)


def test_build_sector_vectors_returns_centroids():
    # 9 vectors in 3 tight clusters of 3
    base_a = _unit_vectors(1, dim=16, seed=10)[0]
    base_b = _unit_vectors(1, dim=16, seed=20)[0]
    base_c = _unit_vectors(1, dim=16, seed=30)[0]
    rng = np.random.default_rng(0)
    vecs = np.vstack(
        [
            base_a + rng.standard_normal((3, 16)).astype(np.float32) * 0.01,
            base_b + rng.standard_normal((3, 16)).astype(np.float32) * 0.01,
            base_c + rng.standard_normal((3, 16)).astype(np.float32) * 0.01,
        ]
    )
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms
    centroids = build_sector_vectors(vecs, min_cluster_size=3)
    assert centroids.ndim == 2
    assert centroids.shape[1] == 16
    assert len(centroids) >= 1


def test_build_sector_vectors_too_few_returns_empty():
    vecs = _unit_vectors(2, dim=16)
    result = build_sector_vectors(vecs, min_cluster_size=3)
    assert result.shape[0] == 0


def test_build_sector_vectors_all_noise():
    # Random orthogonal-ish vectors — HDBSCAN may assign all as noise
    vecs = _unit_vectors(5, dim=16, seed=99)
    result = build_sector_vectors(vecs, min_cluster_size=4)
    # Either 0 centroids (all noise) or some centroids — both valid
    assert result.ndim == 2


# ---------------------------------------------------------------------------
# embed_peer_ptcs (Bedrock mocked)
# ---------------------------------------------------------------------------


def _make_peer_pool(tmp_path: Path, peers: list[str]) -> Path:
    from datetime import date

    from ptc_extraction.io import append_to_pool
    from ptc_extraction.schema import PTC

    pool_path = tmp_path / "pool.jsonl"
    ptcs = []
    for i, peer in enumerate(peers):
        ptcs.append(
            PTC(
                mechanism=f"Mechanism for peer {peer} number {i} long enough",
                raw_text="raw text",
                direction="precursor",
                polarity=-1,
                source_date=date(2024, 1, 1),
                source_type="press",
                source_id=f"exa-{peer}-abc{i:04d}",
                extracted_by="test",
                extraction_version="v1.1",
            )
        )
    append_to_pool(pool_path, ptcs)
    return pool_path


def test_embed_peer_ptcs_creates_files(tmp_path):
    peers = ["MT.AS", "CGEM.PA"]
    pool_path = _make_peer_pool(tmp_path, peers)
    output_dir = tmp_path / "interim"

    import json as _json

    body_mock = MagicMock()
    body_mock.read.return_value = _json.dumps(
        {"embedding": list(np.random.rand(1024).astype(float))}
    ).encode()
    client = MagicMock()
    client.invoke_model.return_value = {"body": body_mock}

    with patch("signal_intelligence.sector_ontology.boto3.client", return_value=client):
        vectors, index_df = embed_peer_ptcs(pool_path, peers, output_dir)

    assert (output_dir / "sector_embeddings.npy").exists()
    assert (output_dir / "sector_embedding_index.csv").exists()
    assert vectors.shape == (2, 1024)
    assert len(index_df) == 2
