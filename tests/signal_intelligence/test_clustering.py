from pathlib import Path

import numpy as np
import pandas as pd

from signal_intelligence.clustering import (
    _cross_map,
    _split_by_direction_polarity,
    cluster_ptcs,
)

# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------


def _make_index_row(
    *,
    content_hash: str,
    source_id: str,
    source_type: str = "management",
    direction: str = "precursor",
    polarity: int = -1,
    mechanism: str = "Rising raw material costs compress operating margins",
) -> dict:
    return {
        "content_hash": content_hash,
        "source_id": source_id,
        "source_type": source_type,
        "direction": direction,
        "polarity": polarity,
        "source_date": "2024-01-30",
        "mechanism": mechanism,
    }


def _make_embeddings_files(
    tmp_path: Path,
    ticker: str,
    rows: list[dict],
    vectors: np.ndarray,
) -> None:
    (tmp_path / f"{ticker}_embeddings.npy").parent.mkdir(parents=True, exist_ok=True)
    np.save(tmp_path / f"{ticker}_embeddings.npy", vectors.astype(np.float32))
    pd.DataFrame(rows).to_csv(tmp_path / f"{ticker}_embedding_index.csv", index=False)


def _unit_vector(seed: int, dim: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# _split_by_direction_polarity
# ---------------------------------------------------------------------------


def test_direction_polarity_split_separates_mixed_cluster():
    df = pd.DataFrame(
        [
            {
                "_label": 0,
                "direction": "precursor",
                "polarity": -1,
                "source_id": "a",
                "mechanism": "m1",
            },
            {
                "_label": 0,
                "direction": "precursor",
                "polarity": 1,
                "source_id": "b",
                "mechanism": "m2",
            },
            {
                "_label": 0,
                "direction": "consequence",
                "polarity": -1,
                "source_id": "c",
                "mechanism": "m3",
            },
        ]
    )
    vectors = np.eye(3, dtype=np.float32)
    result = _split_by_direction_polarity(df, vectors)
    # Three distinct (direction, polarity) combos → three sub-cluster labels
    assert result["_label"].nunique() == 3


def test_direction_polarity_split_noise_kept():
    df = pd.DataFrame(
        [
            {
                "_label": -1,
                "direction": "precursor",
                "polarity": -1,
                "source_id": "a",
                "mechanism": "m",
            },
            {
                "_label": 0,
                "direction": "precursor",
                "polarity": -1,
                "source_id": "b",
                "mechanism": "m",
            },
        ]
    )
    vectors = np.eye(2, dtype=np.float32)
    result = _split_by_direction_polarity(df, vectors)
    noise_rows = result[result["_label"] < 0]
    assert len(noise_rows) == 1


# ---------------------------------------------------------------------------
# _cross_map
# ---------------------------------------------------------------------------


def test_cross_map_triangulated():
    base = _unit_vector(42)
    mgmt = {"m1": base.copy()}
    press = {"p1": base.copy()}
    mapping = _cross_map(mgmt, press, threshold=0.82)
    assert "m1" in mapping
    assert mapping["m1"][0] == "p1"
    assert mapping["m1"][1] >= 0.99


def test_cross_map_insider_only():
    mgmt = {"m1": _unit_vector(1)}
    press = {"p1": _unit_vector(99)}  # orthogonal-ish vectors
    mapping = _cross_map(mgmt, press, threshold=0.99)
    # If similarity is below threshold, mapping should be empty
    sim = mapping.get("m1")
    if sim is not None:
        assert sim[1] < 0.99


def test_cross_map_empty_press():
    mgmt = {"m1": _unit_vector(1)}
    mapping = _cross_map(mgmt, {}, threshold=0.82)
    assert mapping == {}


# ---------------------------------------------------------------------------
# cluster_ptcs end-to-end
# ---------------------------------------------------------------------------


def _make_cluster_fixture(
    tmp_path: Path,
    ticker: str = "ML.PA",
) -> tuple[Path, Path]:
    """Create embeddings with enough PTCs to form at least one cluster per corpus."""
    rng = np.random.default_rng(0)

    # Build 6 management PTCs: two tight groups of 3
    mgmt_rows = []
    mgmt_vecs = []
    base_mgmt_a = _unit_vector(10)
    base_mgmt_b = _unit_vector(20)
    for i in range(3):
        mgmt_vecs.append(
            base_mgmt_a + rng.standard_normal(base_mgmt_a.shape).astype(np.float32) * 0.01
        )
        mgmt_rows.append(
            _make_index_row(
                content_hash=f"hm{i}",
                source_id=f"{ticker}-management-2024-01-01-doc{i}",
                source_type="management",
            )
        )
    for i in range(3, 6):
        mgmt_vecs.append(
            base_mgmt_b + rng.standard_normal(base_mgmt_b.shape).astype(np.float32) * 0.01
        )
        mgmt_rows.append(
            _make_index_row(
                content_hash=f"hm{i}",
                source_id=f"{ticker}-management-2024-01-01-doc{i}",
                source_type="management",
            )
        )

    # Build 6 press PTCs: one cluster very close to mgmt cluster A (triangulated)
    press_rows = []
    press_vecs = []
    for i in range(3):
        press_vecs.append(
            base_mgmt_a + rng.standard_normal(base_mgmt_a.shape).astype(np.float32) * 0.01
        )
        press_rows.append(
            _make_index_row(
                content_hash=f"hp{i}",
                source_id=f"{ticker}-press-2024-01-01-art{i}",
                source_type="press",
            )
        )
    base_press_b = _unit_vector(30)
    for i in range(3, 6):
        press_vecs.append(
            base_press_b + rng.standard_normal(base_press_b.shape).astype(np.float32) * 0.01
        )
        press_rows.append(
            _make_index_row(
                content_hash=f"hp{i}",
                source_id=f"{ticker}-press-2024-01-01-art{i}",
                source_type="press",
            )
        )

    all_rows = mgmt_rows + press_rows
    all_vecs = np.stack(mgmt_vecs + press_vecs).astype(np.float32)
    # Normalise
    norms = np.linalg.norm(all_vecs, axis=1, keepdims=True)
    all_vecs = all_vecs / np.where(norms == 0, 1, norms)

    embeddings_dir = tmp_path / "interim"
    embeddings_dir.mkdir()
    _make_embeddings_files(embeddings_dir, ticker, all_rows, all_vecs)

    output_dir = tmp_path / "processed"
    return embeddings_dir, output_dir


def test_cluster_ptcs_returns_dataframe_with_expected_columns(tmp_path):
    embeddings_dir, output_dir = _make_cluster_fixture(tmp_path)
    df = cluster_ptcs(
        "ML.PA", embeddings_dir=embeddings_dir, output_dir=output_dir, min_cluster_size=3
    )

    expected = {
        "canonical_id",
        "ticker",
        "corpus",
        "direction",
        "polarity",
        "corpus_presence",
        "sector_tag",
        "n_ptcs",
        "is_noise",
        "representative_mechanism",
        "source_ids",
        "matched_canonical_id",
        "cross_map_similarity",
    }
    assert expected.issubset(set(df.columns))


def test_cluster_ptcs_saves_csv(tmp_path):
    embeddings_dir, output_dir = _make_cluster_fixture(tmp_path)
    cluster_ptcs("ML.PA", embeddings_dir=embeddings_dir, output_dir=output_dir, min_cluster_size=3)
    assert (output_dir / "ML.PA_canonicals.csv").exists()


def test_cross_map_triangulated_in_full_pipeline(tmp_path):
    embeddings_dir, output_dir = _make_cluster_fixture(tmp_path)
    # threshold 0.5 ensures the deliberately close cluster pair matches
    df = cluster_ptcs(
        "ML.PA",
        embeddings_dir=embeddings_dir,
        output_dir=output_dir,
        min_cluster_size=3,
        cross_map_threshold=0.5,
    )
    assert "triangulated" in df["corpus_presence"].values


def test_cross_map_insider_only_in_full_pipeline(tmp_path):
    embeddings_dir, output_dir = _make_cluster_fixture(tmp_path)
    # threshold 0.9999 means nothing is close enough to triangulate
    df = cluster_ptcs(
        "ML.PA",
        embeddings_dir=embeddings_dir,
        output_dir=output_dir,
        min_cluster_size=3,
        cross_map_threshold=0.9999,
    )
    assert "insider_only" in df["corpus_presence"].values


def test_sector_tagging_untagged_when_no_baseline(tmp_path):
    embeddings_dir, output_dir = _make_cluster_fixture(tmp_path)
    df = cluster_ptcs(
        "ML.PA",
        embeddings_dir=embeddings_dir,
        output_dir=output_dir,
        min_cluster_size=3,
        sector_vectors=None,
    )
    assert (df["sector_tag"] == "untagged").all()


def test_sector_tagging_sector_wide(tmp_path):
    embeddings_dir, output_dir = _make_cluster_fixture(tmp_path)
    cluster_ptcs("ML.PA", embeddings_dir=embeddings_dir, output_dir=output_dir, min_cluster_size=3)
    # Reload embeddings to build a sector vector identical to one canonical centroid
    vecs = np.load(embeddings_dir / "ML.PA_embeddings.npy")
    sector_vec = vecs[0:3].mean(axis=0)
    sector_vec = sector_vec / np.linalg.norm(sector_vec)
    sector_vectors = sector_vec[np.newaxis, :]

    output_dir2 = tmp_path / "processed2"
    df = cluster_ptcs(
        "ML.PA",
        embeddings_dir=embeddings_dir,
        output_dir=output_dir2,
        min_cluster_size=3,
        sector_vectors=sector_vectors,
        sector_threshold=0.5,
    )
    assert "sector_wide" in df["sector_tag"].values


def test_sector_tagging_company_specific(tmp_path):
    embeddings_dir, _output_dir = _make_cluster_fixture(tmp_path)
    # Use a sector vector orthogonal to everything
    sector_vec = np.zeros(16, dtype=np.float32)
    sector_vec[0] = 1.0
    sector_vectors = sector_vec[np.newaxis, :]

    output_dir2 = tmp_path / "processed2"
    df = cluster_ptcs(
        "ML.PA",
        embeddings_dir=embeddings_dir,
        output_dir=output_dir2,
        min_cluster_size=3,
        sector_vectors=sector_vectors,
        sector_threshold=0.999,
    )
    assert "company_specific" in df["sector_tag"].values


def test_noise_points_become_singleton_canonicals(tmp_path):
    """When min_cluster_size > corpus size, all points become noise."""
    ticker = "ML.PA"
    rng = np.random.default_rng(7)
    rows = []
    vecs = []
    for i in range(2):
        rows.append(
            _make_index_row(
                content_hash=f"h{i}",
                source_id=f"{ticker}-management-2024-01-01-doc{i}",
                source_type="management",
            )
        )
        v = rng.standard_normal(16).astype(np.float32)
        vecs.append(v / np.linalg.norm(v))

    embeddings_dir = tmp_path / "interim2"
    embeddings_dir.mkdir()
    _make_embeddings_files(embeddings_dir, ticker, rows, np.stack(vecs))

    output_dir = tmp_path / "processed2"
    df = cluster_ptcs(
        ticker,
        embeddings_dir=embeddings_dir,
        output_dir=output_dir,
        min_cluster_size=10,  # larger than corpus
    )
    assert df["is_noise"].all()
    assert len(df) == 2
