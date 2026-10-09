import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN

logger = logging.getLogger(__name__)

_CANONICAL_COLUMNS = [
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
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _hdbscan_cluster(vectors: np.ndarray, min_cluster_size: int) -> np.ndarray:
    if len(vectors) < min_cluster_size:
        return np.full(len(vectors), -1, dtype=int)
    model = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    return model.fit_predict(vectors)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _split_by_direction_polarity(cluster_df: pd.DataFrame, vectors: np.ndarray) -> pd.DataFrame:
    """Re-assign cluster labels so each (direction, polarity) pair is its own sub-cluster."""
    rows = []
    sub_id = 0
    for orig_label in sorted(cluster_df["_label"].unique()):
        if orig_label == -1:
            noise_mask = cluster_df["_label"] == -1
            noise_df = cluster_df[noise_mask].copy()
            noise_df["_label"] = -(np.arange(len(noise_df)) + 1)
            rows.append(noise_df)
            continue
        group = cluster_df[cluster_df["_label"] == orig_label]
        for (_direction, _polarity), sub_group in group.groupby(
            ["direction", "polarity"], sort=False
        ):
            sub = sub_group.copy()
            sub["_label"] = sub_id
            sub_id += 1
            rows.append(sub)

    return pd.concat(rows, ignore_index=True) if rows else cluster_df.copy()


def _cross_map(
    mgmt_centroids: dict[str, np.ndarray],
    press_centroids: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, tuple[str, float]]:
    """Return mapping: mgmt_canonical_id → (press_canonical_id, similarity).

    Only pairs above threshold are included.
    """
    mapping: dict[str, tuple[str, float]] = {}
    for m_id, m_vec in mgmt_centroids.items():
        best_id, best_sim = None, -1.0
        for p_id, p_vec in press_centroids.items():
            sim = _cosine_sim(m_vec, p_vec)
            if sim > best_sim:
                best_sim = sim
                best_id = p_id
        if best_sim >= threshold and best_id is not None:
            mapping[m_id] = (best_id, best_sim)
    return mapping


def _build_canonicals(
    ticker: str,
    corpus: str,
    sub_df: pd.DataFrame,
    vectors: np.ndarray,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Build canonical rows + centroid map for one corpus."""
    rows: list[dict] = []
    centroids: dict[str, np.ndarray] = {}

    for label in sorted(sub_df["_label"].unique()):
        is_noise = label < 0
        members = sub_df[sub_df["_label"] == label]
        member_indices = members.index.tolist()
        member_vectors = vectors[member_indices]

        direction = members["direction"].mode().iloc[0]
        polarity = int(members["polarity"].mode().iloc[0])

        if is_noise:
            canonical_id = f"{ticker}-{corpus}-noise-{abs(label)}"
            centroid = member_vectors[0]
        else:
            canonical_id = f"{ticker}-{corpus}-{label}"
            centroid = member_vectors.mean(axis=0)

        centroids[canonical_id] = centroid

        # representative = PTC closest to centroid
        sims = [_cosine_sim(v, centroid) for v in member_vectors]
        rep_idx = int(np.argmax(sims))
        rep_mechanism = members.iloc[rep_idx]["mechanism"]

        rows.append(
            {
                "canonical_id": canonical_id,
                "ticker": ticker,
                "corpus": corpus,
                "direction": direction,
                "polarity": polarity,
                "corpus_presence": None,  # filled later
                "sector_tag": "untagged",
                "n_ptcs": len(members),
                "is_noise": is_noise,
                "representative_mechanism": rep_mechanism,
                "source_ids": json.dumps(members["source_id"].tolist()),
                "matched_canonical_id": None,
                "cross_map_similarity": None,
            }
        )

    return rows, centroids


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cluster_ptcs(
    ticker: str,
    embeddings_dir: Path = Path("data/interim"),
    output_dir: Path = Path("data/processed"),
    min_cluster_size: int = 3,
    cross_map_threshold: float = 0.82,
    sector_vectors: np.ndarray | None = None,
    sector_threshold: float = 0.80,
) -> pd.DataFrame:
    """Load embeddings for ticker, run all four phases, return canonical table.

    Saves canonical table to {output_dir}/{ticker}_canonicals.csv.
    """
    npy_path = embeddings_dir / f"{ticker}_embeddings.npy"
    index_path = embeddings_dir / f"{ticker}_embedding_index.csv"

    if not npy_path.exists() or not index_path.exists():
        logger.warning(
            "No embeddings on disk for %s (%s) — returning empty table.", ticker, npy_path
        )
        return pd.DataFrame()

    vectors = np.load(npy_path)
    index_df = pd.read_csv(index_path, dtype=str)
    index_df["polarity"] = index_df["polarity"].astype(int)

    # Phase A + B: per-corpus HDBSCAN with direction x polarity split
    all_rows: list[dict] = []
    mgmt_centroids: dict[str, np.ndarray] = {}
    press_centroids: dict[str, np.ndarray] = {}

    for corpus, group_df in index_df.groupby("source_type"):
        group_df = group_df.copy().reset_index()
        orig_indices = group_df["index"].tolist()
        corpus_vectors = vectors[orig_indices]

        labels = _hdbscan_cluster(corpus_vectors, min_cluster_size)
        group_df["_label"] = labels
        group_df = _split_by_direction_polarity(group_df, corpus_vectors)

        # Rebuild vectors aligned to the (possibly re-indexed) group_df
        local_vectors = corpus_vectors  # indices still match after reset_index on a copy

        rows, centroids = _build_canonicals(ticker, corpus, group_df, local_vectors)
        all_rows.extend(rows)

        if corpus == "management":
            mgmt_centroids = centroids
        else:
            press_centroids = centroids

    canonical_df = pd.DataFrame(all_rows, columns=_CANONICAL_COLUMNS)

    # Phase C: cross-map management ↔ press
    cross = _cross_map(mgmt_centroids, press_centroids, cross_map_threshold)
    matched_press: set[str] = set()

    for m_id, (p_id, sim) in cross.items():
        canonical_df.loc[canonical_df["canonical_id"] == m_id, "corpus_presence"] = "triangulated"
        canonical_df.loc[canonical_df["canonical_id"] == m_id, "matched_canonical_id"] = p_id
        canonical_df.loc[canonical_df["canonical_id"] == m_id, "cross_map_similarity"] = round(
            sim, 4
        )

        canonical_df.loc[canonical_df["canonical_id"] == p_id, "corpus_presence"] = "triangulated"
        canonical_df.loc[canonical_df["canonical_id"] == p_id, "matched_canonical_id"] = m_id
        canonical_df.loc[canonical_df["canonical_id"] == p_id, "cross_map_similarity"] = round(
            sim, 4
        )
        matched_press.add(p_id)

    mask_mgmt = canonical_df["corpus"] == "management"
    mask_press = canonical_df["corpus"] == "press"
    mask_unset = canonical_df["corpus_presence"].isna()

    canonical_df.loc[mask_mgmt & mask_unset, "corpus_presence"] = "insider_only"
    canonical_df.loc[mask_press & mask_unset, "corpus_presence"] = "outsider_only"

    # Phase D: sector tagging (optional)
    if sector_vectors is not None and len(sector_vectors) > 0:
        for idx, row in canonical_df.iterrows():
            cid = row["canonical_id"]
            centroid = mgmt_centroids.get(cid)
            if centroid is None:
                centroid = press_centroids.get(cid)
            if centroid is None:
                continue
            max_sim = max(_cosine_sim(centroid, sv) for sv in sector_vectors)
            tag = "sector_wide" if max_sim >= sector_threshold else "company_specific"
            canonical_df.at[idx, "sector_tag"] = tag

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{ticker}_canonicals.csv"
    canonical_df.to_csv(out_path, index=False)
    logger.info("Saved canonical table to %s (%d rows)", out_path, len(canonical_df))

    return canonical_df
