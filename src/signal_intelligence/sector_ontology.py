"""Phase 0 — Sector ontology.

Builds a sector baseline of causal mechanisms from peer companies so that
focal-company canonicals can be tagged sector_wide vs company_specific.

Pipeline
--------
1. discover_peers()   — ask Bedrock for top-N sector peers of a focal ticker.
2. fetch_peer_press() — collect press articles for each peer via Exa API.
3. extract_peer_ptcs() — run ptc_extraction on the peer press corpus.
4. embed_peer_ptcs()  — embed via Bedrock Titan (reuses embedding.py logic).
5. build_sector_vectors() — HDBSCAN cluster peer PTCs → centroid per cluster.

The returned sector_vectors array is passed directly to cluster_ptcs() as the
sector_vectors parameter.

All Bedrock and Exa calls are isolated behind thin wrappers so tests can mock
them without AWS or network access.
"""

import json
import logging
import os
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DISCOVER_MODEL = "eu.anthropic.claude-sonnet-4-6"
_DISCOVER_REGION = "eu-west-3"
_TITAN_MODEL_ID = "amazon.titan-embed-text-v2:0"
_EMBED_DIM = 1024
_DEFAULT_N_PEERS = 5
_DEFAULT_N_ARTICLES = 10
_MIN_CLUSTER_SIZE = 3


# ---------------------------------------------------------------------------
# Step 1 — peer discovery
# ---------------------------------------------------------------------------


def discover_peers(
    ticker: str,
    company_name: str,
    sector: str,
    n_peers: int = _DEFAULT_N_PEERS,
    region: str = _DISCOVER_REGION,
) -> list[str]:
    """Return a list of n_peers competitor ticker symbols via Bedrock.

    Uses a single zero-temperature Claude call. Returns an empty list on error
    so the pipeline can continue without a sector baseline.
    """
    client = boto3.client("bedrock-runtime", region_name=region)
    prompt = (
        f"List exactly {n_peers} publicly traded competitors of {company_name} "
        f"({ticker}), a company in the {sector} sector. "
        "Return ONLY a JSON array of ticker symbols (exchange-qualified if needed, "
        'e.g. ["MT.AS","CGEM.PA","5108.T"]). No explanations, no markdown.'
    )
    try:
        response = client.converse(
            modelId=_DISCOVER_MODEL,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": 0, "maxTokens": 256},
        )
        text = response["output"]["message"]["content"][0]["text"].strip()
        peers: list[str] = json.loads(text)
        logger.info("Discovered %d peers for %s: %s", len(peers), ticker, peers)
        return peers
    except Exception:
        logger.exception("Peer discovery failed for %s", ticker)
        return []


# ---------------------------------------------------------------------------
# Step 2 — press fetch via Exa
# ---------------------------------------------------------------------------


def fetch_peer_press(
    peers: list[str],
    company_names: list[str] | None = None,
    n_articles_per_peer: int = _DEFAULT_N_ARTICLES,
) -> list[dict]:
    """Fetch recent press articles for each peer using the Exa API.

    Requires EXA_API_KEY in environment.

    Returns list of dicts: {source_id, text, source_type, source_date}.
    source_type is always "press". source_id = "exa-{peer}-{url_hash[:8]}".
    """
    from datetime import date, timedelta

    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        raise OSError("EXA_API_KEY is not set. Add it to your .env file.")

    try:
        from exa_py import Exa  # type: ignore[import]
    except ImportError as e:
        raise ImportError("exa-py is required for sector ontology. Run: uv add exa-py") from e

    exa = Exa(api_key=api_key)
    names = company_names or peers
    docs: list[dict] = []
    cutoff = (date.today() - timedelta(days=365)).isoformat()

    for peer, name in zip(peers, names, strict=False):
        try:
            results = exa.search_and_contents(
                f"{name} earnings results financial performance",
                num_results=n_articles_per_peer,
                start_published_date=cutoff,
                text=True,
            )
            for item in results.results:
                if not item.text:
                    continue
                url_hash = str(abs(hash(item.url)))[:8]
                source_id = f"exa-{peer}-{url_hash}"
                pub_date = (
                    date.fromisoformat(item.published_date[:10])
                    if item.published_date
                    else date.today()
                )
                docs.append(
                    {
                        "source_id": source_id,
                        "text": item.text,
                        "source_type": "press",
                        "source_date": pub_date,
                    }
                )
            logger.info("Fetched %d articles for peer %s", len(results.results), peer)
        except Exception:
            logger.exception("Exa fetch failed for peer %s", peer)

    return docs


# ---------------------------------------------------------------------------
# Step 3 — PTC extraction on peer press
# ---------------------------------------------------------------------------


def extract_peer_ptcs(
    peer_docs: list[dict],
    pool_path: Path,
    region: str = _DISCOVER_REGION,
) -> int:
    """Extract PTCs from peer press documents and append to pool_path.

    Returns the number of new PTCs written.
    """
    from ptc_extraction.backends.bedrock import BedrockBackend
    from ptc_extraction.extractor import PTCExtractor

    backend = BedrockBackend(region=region)
    extractor = PTCExtractor(backend=backend)
    stats = extractor.extract_from_documents(peer_docs, output_path=pool_path, resume=True)
    logger.info("Peer PTC extraction: %s", stats)
    return stats.get("written", 0)


# ---------------------------------------------------------------------------
# Step 4 — embed peer PTCs
# ---------------------------------------------------------------------------


def embed_peer_ptcs(
    pool_path: Path,
    peer_tickers: list[str],
    output_dir: Path,
    region: str = _DISCOVER_REGION,
    refresh: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Embed all PTCs whose source_id starts with 'exa-{peer}-' for any peer.

    Reuses the same Bedrock Titan logic as embedding.py but filters by peer
    source_ids instead of a single ticker prefix.
    """
    from ptc_extraction.io import load_pool

    from signal_intelligence.embedding import _INDEX_COLUMNS, _embed_text

    output_dir.mkdir(parents=True, exist_ok=True)
    npy_path = output_dir / "sector_embeddings.npy"
    index_path = output_dir / "sector_embedding_index.csv"

    peer_prefixes = tuple(f"exa-{p}-" for p in peer_tickers)
    ptcs = [p for p in load_pool(pool_path) if p.source_id.startswith(peer_prefixes)]

    existing_hashes: set[str] = set()
    existing_vectors: list[np.ndarray] = []
    existing_rows: list[dict] = []

    if not refresh and index_path.exists() and npy_path.exists():
        existing_index = pd.read_csv(index_path, dtype=str)
        existing_hashes = set(existing_index["content_hash"].tolist())
        existing_vectors = list(np.load(npy_path))
        existing_rows = existing_index.to_dict("records")

    to_embed = [p for p in ptcs if p.content_hash() not in existing_hashes]
    if not to_embed and existing_vectors:
        return np.stack(existing_vectors).astype(np.float32), pd.DataFrame(
            existing_rows, columns=_INDEX_COLUMNS
        )

    client = boto3.client("bedrock-runtime", region_name=region)
    new_vectors: list[np.ndarray] = []
    new_rows: list[dict] = []

    for ptc in to_embed:
        embed_text = f"[{ptc.direction}] [{ptc.polarity:+d}] {ptc.mechanism}"
        vector = None
        for attempt in range(3):
            try:
                vector = _embed_text(client, embed_text)
                break
            except Exception:
                if attempt == 2:
                    logger.exception("Failed to embed peer PTC %s", ptc.content_hash())
        if vector is None:
            continue
        new_vectors.append(vector)
        new_rows.append(
            {
                "content_hash": ptc.content_hash(),
                "source_id": ptc.source_id,
                "source_type": ptc.source_type,
                "direction": ptc.direction,
                "polarity": ptc.polarity,
                "source_date": ptc.source_date.isoformat(),
                "mechanism": ptc.mechanism[:120],
            }
        )

    all_vectors = existing_vectors + new_vectors
    all_rows = existing_rows + new_rows
    if not all_vectors:
        return np.zeros((0, _EMBED_DIM), dtype=np.float32), pd.DataFrame(columns=_INDEX_COLUMNS)

    vectors = np.stack(all_vectors).astype(np.float32)
    np.save(npy_path, vectors)
    index_df = pd.DataFrame(all_rows, columns=_INDEX_COLUMNS)
    index_df.to_csv(index_path, index=False)
    return vectors, index_df


# ---------------------------------------------------------------------------
# Step 5 — build sector centroid vectors
# ---------------------------------------------------------------------------


def build_sector_vectors(
    vectors: np.ndarray,
    min_cluster_size: int = _MIN_CLUSTER_SIZE,
) -> np.ndarray:
    """Cluster peer embeddings with HDBSCAN and return one centroid per cluster.

    Noise points are excluded. Returns shape (n_clusters, embed_dim).
    Returns empty array if fewer than min_cluster_size vectors available.
    """
    from sklearn.cluster import HDBSCAN

    if len(vectors) < min_cluster_size:
        logger.warning(
            "Only %d peer vectors — not enough for sector clustering (min %d)",
            len(vectors),
            min_cluster_size,
        )
        return np.zeros(
            (0, vectors.shape[1] if vectors.ndim == 2 else _EMBED_DIM), dtype=np.float32
        )

    model = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = model.fit_predict(vectors)

    centroids = []
    for label in sorted(set(labels)):
        if label == -1:
            continue
        members = vectors[labels == label]
        centroids.append(members.mean(axis=0))

    if not centroids:
        logger.warning("All peer PTCs were noise — sector baseline is empty.")
        return np.zeros((0, vectors.shape[1]), dtype=np.float32)

    result = np.stack(centroids).astype(np.float32)
    logger.info("Built %d sector centroid vectors from %d peer PTCs", len(result), len(vectors))
    return result


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------


def build_sector_ontology(
    focal_ticker: str,
    company_name: str,
    sector: str,
    pool_path: Path,
    output_dir: Path = Path("data/interim"),
    n_peers: int = _DEFAULT_N_PEERS,
    n_articles_per_peer: int = _DEFAULT_N_ARTICLES,
    region: str = _DISCOVER_REGION,
    refresh: bool = False,
) -> np.ndarray:
    """Run the full Phase 0 pipeline.

    Returns sector_vectors array ready to pass into cluster_ptcs().
    Returns an empty array if any step produces no data (graceful degradation).
    """
    peers = discover_peers(focal_ticker, company_name, sector, n_peers, region)
    if not peers:
        logger.warning("No peers discovered — sector baseline will be empty.")
        return np.zeros((0, _EMBED_DIM), dtype=np.float32)

    peer_docs = fetch_peer_press(peers, n_articles_per_peer=n_articles_per_peer)
    if not peer_docs:
        logger.warning("No peer press docs fetched — sector baseline will be empty.")
        return np.zeros((0, _EMBED_DIM), dtype=np.float32)

    extract_peer_ptcs(peer_docs, pool_path, region)

    vectors, _ = embed_peer_ptcs(pool_path, peers, output_dir, region, refresh)
    if len(vectors) == 0:
        return np.zeros((0, _EMBED_DIM), dtype=np.float32)

    return build_sector_vectors(vectors)
