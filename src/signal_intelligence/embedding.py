import json
import logging
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from ptc_extraction.io import load_pool

logger = logging.getLogger(__name__)

_TITAN_MODEL_ID = "amazon.titan-embed-text-v2:0"
_EMBED_DIM = 1024

_INDEX_COLUMNS = [
    "content_hash",
    "source_id",
    "source_type",
    "direction",
    "polarity",
    "source_date",
    "mechanism",
]


def _embed_text(client, text: str) -> np.ndarray:
    body = json.dumps({"inputText": text})
    response = client.invoke_model(
        modelId=_TITAN_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(response["body"].read())
    return np.array(result["embedding"], dtype=np.float32)


def embed_ptcs(
    ticker: str,
    pool_path: Path,
    output_dir: Path = Path("data/interim"),
    region: str = "eu-west-3",
    refresh: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Embed all PTCs in pool_path for the given ticker.

    Returns (vectors, index_df). Saves .npy and index CSV to output_dir.
    Skips PTCs whose content_hash is already in the index (resume-safe).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_path = output_dir / f"{ticker}_embeddings.npy"
    index_path = output_dir / f"{ticker}_embedding_index.csv"

    ptcs = [p for p in load_pool(pool_path) if p.source_id.startswith(f"{ticker}-")]

    existing_hashes: set[str] = set()
    existing_vectors: list[np.ndarray] = []
    existing_rows: list[dict] = []

    if not refresh and index_path.exists() and npy_path.exists():
        existing_index = pd.read_csv(index_path, dtype=str)
        existing_hashes = set(existing_index["content_hash"].tolist())
        existing_vectors = list(np.load(npy_path))
        existing_rows = existing_index.to_dict("records")
        logger.info("Loaded %d existing embeddings for %s", len(existing_hashes), ticker)

    to_embed = [p for p in ptcs if p.content_hash() not in existing_hashes]
    logger.info(
        "%d PTCs to embed for %s (%d already cached)",
        len(to_embed),
        ticker,
        len(existing_hashes),
    )

    if not to_embed:
        if existing_vectors:
            vectors = np.stack(existing_vectors).astype(np.float32)
            index_df = pd.DataFrame(existing_rows, columns=_INDEX_COLUMNS)
            return vectors, index_df
        return np.zeros((0, _EMBED_DIM), dtype=np.float32), pd.DataFrame(columns=_INDEX_COLUMNS)

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
                    logger.exception(
                        "Failed to embed PTC %s after 3 attempts, skipping",
                        ptc.content_hash(),
                    )
                else:
                    logger.warning(
                        "Bedrock error on attempt %d for %s, retrying",
                        attempt + 1,
                        ptc.content_hash(),
                    )

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

    vectors = np.stack(all_vectors).astype(np.float32)
    np.save(npy_path, vectors)

    index_df = pd.DataFrame(all_rows, columns=_INDEX_COLUMNS)
    index_df.to_csv(index_path, index=False)

    logger.info(
        "Saved %d embeddings to %s (%d new)",
        len(all_rows),
        npy_path,
        len(new_rows),
    )
    return vectors, index_df
