"""Project a company's 1024-dim claim embeddings into 3D for the Claims-page scatter.

Reads data/interim/{ticker}_embeddings.npy (+ _embedding_index.csv) and the
canonical table, runs UMAP down to 3 dimensions, and writes a small
app/demo/{ticker}_projection.csv (one row per claim: coords + cluster + typing).
Only the 3 coordinates ship to the app — never the raw 1024-dim vectors.

    uv run python scripts/build_projection.py ML.PA
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

INTERIM = Path("data/interim")
PROCESSED = Path("data/processed")
DEMO = Path("app/demo")


def build_projection(ticker: str, seed: int = 42) -> None:
    npy = INTERIM / f"{ticker}_embeddings.npy"
    idx = INTERIM / f"{ticker}_embedding_index.csv"
    if not npy.exists() or not idx.exists():
        raise SystemExit(f"No embeddings for {ticker} — run the pipeline first.")

    vectors = np.load(npy)
    index = pd.read_csv(idx, dtype=str)
    index["polarity"] = pd.to_numeric(index["polarity"], errors="coerce").fillna(0).astype(int)

    # map each claim's source_id -> its canonical_id and corpus_presence
    canon = pd.read_csv(PROCESSED / f"{ticker}_canonicals.csv")
    sid_to_canon: dict[str, str] = {}
    sid_to_presence: dict[str, str] = {}
    for _, row in canon.iterrows():
        try:
            sids = json.loads(row["source_ids"])
        except (TypeError, ValueError):
            continue
        for s in sids:
            sid_to_canon[s] = row["canonical_id"]
            sid_to_presence[s] = row.get("corpus_presence", "")

    import umap

    n = len(vectors)
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=min(15, n - 1),
        min_dist=0.1,
        metric="cosine",
        random_state=seed,
    )
    coords = reducer.fit_transform(vectors)

    out = pd.DataFrame(
        {
            "x": coords[:, 0].round(3),
            "y": coords[:, 1].round(3),
            "z": coords[:, 2].round(3),
            "source_id": index["source_id"],
            "source_type": index["source_type"],
            "direction": index["direction"],
            "polarity": index["polarity"],
            "mechanism": index["mechanism"].str.slice(0, 160),
        }
    )
    out["canonical_id"] = out["source_id"].map(sid_to_canon).fillna("noise")
    out["corpus_presence"] = out["source_id"].map(sid_to_presence).fillna("")

    DEMO.mkdir(parents=True, exist_ok=True)
    dest = DEMO / f"{ticker}_projection.csv"
    out.to_csv(dest, index=False)
    print(f"{ticker}: projected {n} claims -> {dest} ({dest.stat().st_size // 1024} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="+")
    args = parser.parse_args()
    for t in args.tickers:
        build_projection(t)


if __name__ == "__main__":
    main()
