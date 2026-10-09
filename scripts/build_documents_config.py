"""Build the `documents` config: the extracted text each span indexes into.

Without this, `span_start`/`span_end` are unusable by anyone but us. They are
offsets into the *pipeline-extracted text* of a document, not into the source
PDF's bytes, so shipping the PDFs would hand a consumer two things that do not
fit together. This config ships the exact text the offsets refer to, which is
what makes the corpus a self-contained span-labelled dataset.

Management documents only. Those are the companies' own regulatory filings,
published to be read and quoted. Press articles are third-party journalism and
are not redistributed in any form — which costs little here, since the
management corpus holds the large majority of resolvable spans.

The script re-reads every document with the same ingestion code that produced
the extraction, so the text is byte-identical to what the offsets were computed
against. It then verifies that claim by claim and refuses to write a document
whose spans do not resolve.

Usage:
    python scripts/build_documents_config.py --out data/hf_dataset
"""

import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from signal_intelligence.ingestion import load_documents

TICKERS = ["ML.PA", "MC.PA", "OR.PA", "SAN.PA", "TTE.PA"]


def _norm(s: str) -> str:
    return " ".join(s.split())


def _norm_hard(s: str) -> str:
    # Escapes, not literals: these characters are confusable by design.
    table = str.maketrans(
        {
            "\u2019": "'",  # right single quote
            "\u2018": "'",  # left single quote
            "\u201c": '"',  # left double quote
            "\u201d": '"',  # right double quote
            "\u2013": "-",  # en dash
            "\u2014": "-",  # em dash
            "\u00a0": " ",  # non-breaking space
            "\ufb01": "fi",  # fi ligature
            "\ufb02": "fl",  # fl ligature
        }
    )
    return _norm(unicodedata.normalize("NFKC", s.translate(table)))


def build(pool_path: Path, out_dir: Path, tickers: list[str]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [json.loads(x) for x in pool_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    spans_by_doc: dict[str, list[dict]] = {}
    for r in rows:
        if r["source_type"] == "management" and r.get("span"):
            spans_by_doc.setdefault(r["source_id"], []).append(r)

    stats: Counter = Counter()
    docs: list[dict] = []

    for ticker in tickers:
        try:
            loaded = load_documents(ticker)
        except Exception as e:
            print(f"  ! {ticker}: {e}")
            continue
        mgmt = [d for d in loaded if d["source_type"] == "management"]
        print(f"  {ticker}: {len(mgmt)} management documents", flush=True)

        for d in mgmt:
            sid, text = d["source_id"], d["text"]
            # Verify every span this document is supposed to support.
            ok = bad = 0
            for claim in spans_by_doc.get(sid, ()):
                start, end = claim["span"]
                sliced = text[start:end]
                method = claim.get("span_match") or "verbatim"
                if method == "verbatim":
                    good = sliced == claim["raw_text"]
                elif method == "whitespace":
                    good = _norm(sliced) == _norm(claim["raw_text"])
                else:
                    good = _norm_hard(sliced) == _norm_hard(claim["raw_text"])
                ok += good
                bad += not good
            if bad:
                stats["documents_with_unresolvable_spans"] += 1
                stats["unresolvable_spans"] += bad
                print(f"    ! {sid}: {bad}/{ok + bad} spans do not resolve — document skipped")
                continue

            stats["documents"] += 1
            stats["spans_verified"] += ok
            stats["chars"] += len(text)
            docs.append(
                {
                    "source_id": sid,
                    "ticker": ticker,
                    "source_type": "management",
                    "source_date": d["source_date"].isoformat()
                    if hasattr(d["source_date"], "isoformat")
                    else str(d["source_date"]),
                    "n_chars": len(text),
                    "text": text,
                }
            )

    out = out_dir / "documents"
    out.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("source_id", pa.string()),
            ("ticker", pa.string()),
            ("source_type", pa.string()),
            ("source_date", pa.string()),
            ("n_chars", pa.int64()),
            ("text", pa.string()),
        ]
    )
    path = out / "documents.parquet"
    pq.write_table(pa.Table.from_pylist(docs, schema=schema), path, compression="zstd")

    size = path.stat().st_size
    print("\n--- documents config ---")
    print(f"  documents written        : {stats['documents']}")
    print(f"  total characters         : {stats['chars']:,}")
    print(f"  spans verified resolvable: {stats['spans_verified']:,}")
    if stats["unresolvable_spans"]:
        print(
            f"  ! unresolvable spans     : {stats['unresolvable_spans']} "
            f"across {stats['documents_with_unresolvable_spans']} skipped documents"
        )
    print(f"  parquet on disk          : {size / 1e6:.1f} MB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=Path, default=Path("data/processed/ptc_pool_dataset.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/hf_dataset"))
    ap.add_argument("--tickers", nargs="*", default=TICKERS)
    args = ap.parse_args()
    build(args.pool, args.out, args.tickers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
