"""Bundle one ticker's pipeline artefacts into app/demo/ for a self-contained app.

The deployed demo app ships with its data baked in — no S3 or network calls at
runtime. Run this after a pipeline run to refresh (or add) a company:

    uv run python scripts/build_demo_bundle.py ML.PA

Produces, under app/demo/:
    {ticker}_canonicals.csv        (as produced by the pipeline)
    {ticker}_eventstudy.csv
    {ticker}_car_profiles.csv
    {ticker}_mention_matrix.csv
    {ticker}_prices.csv
    {ticker}_ptc_pool.jsonl        (pool filtered to this ticker)
    {ticker}_sources.csv           (document inventory built from data/raw/{ticker}/)
"""

import argparse
import csv
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from signal_intelligence.ingestion import (
    _extract_date_from_stem,
    _extract_date_from_text,
)

DEMO_DIR = Path("app/demo")
PROCESSED = Path("data/processed")
MARKET = Path("data/market")
RAW = Path("data/raw")

_EXCERPT_CHARS = 280

# Public S3 prefix where the analysis documents are mirrored (GetObject is
# public on this prefix only — the rest of the bucket stays private).
# Set SI_PUBLIC_SOURCES_BASE to the public origin that serves the mirrored
# management documents, e.g. a CloudFront distribution or custom domain in
# front of the bucket. When it is unset the bundle omits download links
# entirely and the app renders the inventory without a Download column.
PUBLIC_SOURCES_BASE = os.getenv("SI_PUBLIC_SOURCES_BASE", "").rstrip("/")

# Only management documents are redistributable. Those are the companies' own
# regulatory filings, published so that they are read and quoted. Press articles
# are third-party journalism: they are listed in the inventory so the corpus
# stays fully traceable, but the full text is not served. They live under a
# private S3 prefix and carry no download link.
DOWNLOADABLE_SOURCE_TYPES = frozenset({"management"})


_HASH_PREFIX_RE = re.compile(r"^[a-z0-9]{16,}[-_]", re.IGNORECASE)
_DATE_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-")


def _humanize_title(stem: str) -> str:
    """Turn a management filename stem into a clean, human-readable title.

    Strips a leading CDN hash prefix and/or YYYY-MM-DD date, converts
    separators to spaces, title-cases, and restores known acronyms.
    e.g. 'du5l2qtng...-2025-annual-results-presentation' -> 'Annual Results 2025 Presentation'
         '2019-03-25-form-20-f-2018' -> 'Form 20-F 2018'
    """
    s = _HASH_PREFIX_RE.sub("", stem)
    s = _DATE_PREFIX_RE.sub("", s)
    s = s.replace("gdelt-", "").replace("_", " ").replace("-", " ").strip()
    s = s.title()
    s = re.sub(r"\b20 F\b", "20-F", s)
    s = re.sub(r"\b6 K\b", "6-K", s)
    s = re.sub(r"\b10 ([KQ])\b", r"10-\1", s)
    s = re.sub(r"\bUrd\b", "URD", s)
    s = re.sub(r"\bDeu\b", "DEU", s)
    s = re.sub(r"\b([HQ])([1-4])\b", r"\1\2", s)
    s = re.sub(r"\bFr\b", "(FR)", s)
    s = re.sub(r"\s+En\b", "", s)  # drop a trailing language tag
    return " ".join(s.split()) or stem


def _doc_inventory(ticker: str) -> list[dict]:
    """One row per raw document: type, date, name, size, short excerpt."""
    rows = []
    for source_type in ("management", "press"):
        d = RAW / ticker / source_type
        if not d.exists():
            continue
        for path in sorted(d.iterdir()):
            if path.suffix.lower() not in {".pdf", ".txt", ".html", ".htm"}:
                continue
            doc_date = _extract_date_from_stem(path.stem)
            excerpt = ""
            title = _humanize_title(path.stem)
            if path.suffix.lower() == ".txt":
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                    lines = text.splitlines()
                    if lines:
                        title = lines[0][:120]
                    body = "\n".join(lines[1:]).strip()
                    excerpt = " ".join(body.split())[:_EXCERPT_CHARS]
                    if doc_date is None:
                        doc_date = _extract_date_from_text(text)
                except Exception:
                    pass
            from urllib.parse import quote

            rows.append(
                {
                    "source_type": source_type,
                    "date": doc_date.isoformat() if doc_date else "",
                    "title": title,
                    "filename": path.name,
                    "format": path.suffix.lstrip(".").upper(),
                    "size_kb": round(path.stat().st_size / 1024),
                    "excerpt": excerpt,
                }
            )
            if PUBLIC_SOURCES_BASE:
                rows[-1]["download_url"] = (
                    f"{PUBLIC_SOURCES_BASE}/{quote(ticker)}/{source_type}/{quote(path.name)}"
                    if source_type in DOWNLOADABLE_SOURCE_TYPES
                    else ""
                )
    rows.sort(key=lambda r: (r["source_type"], r["date"]))
    return rows


def build_bundle(ticker: str) -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    copied = []
    for name in (
        f"{ticker}_canonicals.csv",
        f"{ticker}_eventstudy.csv",
        f"{ticker}_car_profiles.csv",
        f"{ticker}_mention_matrix.csv",
        f"{ticker}_granger.csv",
        # note: {ticker}_aligned.csv is intentionally NOT bundled — the app
        # never loads it (Granger reads granger.csv), and it is large.
    ):
        src = PROCESSED / name
        if src.exists():
            shutil.copy2(src, DEMO_DIR / name)
            copied.append(name)

    prices = MARKET / f"{ticker}_prices.csv"
    if prices.exists():
        shutil.copy2(prices, DEMO_DIR / prices.name)
        copied.append(prices.name)

    pool_src = PROCESSED / "ptc_pool.jsonl"
    if pool_src.exists():
        out = DEMO_DIR / f"{ticker}_ptc_pool.jsonl"
        n = 0
        with open(pool_src, encoding="utf-8") as fin, open(out, "w", encoding="utf-8") as fout:
            for line in fin:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(rec.get("source_id", "")).startswith(f"{ticker}-"):
                    fout.write(line)
                    n += 1
        copied.append(f"{ticker}_ptc_pool.jsonl ({n} PTCs)")

    inventory = _doc_inventory(ticker)
    if inventory:
        with open(DEMO_DIR / f"{ticker}_sources.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(inventory[0].keys()))
            writer.writeheader()
            writer.writerows(inventory)
        copied.append(f"{ticker}_sources.csv ({len(inventory)} documents)")

    print(f"Demo bundle for {ticker}:")
    for c in copied:
        print("  +", c)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticker", help="Ticker with completed pipeline artefacts (e.g. ML.PA)")
    args = parser.parse_args()
    build_bundle(args.ticker)


if __name__ == "__main__":
    main()
