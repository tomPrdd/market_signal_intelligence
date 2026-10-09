"""Demo: extract PTCs from a single text file and persist to a JSONL pool.

Example:
python scripts/demo_ptc_extraction.py data/raw/sample_press_michelin.txt --source-id SRC123 --source-type management --source-date 2025-01-01 --output data/processed/ptc_pool.jsonl
"""

import argparse
from datetime import date
from pathlib import Path

from ptc_extraction import BedrockBackend, PTCExtractor, SourceType, load_pool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path, help="Path to a text file to process.")
    parser.add_argument("--source-id", required=True, help="Unique identifier for the source.")
    parser.add_argument(
        "--source-type",
        choices=["management", "press"],
        required=True,
    )
    parser.add_argument(
        "--source-date",
        required=True,
        help="Publication date of the document (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/ptc_pool.jsonl"),
        help="Output JSONL pool path.",
    )
    args = parser.parse_args()

    text = args.input_file.read_text(encoding="utf-8")
    source_date = date.fromisoformat(args.source_date)
    source_type = SourceType(args.source_type)

    extractor = PTCExtractor(backend=BedrockBackend())

    print(f"Extracting PTCs from {args.input_file} ({len(text)} chars)...")
    ptcs = extractor.extract_from_text(
        text=text,
        source_id=args.source_id,
        source_type=source_type,
        source_date=source_date,
    )

    print(f"Extracted {len(ptcs)} PTCs:")
    for ptc in ptcs:
        print(f"  [{ptc.direction.value:11}] (pol={ptc.polarity.value:+d}) {ptc.mechanism[:80]}")

    from ptc_extraction import append_to_pool

    append_to_pool(args.output, ptcs)
    print(f"\nAppended to {args.output}. Pool now has {len(load_pool(args.output))} PTCs total.")


if __name__ == "__main__":
    main()
