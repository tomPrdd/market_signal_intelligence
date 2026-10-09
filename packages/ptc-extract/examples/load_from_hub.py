"""Start from a pre-extracted pool instead of paying for extraction.

    pip install 'ptc-extract[hub] @ git+https://github.com/tomPrdd/ptc-extract'
    python examples/load_from_hub.py

DEFAULT_POOL_REPO points at the reference pool,
https://huggingface.co/datasets/tomPrdd/ptc-pool-eu-largecap. Point REPO_ID at
any other dataset repo holding a PTC pool to read that instead.
"""

from collections import Counter

from ptc_extraction.hub import DEFAULT_POOL_REPO, list_hub_pools, load_pool_from_hub

REPO_ID = DEFAULT_POOL_REPO


def main() -> None:
    print("Known pools:")
    for repo_id, description in list_hub_pools().items():
        print(f"  {repo_id} — {description}")
    print()

    # Only forward-looking claims from the companies' own disclosures.
    ptcs = load_pool_from_hub(
        REPO_ID,
        revision="main",
        filters={"source_type": "management", "direction": "precursor"},
    )

    print(f"{len(ptcs)} claims loaded from {REPO_ID}\n")
    print("By polarity:", Counter(p.polarity.name for p in ptcs))
    print("By model:   ", Counter(p.extracted_by for p in ptcs))

    located = sum(1 for p in ptcs if p.span is not None)
    print(f"With character offsets: {located}/{len(ptcs)}")

    for ptc in ptcs[:5]:
        print(f"\n{ptc.source_date}  {ptc.mechanism}")


if __name__ == "__main__":
    main()
