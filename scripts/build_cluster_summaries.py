"""Generate an LLM-written generalized mechanism for each sizeable cluster.

For the top clusters (by number of claims) of a company, read the member
claims' mechanism sentences and ask Nova to distil them into ONE general
mechanism statement. Writes app/demo/{ticker}_cluster_summaries.csv
(canonical_id, n_ptcs, direction, polarity, corpus_presence, summary).

An LLM cannot read embeddings directly; it reads the claim TEXTS that a
cluster groups together and verbalizes their common mechanism.

    uv run python scripts/build_cluster_summaries.py ML.PA --top 40
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ptc_extraction import BedrockBackend

PROCESSED = Path("data/processed")
DEMO = Path("app/demo")

_SYSTEM = (
    "You are a financial analyst. You are given several related causal claims that an "
    "algorithm has grouped into one cluster. Write ONE concise sentence (max 30 words) that "
    "generalizes the common causal mechanism they share. State the driver and its effect on the "
    "company. Do not list the claims; synthesize. Reply with the sentence only, no preamble."
)


def _ptc_texts(ticker: str) -> dict[str, str]:
    texts: dict[str, str] = {}
    pool = DEMO / f"{ticker}_ptc_pool.jsonl"
    if not pool.exists():
        pool = PROCESSED / "ptc_pool.jsonl"
    for line in pool.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(r.get("source_id", "")).startswith(f"{ticker}-"):
            texts[r["source_id"] + "|" + r["mechanism"][:20]] = r["mechanism"]
    return texts


def build_summaries(ticker: str, top: int = 40) -> None:
    canon = pd.read_csv(PROCESSED / f"{ticker}_canonicals.csv")
    if "is_noise" in canon.columns:
        canon = canon[~canon["is_noise"].astype(str).str.lower().eq("true")]
    canon = canon.sort_values("n_ptcs", ascending=False).head(top)

    # map source_id -> mechanism text
    pool = DEMO / f"{ticker}_ptc_pool.jsonl"
    id_to_mech: dict[str, str] = {}
    if pool.exists():
        for line in pool.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            id_to_mech[r["source_id"]] = r["mechanism"]

    backend = BedrockBackend()
    rows = []
    for _, c in canon.iterrows():
        try:
            sids = json.loads(c["source_ids"])
        except (TypeError, ValueError):
            sids = []
        mechs = [id_to_mech[s] for s in sids if s in id_to_mech][:25]
        if not mechs:
            mechs = [str(c.get("representative_mechanism", ""))]
        user = "Claims in this cluster:\n- " + "\n- ".join(mechs)
        try:
            summary = backend.complete(_SYSTEM, user, max_tokens=80).strip().strip('"')
        except Exception as e:
            summary = str(c.get("representative_mechanism", ""))
            print(f"  ! {c['canonical_id']}: fell back ({e})")
        rows.append(
            {
                "canonical_id": c["canonical_id"],
                "n_ptcs": int(c["n_ptcs"]),
                "direction": c["direction"],
                "polarity": int(c["polarity"]),
                "corpus_presence": c.get("corpus_presence", ""),
                "summary": summary,
            }
        )

    out = pd.DataFrame(rows)
    DEMO.mkdir(parents=True, exist_ok=True)
    dest = DEMO / f"{ticker}_cluster_summaries.csv"
    out.to_csv(dest, index=False)
    u = backend.usage
    cost = u["input_tokens"] / 1e6 * 0.80 + u["output_tokens"] / 1e6 * 3.20
    print(f"{ticker}: {len(rows)} cluster summaries -> {dest}  (~${cost:.2f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()
    for t in args.tickers:
        build_summaries(t, top=args.top)


if __name__ == "__main__":
    main()
