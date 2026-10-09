"""Generate the figures for the PTC-extraction article.

Reads the published Hugging Face dataset artefacts under ``data/hf_dataset`` and
writes PNGs into ``article/figures/``. Every number in the article comes from
here, so the article and the figures cannot drift apart: re-run this and the
printed STATS block is the source for the prose.

Needs matplotlib, which is not in the base install — it lives in the ``figures``
extra, since nothing in the pipeline or the app imports it:

    uv sync --extra figures

Usage:
    uv run python scripts/make_article_figures.py                 # article draft
    uv run python scripts/make_article_figures.py --out docs/figures   # dataset card

Or without touching the project environment at all:

    uv run --with matplotlib python scripts/make_article_figures.py
"""

import argparse
import collections
import statistics
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq
from matplotlib.patches import Rectangle

# Palette: the validated reference instance, light surface.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

POS = "#2a78d6"  # diverging pole: polarity +1
NEG = "#e34948"  # diverging pole: polarity -1
MANAGEMENT = "#2a78d6"  # categorical slot 1
PRESS = "#eb6834"  # categorical slot 2

# Sequential blue ramp, for the ordered span-match classes.
BLUE_450, BLUE_350, BLUE_250 = "#2a78d6", "#5598e7", "#86b6ef"


def style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": BASELINE,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "figure.dpi": 200,
        }
    )


def load(root: Path) -> list[dict]:
    parts = [pq.read_table(root / "data" / f"{s}.parquet") for s in ("train", "validation", "test")]
    return pa.concat_tables(parts).to_pylist()


def title(ax, text: str, subtitle: str | None = None) -> None:
    """Title above subtitle, both left-aligned to the plot area.

    The title is pushed clear with `pad`, and the subtitle sits in the gap that
    leaves — otherwise matplotlib stacks them at the same y and they overlap.
    """
    ax.set_title(text, loc="left", pad=34 if subtitle else 10, color=INK)
    if subtitle:
        ax.annotate(
            subtitle,
            xy=(0, 1),
            xytext=(0, 12),
            xycoords="axes fraction",
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=10,
            color=INK_2,
        )


# --- figure 1: the interpretive grid --------------------------------------------


def fig_quadrants(rows: list[dict], out: Path) -> None:
    """The 2x2 the whole schema turns on: direction against polarity."""
    counts = collections.Counter((r["direction"], r["polarity"]) for r in rows)
    total = len(rows)

    gloss = {
        ("precursor", 1): "A tailwind that has not\nlanded yet",
        ("precursor", -1): "A risk flagged before\nit bites",
        ("consequence", 1): "Credit claimed for\nsomething already done",
        ("consequence", -1): "Damage already\nrecorded",
    }
    label = {
        ("precursor", 1): "Promise",
        ("precursor", -1): "Warning",
        ("consequence", 1): "Achievement",
        ("consequence", -1): "Damage",
    }

    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.axis("off")

    for col, pol in enumerate((1, -1)):
        for row, direction in enumerate(("precursor", "consequence")):
            n = counts[(direction, pol)]
            colour = POS if pol == 1 else NEG
            x, y = col, 1 - row
            # 2px-equivalent surface gap between cells: inset the patch.
            ax.add_patch(
                Rectangle(
                    (x + 0.03, y + 0.03),
                    0.94,
                    0.94,
                    facecolor=colour,
                    alpha=0.07,
                    edgecolor="none",
                )
            )
            ax.add_patch(
                Rectangle((x + 0.03, y + 0.03), 0.94, 0.035, facecolor=colour, edgecolor="none")
            )
            ax.text(
                x + 0.09,
                y + 0.80,
                label[(direction, pol)],
                fontsize=12,
                fontweight="bold",
                color=colour,
            )
            ax.text(x + 0.09, y + 0.56, f"{n:,}", fontsize=27, color=INK)
            ax.text(
                x + 0.09,
                y + 0.45,
                f"{100 * n / total:.0f}% of all claims",
                fontsize=10,
                color=MUTED,
            )
            ax.text(
                x + 0.09,
                y + 0.16,
                gloss[(direction, pol)],
                fontsize=10.5,
                color=INK_2,
                linespacing=1.5,
            )

    for col, text in enumerate(("polarity  +1", "polarity  −1")):  # noqa: RUF001 (typographic minus / en dash)
        ax.text(col + 0.5, 2.06, text, ha="center", fontsize=11, color=INK, fontweight="bold")
    for row, text in enumerate(("Dir.\nPRECURSOR", "Dir.\nCONSEQUENCE")):
        ax.text(
            -0.04,
            1.5 - row,
            text,
            ha="right",
            va="center",
            fontsize=11,
            color=INK,
            fontweight="bold",
            linespacing=1.6,
        )

    fig.suptitle(
        "Every claim lands in one of four boxes",
        x=0.008,
        y=0.985,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.008,
        0.925,
        "Two independent axes: when the effect happens, and which way it pushes.",
        ha="left",
        fontsize=10.5,
        color=INK_2,
    )
    fig.tight_layout(rect=(0.05, 0.02, 1, 0.88))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# --- figure 2: why a single mood score loses the plot ----------------------------


def fig_washout(rows: list[dict], out: Path) -> tuple[float, float]:
    """Per-document net polarity: the number a sentiment model would report."""
    bydoc = collections.defaultdict(list)
    for r in rows:
        bydoc[r["source_id"]].append(r)
    docs = {k: v for k, v in bydoc.items() if len(v) >= 10}
    nets = [sum(r["polarity"] for r in v) / len(v) for v in docs.values()]
    both = sum(1 for v in docs.values() if {1, -1} <= {r["polarity"] for r in v})

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.hist(nets, bins=32, range=(-1, 1), color=POS, alpha=0.85, edgecolor=SURFACE, linewidth=1.2)
    ax.axvline(0, color=BASELINE, linewidth=1.2)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    ax.set_xlabel("Average polarity of a document's claims", fontsize=10.5)
    ax.set_ylabel("Documents", fontsize=10.5)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(
        ["−1\nall negative", "−0.5", "0\nbalanced", "+0.5", "+1\nall positive"],  # noqa: RUF001 (typographic minus / en dash)
        fontsize=9.5,
    )
    ax.tick_params(length=0)

    title(
        ax,
        "Almost no document is only positive / only negative",
        f"{len(docs)} documents with 10+ claims, and {100 * both / len(docs):.0f}% of them argue\nboth ways at once (positive & negative).",
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return 100 * both / len(docs), statistics.median(nets)


# --- figure 3: what the corpus actually covers -----------------------------------


def fig_timeline(rows: list[dict], out: Path) -> None:
    years = sorted({r["source_date"][:4] for r in rows})
    series = {}
    for corpus in ("management", "press"):
        c = collections.Counter(r["source_date"][:4] for r in rows if r["source_type"] == corpus)
        series[corpus] = [c.get(y, 0) for y in years]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    for corpus, colour in (("management", MANAGEMENT), ("press", PRESS)):
        ax.plot(
            years,
            series[corpus],
            color=colour,
            linewidth=2,
            marker="o",
            markersize=5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=corpus,
        )
        ax.annotate(
            corpus,
            xy=(len(years) - 1, series[corpus][-1]),
            xytext=(8, 0),
            textcoords="offset points",
            color=colour,
            fontsize=10.5,
            fontweight="bold",
            va="center",
        )
    ax.legend(frameon=False, loc="upper left", fontsize=10, labelcolor=INK_2)
    ax.set_ylabel("Claims extracted", fontsize=10.5)
    ax.tick_params(length=0)
    ax.set_xlim(-0.4, len(years) + 0.9)
    ax.annotate(
        "2026 partial (to July)",
        xy=(len(years) - 1, 0),
        xytext=(0, -44),
        textcoords="offset points",
        fontsize=9,
        color=MUTED,
        ha="center",
        annotation_clip=False,
    )
    title(
        ax,
        "The corpus is mostly made of corporate publications, and is thin on press",
        "Claims by publication year. Filings run 2012–2026; the news index only reaches back "  # noqa: RUF001 (typographic minus / en dash)
        "a few years.",
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# --- figure 4: can you check the claim against the source? -----------------------


def fig_provenance(rows: list[dict], out: Path) -> None:
    order = ["verbatim", "whitespace", "unicode", None]
    names = {
        "verbatim": "exact match",
        "whitespace": "match after line-wrap repair",
        "unicode": "match after character repair",
        None: "not located",
    }
    colours = {"verbatim": BLUE_450, "whitespace": BLUE_350, "unicode": BLUE_250, None: "#dedcd4"}

    fig, ax = plt.subplots(figsize=(9.5, 3.5))
    for i, corpus in enumerate(("management", "press")):
        sub = [r for r in rows if r["source_type"] == corpus]
        counts = collections.Counter(r["span_match"] for r in sub)
        left = 0.0
        for key in order:
            share = 100 * counts.get(key, 0) / len(sub)
            if share <= 0:
                continue
            ax.barh(
                i, share, left=left, height=0.38, color=colours[key], edgecolor=SURFACE, linewidth=2
            )
            if share >= 7:
                # Ink on the pale "not located" fill; surface on the saturated blues.
                on_pale = key in (None, "unicode")
                ax.text(
                    left + share / 2,
                    i,
                    f"{share:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color=INK_2 if on_pale else SURFACE,
                    fontweight="bold",
                )
            left += share
    ax.set_yticks([0, 1])
    ax.set_yticklabels(
        ["management\n32,157 claims", "press\n3,425 claims"],
        fontsize=10,
        color=INK,
        linespacing=1.5,
    )
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"], fontsize=9.5)
    ax.tick_params(length=0)
    ax.spines["left"].set_visible(False)

    handles = [Rectangle((0, 0), 1, 1, color=colours[k]) for k in order]
    ax.legend(
        handles,
        [names[k] for k in order],
        frameon=False,
        fontsize=9.5,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.62),
        ncols=4,
        labelcolor=INK_2,
        handlelength=1.1,
        handleheight=1.1,
        columnspacing=1.4,
    )
    title(
        ax,
        "How often the quote can be pinned back to the source",
        "PDF filings reflow their text, so exact matching fails far more often there than "
        "on clean news HTML.",
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/hf_dataset"))
    p.add_argument("--out", type=Path, default=Path("article/figures"))
    args = p.parse_args()

    style()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = load(args.data)

    fig_quadrants(rows, args.out / "ptc-quadrants.png")
    both, median_net = fig_washout(rows, args.out / "ptc-sentiment-washout.png")
    fig_timeline(rows, args.out / "ptc-timeline.png")
    fig_provenance(rows, args.out / "ptc-provenance.png")

    # Everything the article asserts, printed so prose and figures stay in step.
    total = len(rows)
    q = collections.Counter((r["direction"], r["polarity"]) for r in rows)
    prec = sum(v for k, v in q.items() if k[0] == "precursor")
    cons = total - prec
    prec_pos = q[("precursor", 1)]
    cons_pos = q[("consequence", 1)]
    print("\n--- STATS ---")
    print(f"claims {total:,}   documents {len({r['source_id'] for r in rows}):,}")
    print(f"precursor positive   {100 * prec_pos / prec:.1f}%  (n={prec:,})")
    print(f"consequence positive {100 * cons_pos / cons:.1f}%  (n={cons:,})")
    print(f"documents containing both polarities: {both:.1f}%   median net {median_net:+.2f}")
    print(f"figures written to {args.out}/")


if __name__ == "__main__":
    main()
