"""Step 2 — Claims: an AI reads every document and extracts cause-and-effect statements."""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import data_io
import theme
import ui

st.set_page_config(page_title="Claims — Signal Intelligence", page_icon="📡", layout="wide")

ticker = ui.get_ticker()
ptcs = data_io.load_ptcs(ticker)

ui.step_header(
    2,
    4,
    "🧩 Claims",
    "An AI reads every document, sentence by sentence, and pulls out each **cause-and-effect "
    "claim** it finds — keeping the exact quote as proof. One claim = one story about something "
    "affecting the business.",
)

if ptcs.empty:
    st.warning("No claims available for this company.")
    st.stop()

# --- anatomy of one claim ---
st.subheader("Anatomy of a claim")
st.caption(
    "A real example, picked from the data below. Every claim carries the same five ingredients."
)

example_pool = ptcs[(ptcs["direction"] == "precursor") & (ptcs["raw_text"].str.len() > 80)]
example = example_pool.iloc[0] if not example_pool.empty else ptcs.iloc[0]

with st.container(border=True):
    st.markdown(f"##### “{example['mechanism']}”")
    a, b, c, d = st.columns(4)
    a.markdown(
        f"**Direction**\n\n{'🔮 Forward-looking' if example['direction'] == 'precursor' else '📜 Already happened'}"
    )
    b.markdown(
        f"**Expected effect**\n\n{'👍 Positive' if example['polarity'] > 0 else '👎 Negative'} for the business"
    )
    c.markdown(
        f"**Who said it**\n\n{'🏢 The company' if example['source_type'] == 'management' else '📰 The press'}"
    )
    d.markdown(f"**When**\n\n{example['source_date']:%d %b %Y}")
    st.markdown(f"**The exact quote it came from:**\n> {str(example['raw_text'])[:400]}")

with st.expander("🔍 How this works — the extraction rules"):
    st.markdown(
        """
- The AI (a large language model) reads each document in overlapping chunks and returns
  claims in a **strict format**; anything malformed is thrown away, never "repaired".
- **One claim = one mechanism.** A sentence with two causal stories becomes two claims.
- **Direction** is judged *relative to the publication date*: is the effect still ahead
  (forward-looking) or already realised? Forward-looking claims are the interesting ones
  for markets — they arrive *before* any impact should show in the price.
- The **verbatim quote** is always kept, so every claim can be audited against its source.
- On this corpus the extraction cost about **$12 of AI compute** — the pipeline now runs
  on a model 5x cheaper for future companies.
"""
    )

st.divider()

# --- claims over time ---
st.subheader("Claims through time")
st.caption(
    "Forward-looking claims (the ones that could anticipate market moves) vs claims about "
    "things that already happened. Company reports skew recent; press coverage reaches back further."
)

by_time = ptcs.dropna(subset=["source_date"]).copy()
by_time["year"] = by_time["source_date"].dt.year
counts = by_time.groupby(["year", "direction"]).size().unstack(fill_value=0)

fig = go.Figure()
if "precursor" in counts.columns:
    fig.add_trace(
        go.Bar(
            x=counts.index,
            y=counts["precursor"],
            name="Forward-looking",
            marker={"color": theme.BLUE},
        )
    )
if "consequence" in counts.columns:
    fig.add_trace(
        go.Bar(
            x=counts.index,
            y=counts["consequence"],
            name="Already happened",
            marker={"color": theme.MAGENTA},
        )
    )
fig.update_layout(
    **theme.plotly_layout(
        height=320,
        barmode="stack",
        xaxis_title="Year",
        yaxis_title="Claims",
        legend={"orientation": "h", "y": 1.15},
    )
)
st.plotly_chart(fig, width="stretch")

st.divider()

# --- the claim galaxy (3D) ---
projection = data_io.load_projection(ticker)
summaries = data_io.load_cluster_summaries(ticker)
if not projection.empty:
    st.divider()
    st.subheader("The claim galaxy")
    st.caption(
        "Every claim, placed in space by **meaning**: the AI turns each claim into a list of "
        "numbers (an *embedding*), and this view squeezes those down to 3D so that claims about "
        "the same thing land near each other. Colour shows who tells it. Rotate, zoom, hover."
    )

    _PRESENCE = {
        "triangulated": (theme.BLUE, "Both company & press"),
        "insider_only": (theme.GREEN, "Company only"),
        "outsider_only": (theme.MAGENTA, "Press only"),
    }
    highlight = None
    if not summaries.empty:
        options = ["(show everything)"] + [
            f"{r.n_ptcs:>4} · {str(r.summary)[:80]}" for r in summaries.itertuples()
        ]
        pick = st.selectbox(
            "Highlight a recurring theme (LLM-generalized from each cluster's claims):",
            options,
        )
        if pick != "(show everything)":
            highlight = summaries.iloc[options.index(pick) - 1]["canonical_id"]

    fig = go.Figure()
    if highlight is not None:
        base = projection[projection["canonical_id"] != highlight]
        hot = projection[projection["canonical_id"] == highlight]
        fig.add_trace(
            go.Scatter3d(
                x=base["x"],
                y=base["y"],
                z=base["z"],
                mode="markers",
                marker={"size": 2, "color": theme.GRIDLINE, "opacity": 0.25},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=hot["x"],
                y=hot["y"],
                z=hot["z"],
                mode="markers",
                marker={
                    "size": 5,
                    "color": theme.BLUE,
                    "opacity": 0.95,
                    "line": {"color": theme.SURFACE, "width": 1},
                },
                text=hot["mechanism"],
                hovertemplate="%{text}<extra></extra>",
                name="Selected theme",
            )
        )
    else:
        for presence, (color, label) in _PRESENCE.items():
            sub = projection[projection["corpus_presence"] == presence]
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter3d(
                    x=sub["x"],
                    y=sub["y"],
                    z=sub["z"],
                    mode="markers",
                    marker={"size": 2.5, "color": color, "opacity": 0.7},
                    text=sub["mechanism"],
                    hovertemplate="%{text}<extra></extra>",
                    name=label,
                )
            )
    fig.update_layout(
        **theme.plotly_layout(
            height=560,
            legend={"orientation": "h", "y": 1.05},
            margin={"l": 0, "r": 0, "t": 30, "b": 0},
        )
    )
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    st.plotly_chart(fig, width="stretch")

    if not summaries.empty:
        st.markdown("##### The biggest recurring themes")
        st.caption(
            "Each row is one cluster of claims, with a one-sentence mechanism an LLM wrote by "
            "reading all the claims in that group. This is the machine's own summary of what the "
            "company and press keep coming back to."
        )
        show_s = summaries.head(20)[["n_ptcs", "corpus_presence", "direction", "summary"]].rename(
            columns={
                "n_ptcs": "Claims",
                "corpus_presence": "Told by",
                "direction": "Type",
                "summary": "Generalized mechanism (LLM)",
            }
        )
        st.dataframe(show_s, width="stretch", hide_index=True)

# --- browse ---
st.divider()
st.subheader("Browse the claims yourself")
f1, f2, f3 = st.columns(3)
who = f1.selectbox("Who said it", ["Everyone", "The company", "The press"])
direction = f2.selectbox("Direction", ["All", "Forward-looking", "Already happened"])
effect = f3.selectbox("Expected effect", ["All", "Positive", "Negative"])

view = ptcs.copy()
if who == "The company":
    view = view[view["source_type"] == "management"]
elif who == "The press":
    view = view[view["source_type"] == "press"]
if direction == "Forward-looking":
    view = view[view["direction"] == "precursor"]
elif direction == "Already happened":
    view = view[view["direction"] == "consequence"]
if effect == "Positive":
    view = view[view["polarity"] > 0]
elif effect == "Negative":
    view = view[view["polarity"] < 0]

st.caption(
    f"{len(view):,} claims match. Showing the most recent 200 — hover cells to read in full."
)
show = view.sort_values("source_date", ascending=False).head(200)
st.dataframe(
    show[["source_date", "source_type", "direction", "polarity", "mechanism", "raw_text"]].rename(
        columns={
            "source_date": "Date",
            "source_type": "Source",
            "direction": "Direction",
            "polarity": "Effect",
            "mechanism": "Claim",
            "raw_text": "Exact quote",
        }
    ),
    width="stretch",
    hide_index=True,
    height=420,
)

ui.next_step("pages/3_Mechanisms.py", "Group claims into mechanisms")
