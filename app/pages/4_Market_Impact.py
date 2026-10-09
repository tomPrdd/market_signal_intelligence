"""Step 4 — Market impact: did the stock move unusually after each mechanism was mentioned?"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import data_io
import theme
import ui

st.set_page_config(page_title="Market impact — Signal Intelligence", page_icon="📡", layout="wide")

ticker = ui.get_ticker()
canonicals = data_io.load_canonicals(ticker)
eventstudy = data_io.load_eventstudy(ticker)
profiles = data_io.load_car_profiles(ticker)
prices = data_io.load_prices(ticker)
ptcs = data_io.load_ptcs(ticker)

ui.step_header(
    4,
    4,
    "📈 Market impact",
    "The core question: **after** a forward-looking mechanism is mentioned, does the stock do "
    "anything unusual in the following weeks? Not just *did it go up or down* — did it move "
    "**more than its normal drift**?",
)

robustness = data_io.load_robustness(ticker)

st.error(
    "**Read this before the charts below — we tested this method against itself, and it "
    "did not pass.**\n\n"
    "The honest finding: the *direction* a claim predicts (good or bad for the business) "
    "**does not predict the direction the stock actually moved**. Across all five companies, "
    "mechanisms flagged positive and mechanisms flagged negative are followed by "
    "statistically indistinguishable returns — and where a difference does appear, it points "
    "the wrong way. See **Method & limits → What we found when we tested ourselves**.\n\n"
    "So treat everything below as **an exploratory map, not evidence of predictive power**."
)

with st.expander("🔍 How this works — the event-study idea, in one minute", expanded=False):
    st.markdown(
        """
1. Every time a mechanism is mentioned in a document, that publication date becomes an **event**.
2. The stock's **normal daily drift** is measured over the ~6 months *before* the event.
3. For each of the **30 trading days after** the event, the actual return is compared to that
   normal drift. The running difference is the **abnormal return** — the part of the move
   that the stock's own habit doesn't explain.
4. All events for the same mechanism are averaged. If the average abnormal move is clearly
   different from zero (and consistent across events), that mechanism *may* carry signal.

Two honest caveats, expanded in *Method & limits*: many mechanisms have only a handful of
events (small samples make noisy averages), and an unusual move *after* a mention is a
**correlation**, not proof the mention caused it.
"""
    )

tested = (
    eventstudy[eventstudy["n_events"] >= 2].dropna(subset=["mean_car"])
    if not eventstudy.empty
    else pd.DataFrame()
)

if tested.empty:
    st.warning("No mechanisms have enough dated events for a market test.")
    st.stop()

st.divider()

# --- impact map ---
st.subheader("The impact map")
st.caption(
    "Each dot is one forward-looking mechanism. Left = stock tended to fall after mentions; "
    "right = tended to rise. Higher = mentioned more often. Colour shows **who tells the story**: "
    "🔵 blue = both company and press (triangulated), 🟢 green = company only, "
    "🩷 pink = press only. Hover any dot for details."
)

merged = tested.merge(canonicals, on="canonical_id", suffixes=("", "_c"))
_presences = set(merged["corpus_presence"].unique())
if _presences == {"insider_only"}:
    st.info(
        "For this company the analysis currently uses **company documents only** — no press "
        'corpus — so every mechanism is green ("company only"). Add press coverage to unlock '
        "the blue/pink triangulation, as on Michelin."
    )
fig = go.Figure()
for presence, color in theme.CORPUS_PRESENCE_COLORS.items():
    sub = merged[merged["corpus_presence"] == presence]
    if sub.empty:
        continue
    fig.add_trace(
        go.Scatter(
            x=sub["mean_car"] * 100,
            y=sub["n_ptcs"],
            mode="markers",
            name=theme.PRESENCE_LABELS[presence],
            marker={
                "color": color,
                "size": 12,
                "opacity": 0.85,
                "line": {"color": theme.SURFACE, "width": 2},
            },
            customdata=sub[["representative_mechanism", "p_value", "n_events"]],
            hovertemplate=(
                "%{customdata[0]}<br>"
                "avg abnormal move: %{x:.1f}% over 30 days · "
                "%{customdata[2]} events · p=%{customdata[1]:.3f}<extra></extra>"
            ),
        )
    )
fig.add_vline(x=0, line_color=theme.BASELINE, line_width=1)
fig.update_layout(
    **theme.plotly_layout(
        height=440,
        xaxis_title="Average abnormal stock move in the 30 trading days after a mention (%)",
        yaxis_title="How often the mechanism appears (claims)",
        legend={"orientation": "h", "y": 1.14},
    )
)
st.plotly_chart(fig, width="stretch")

st.divider()

# --- drill into one mechanism ---
st.subheader("Inspect one mechanism against the stock")

merged["_label"] = merged.apply(data_io.canonical_label, axis=1)
merged["_abs"] = merged["mean_car"].abs()
options = merged.sort_values("_abs", ascending=False)
selected = st.selectbox("Mechanism (sorted by size of market reaction)", options["_label"].tolist())
row = merged[merged["_label"] == selected].iloc[0]
cid = row["canonical_id"]

m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Avg abnormal move (+30 days)",
    f"{row['mean_car']:+.1%}",
    help="Average cumulative abnormal return across all this mechanism's events.",
)
m2.metric(
    "Events", int(row["n_events"]), help="How many dated mentions had a full 30-day price window."
)
m3.metric(
    "Consistency",
    f"{row['positive_pct']:.0f}% up" if pd.notna(row["positive_pct"]) else "—",
    help="Share of events where the abnormal move was positive.",
)
m4.metric(
    "p-value (raw)",
    f"{row['p_value']:.3f}" if pd.notna(row["p_value"]) else "—",
    help="Uncorrected. With hundreds of mechanisms tested at once, some will look "
    "significant by luck — see the corrected figures below.",
)

# --- robustness under a correct specification ---
rob = robustness[robustness["canonical_id"] == cid] if not robustness.empty else pd.DataFrame()
if not rob.empty:
    r = rob.iloc[0]
    st.markdown("**Same mechanism, tested properly:**")
    r1, r2, r3 = st.columns(3)
    r1.metric(
        "Independent dates",
        int(r["n_dates"]),
        help="The events above often come from the SAME document. This is the number of "
        "distinct publication dates — the real number of independent observations.",
    )
    r2.metric(
        "CAR, market model",
        f"{r['mean_car_market']:+.1%}",
        help="Abnormal return after removing the CAC 40's move (alpha + beta x market), "
        "the standard specification. The headline figure above removes only the "
        "stock's own past average, so it still contains market movement.",
    )
    bh = r["p_bh_market"]
    r3.metric(
        "p-value, corrected",
        f"{bh:.3f}" if pd.notna(bh) else "—",
        help="Benjamini-Hochberg correction for testing hundreds of mechanisms at once. "
        "This is the number to trust.",
    )
    if pd.notna(r["n_dates"]) and int(r["n_dates"]) < 3:
        st.warning(
            f"⚠️ This mechanism's {int(row['n_events'])} 'events' come from only "
            f"{int(r['n_dates'])} distinct publication date(s) — its statistics are not "
            "meaningful."
        )

# CAR trajectory
prof = profiles[profiles["canonical_id"] == cid] if not profiles.empty else pd.DataFrame()
if not prof.empty:
    x = prof["rel_day"]
    mean = prof["mean_car"] * 100
    fig = go.Figure()
    if prof["sem_car"].notna().all() and int(prof["n_events"].iloc[0]) >= 2:
        upper = (prof["mean_car"] + 1.96 * prof["sem_car"]) * 100
        lower = (prof["mean_car"] - 1.96 * prof["sem_car"]) * 100
        fig.add_trace(
            go.Scatter(
                x=pd.concat([x, x[::-1]]),
                y=pd.concat([upper, lower[::-1]]),
                fill="toself",
                fillcolor="rgba(42,120,214,0.15)",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean,
            mode="lines",
            line={"color": theme.BLUE, "width": 2},
            hovertemplate="day +%{x}: %{y:.1f}%<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_hline(y=0, line_color=theme.BASELINE, line_width=1)
    fig.update_layout(
        **theme.plotly_layout(
            height=340,
            xaxis_title="Trading days after the mention",
            yaxis_title="Abnormal move (%)",
        )
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "The average day-by-day abnormal move after this mechanism is mentioned, with a 95% "
        "confidence band. A wide band means few events — read with caution. The zero line is "
        '"the stock just did its usual thing".'
    )

# mentions vs price
matrix = data_io.load_mention_matrix(ticker)
if not matrix.empty and cid in matrix.columns and not prices.empty:
    mentions = matrix[cid]
    mentions = mentions[mentions > 0]
    price_col = "adj_close" if "adj_close" in prices.columns else "close"
    if not mentions.empty:
        start = mentions.index.min() - pd.offsets.QuarterEnd()
        price = prices[price_col].loc[prices.index >= start]

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.08
        )
        fig.add_trace(
            go.Scatter(
                x=price.index,
                y=price,
                mode="lines",
                name="Share price",
                line={"color": theme.BLUE, "width": 2},
                hovertemplate="%{x|%d %b %Y}: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=mentions.index,
                y=mentions,
                name="Mentions",
                marker={"color": theme.GREEN},
                hovertemplate="%{x|%b %Y}: %{y} mentions<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.update_layout(
            **theme.plotly_layout(height=460, showlegend=False, hovermode="x unified")
        )
        fig.update_yaxes(title_text="Share price", row=1, col=1, gridcolor=theme.GRIDLINE)
        fig.update_yaxes(title_text="Mentions / quarter", row=2, col=1, gridcolor=theme.GRIDLINE)
        fig.update_xaxes(gridcolor=theme.GRIDLINE)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Context view: the share price (top) with the quarters in which this mechanism was "
            "being talked about (bottom). Same timeline, separate scales."
        )

# the receipts
with st.expander("📜 Read the underlying quotes (the receipts)"):
    try:
        member_ids = set(json.loads(row["source_ids"]))
    except (TypeError, ValueError):
        member_ids = set()
    members = (
        ptcs[ptcs["source_id"].isin(member_ids)].sort_values("source_date")
        if member_ids
        else pd.DataFrame()
    )
    if members.empty:
        st.info("No quotes available.")
    for _, p in members.head(12).iterrows():
        src = "🏢 Company" if p["source_type"] == "management" else "📰 Press"
        date_str = f"{p['source_date']:%d %b %Y}" if pd.notna(p["source_date"]) else "undated"
        st.markdown(f"**{date_str} · {src}**\n> {str(p['raw_text'])[:400]}")

st.divider()
ui.link(
    "pages/5_Method_and_Limits.py", "Before you conclude anything: Method & limits →", icon="🔬"
)
