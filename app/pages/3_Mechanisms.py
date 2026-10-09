"""Step 3 — Mechanisms: claims telling the same story are grouped together."""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import data_io
import theme
import ui

st.set_page_config(page_title="Mechanisms — Signal Intelligence", page_icon="📡", layout="wide")

ticker = ui.get_ticker()
canonicals = data_io.load_canonicals(ticker)
ptcs = data_io.load_ptcs(ticker)

ui.step_header(
    3,
    4,
    "🕸️ Mechanisms",
    "Thousands of claims are too many to reason about. Claims that tell **the same underlying "
    "story** — in different words, from different documents — are grouped into one "
    "**mechanism**. This is the map of everything said to drive the business.",
)

if canonicals.empty:
    st.warning("No mechanisms available for this company.")
    st.stop()

n_claims = len(ptcs)
n_mechs = len(canonicals)
c1, c2, c3 = st.columns(3)
c1.metric("Claims in", f"{n_claims:,}")
c2.metric("Mechanisms out", f"{n_mechs:,}")
biggest = canonicals.sort_values("n_ptcs", ascending=False).iloc[0]
c3.metric("Biggest mechanism", f"{int(biggest['n_ptcs'])} claims")

with st.expander("🔍 How this works — grouping without cheating"):
    st.markdown(
        """
- Each claim is converted into a list of numbers (an *embedding*) that captures its meaning —
  claims about rubber prices land near other claims about rubber prices, whatever the wording.
- A clustering algorithm (HDBSCAN) then finds **groups of nearby claims**, separately for
  company documents and for press articles.
- Two safety rules: a group is never allowed to mix **opposite expected effects**
  (positive vs negative), nor **forward-looking with already-happened** claims.
- Finally, company groups and press groups are compared: a mechanism discussed by **both**
  voices is tagged *triangulated* — generally more credible than a story only one side tells.
- No human decides the groups; the structure emerges from the text itself.
"""
    )

st.divider()

# --- the mechanism map: biggest stories ---
st.subheader("The biggest recurring stories")
st.caption(
    "The 20 most-repeated mechanisms. Bar length = number of claims in the group. "
    "Blue: expected positive effect. Red: expected negative effect."
)

top = canonicals.sort_values("n_ptcs", ascending=False).head(20).copy()
top["short"] = top.apply(lambda r: data_io.canonical_label(r, max_len=80), axis=1)
top = top.iloc[::-1]  # horizontal bars read top-down

fig = go.Figure(
    go.Bar(
        x=top["n_ptcs"],
        y=top["short"],
        orientation="h",
        marker={"color": [theme.BLUE if p > 0 else theme.RED for p in top["polarity"]]},
        customdata=top[["direction", "corpus_presence"]],
        hovertemplate="%{y}<br>%{x} claims · %{customdata[0]} · %{customdata[1]}<extra></extra>",
    )
)
fig.update_layout(
    **theme.plotly_layout(
        height=620,
        xaxis_title="Number of claims",
        yaxis={"tickfont": {"size": 11, "color": theme.INK_SECONDARY}},
        margin={"l": 10, "r": 20, "t": 20, "b": 40},
    )
)
st.plotly_chart(fig, width="stretch")

st.divider()

# --- drill into one mechanism ---
st.subheader("Open a mechanism and read its claims")
st.caption("Pick any mechanism to see every claim inside it — with the original quotes.")

canonicals = canonicals.copy()
canonicals["_label"] = canonicals.apply(data_io.canonical_label, axis=1)
ordered = canonicals.sort_values("n_ptcs", ascending=False)
selected = st.selectbox("Mechanism", ordered["_label"].tolist(), label_visibility="collapsed")
row = canonicals[canonicals["_label"] == selected].iloc[0]

with st.container(border=True):
    st.markdown(f"##### {row['representative_mechanism']}")
    a, b, c, d = st.columns(4)
    a.markdown(f"**Claims in group**\n\n{int(row['n_ptcs'])}")
    b.markdown(
        f"**Direction**\n\n{'🔮 Forward-looking' if row['direction'] == 'precursor' else '📜 Already happened'}"
    )
    c.markdown(f"**Expected effect**\n\n{'👍 Positive' if row['polarity'] > 0 else '👎 Negative'}")
    presence = {
        "triangulated": "🏢+📰 Both voices",
        "insider_only": "🏢 Company only",
        "outsider_only": "📰 Press only",
    }.get(str(row["corpus_presence"]), str(row["corpus_presence"]))
    d.markdown(f"**Who talks about it**\n\n{presence}")

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
    st.info("No claims found for this mechanism.")
else:
    st.markdown(f"**{len(members)} claims in this group:**")
    for _, p in members.head(15).iterrows():
        src = "🏢 Company" if p["source_type"] == "management" else "📰 Press"
        date_str = f"{p['source_date']:%d %b %Y}" if pd.notna(p["source_date"]) else "undated"
        st.markdown(f"**{date_str} · {src}** — *{p['mechanism']}*\n> {str(p['raw_text'])[:350]}")
    if len(members) > 15:
        st.caption(f"…and {len(members) - 15} more.")

ui.next_step("pages/4_Market_Impact.py", "Test mechanisms against the stock market")
