"""Step 1 — Sources: the raw documents everything else is built from."""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import data_io
import theme
import ui

st.set_page_config(page_title="Sources — Signal Intelligence", page_icon="📡", layout="wide")

ticker = ui.get_ticker()
sources = data_io.load_sources(ticker)

ui.step_header(
    1,
    4,
    "📄 Sources",
    "Everything starts with **real, public documents**. Two kinds of voices are collected, "
    "on purpose — what the **company says about itself**, and what the **press says about it**. "
    "Comparing the two later is part of the method.",
)

if sources.empty:
    st.warning("No source inventory available for this company.")
    st.stop()

mgmt = sources[sources["source_type"] == "management"]
press = sources[sources["source_type"] == "press"]

c1, c2, c3 = st.columns(3)
c1.metric(
    "Company documents",
    len(mgmt),
    help="Annual results, registration documents, quarterly sales reports — published by the company itself.",
)
c2.metric(
    "Press articles",
    len(press),
    help="Finance-related news articles mentioning the company, found via the free GDELT news index.",
)
c3.metric(
    "Period covered",
    f"{sources['date'].min():%Y} to {sources['date'].max():%Y}"
    if sources["date"].notna().any()
    else "—",
)

st.divider()

# --- timeline of the corpus ---
st.subheader("When the documents were published")
st.caption(
    "Each bar counts documents published that year. Blue: company documents. "
    "Green: press articles. Gaps are honest — free sources do not cover every period equally."
)

dated = sources.dropna(subset=["date"]).copy()
dated["year"] = dated["date"].dt.year
counts = dated.groupby(["year", "source_type"]).size().unstack(fill_value=0)

fig = go.Figure()
if "management" in counts.columns:
    fig.add_trace(
        go.Bar(
            x=counts.index,
            y=counts["management"],
            name="Company documents",
            marker={"color": theme.BLUE},
        )
    )
if "press" in counts.columns:
    fig.add_trace(
        go.Bar(
            x=counts.index, y=counts["press"], name="Press articles", marker={"color": theme.GREEN}
        )
    )
fig.update_layout(
    **theme.plotly_layout(
        height=320,
        barmode="group",
        xaxis_title="Publication year",
        yaxis_title="Documents",
        legend={"orientation": "h", "y": 1.15},
    )
)
st.plotly_chart(fig, width="stretch")

st.divider()

# --- the documents themselves ---
st.subheader("The document inventory")
st.caption(
    "The full list. Company PDFs come from each company's investor-relations page and are "
    "downloadable here; press articles come from GDELT, a free global news index, filtered "
    "to finance-related coverage, and are listed but not redistributed."
)

tab1, tab2 = st.tabs([f"🏢 Company documents ({len(mgmt)})", f"📰 Press articles ({len(press)})"])

_has_links = "download_url" in sources.columns

with tab1:
    cols = ["date", "title", "format", "size_kb"] + (["download_url"] if _has_links else [])
    st.dataframe(
        mgmt[cols].rename(
            columns={
                "date": "Published",
                "title": "Document",
                "format": "Format",
                "size_kb": "Size (KB)",
                "download_url": "Download",
            }
        ),
        width="stretch",
        hide_index=True,
        column_config={
            "Download": st.column_config.LinkColumn("Download", display_text="⬇ Open / download")
        }
        if _has_links
        else None,
    )
with tab2:
    # Press articles are third-party journalism. Every one is listed here so the
    # corpus stays fully traceable and auditable, but the full text is not
    # redistributed — only the headline, the date and the opening lines.
    st.info(
        "Press articles are listed in full for traceability, but their text is **not "
        "redistributed** — they are third-party journalism, unlike the company documents "
        "above, which are public regulatory filings. Each headline below can be searched "
        "for at its original publisher."
    )
    st.dataframe(
        press[["date", "title", "excerpt"]].rename(
            columns={"date": "Published", "title": "Headline", "excerpt": "Opening lines"}
        ),
        width="stretch",
        hide_index=True,
    )

if _has_links:
    st.caption(
        "Every document used in the analysis can be downloaded — they are mirrored on a "
        "public, read-only S3 folder so you can verify the inputs yourself. Press articles "
        "were fetched from the open GDELT news index; copyright remains with their publishers."
    )

with st.expander("🔍 How this works — collection details"):
    st.markdown(
        """
- **Company documents** are harvested automatically from the company's investor-relations
  page (PDF reports: annual results, universal registration documents, quarterly sales).
- **Press articles** come from the [GDELT Project](https://www.gdeltproject.org/), a free,
  open index of world news. The pipeline queries it quarter by quarter with finance-related
  keywords, downloads each matching article, and keeps the main text.
- Every document gets a **publication date** — from its filename, or sniffed from its opening
  text. That date matters enormously later: it is the moment the market could first react.
- **Known limitation:** free press sources under-represent paywalled outlets, and a keyword
  search sometimes catches articles that merely *mention* the company. Both are called out
  honestly in *Method & limits*.
"""
    )

ui.next_step("pages/2_Claims.py", "Turn documents into claims")
