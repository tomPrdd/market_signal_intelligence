"""Signal Intelligence — guided demo.

Home page: what this is, in plain language, and the four-step journey from
raw documents to stock-price impact. Run locally with:

    uv run streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_io
import ui

st.set_page_config(
    page_title="Signal Intelligence",
    page_icon="📡",
    layout="wide",
    menu_items={
        "about": "Signal Intelligence — an open experiment in reading company news the way an analyst would."
    },
)

ticker = ui.get_ticker()
canonicals = data_io.load_canonicals(ticker)
ptcs = data_io.load_ptcs(ticker)
sources = data_io.load_sources(ticker)
eventstudy = data_io.load_eventstudy(ticker)

st.title("📡 Signal Intelligence")
st.markdown(
    """
### Can a machine read the news about a company — and tell which stories actually move its stock?

Every week, a company like **Michelin** publishes reports and gets written about in the press.
Buried in that text are hundreds of small **cause-and-effect stories**: *"rubber prices are rising,
that will squeeze margins"*, *"a factory strike disrupted production"*, *"a divestiture will refocus
the business"*.

Most tools crush all of that into a single mood score (*"sentiment: negative"*). This project does
something different: it **keeps each story intact** — what the cause is, which way it should push
the business, who said it, and when — and then checks against the stock market whether stories
of that kind were followed by unusual price moves.

**This app walks you through the whole journey, step by step, on real Michelin data.**
No finance background needed — each step explains itself.
"""
)

st.divider()

# --- the journey ---
c1, c2, c3, c4 = st.columns(4)
n_docs = len(sources) if not sources.empty else 0
n_ptcs = len(ptcs)
n_mechs = len(canonicals)
tested = (
    eventstudy[eventstudy["n_events"] >= 2].dropna(subset=["mean_car"])
    if not eventstudy.empty
    else pd.DataFrame()
)

with c1:
    st.markdown("#### 📄 1 · Sources")
    st.metric("Documents collected", f"{n_docs}")
    st.markdown("Annual reports and press articles — where the raw text comes from.")
    st.page_link("pages/1_Sources.py", label="Explore the sources", icon="📄")
with c2:
    st.markdown("#### 🧩 2 · Claims")
    st.metric("Causal claims extracted", f"{n_ptcs:,}")
    st.markdown("An AI reads every document and pulls out each cause-and-effect statement.")
    st.page_link("pages/2_Claims.py", label="Browse the claims", icon="🧩")
with c3:
    st.markdown("#### 🕸️ 3 · Mechanisms")
    st.metric("Recurring mechanisms", f"{n_mechs:,}")
    st.markdown("Claims that tell the same story are grouped into one mechanism.")
    st.page_link("pages/3_Mechanisms.py", label="See the mechanisms", icon="🕸️")
with c4:
    st.markdown("#### 📈 4 · Market impact")
    st.metric("Mechanisms market-tested", f"{len(tested)}")
    st.markdown("Did the stock move unusually after each mechanism was mentioned?")
    st.page_link("pages/4_Market_Impact.py", label="See the impact", icon="📈")

st.divider()

st.markdown(
    """
#### How to read this app

- **Follow the steps in order** the first time — each page builds on the previous one.
- Every chart has a plain-language caption; the technical detail lives in *"How this works"*
  expanders for those who want it.
- Everything is traceable: each result links back to the **exact quote** in the **exact document**
  it came from.

#### What this is — and isn't

This is an open research experiment: the code, method, and all sources are public.
It is **not** investment advice, and the statistics come with honest health warnings
(small samples, imperfect coverage) — spelled out in **Method & limits**.
"""
)

st.page_link("pages/5_Method_and_Limits.py", label="Method, limitations and references", icon="🔬")
