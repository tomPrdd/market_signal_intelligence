"""Method & limits — the full method, its honest limitations, and every source used."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ui

st.set_page_config(
    page_title="Method & limits — Signal Intelligence", page_icon="📡", layout="wide"
)

ui.get_ticker()

st.title("🔬 Method & limits")
st.markdown(
    "Everything on this page is the fine print — written to be read. If the four steps were "
    "the demo, this is the honesty section: exactly what was done, what can go wrong, and "
    "where every input came from."
)

tab_method, tab_test, tab_limits, tab_sources = st.tabs(
    [
        "The method",
        "🔬 What we found when we tested ourselves",
        "Honest limitations",
        "Sources & references",
    ]
)

with tab_test:
    st.markdown(
        """
### The result this project is actually proud of

Most write-ups stop at "here are my findings". This one ran the tests that could **destroy**
its own conclusion — and reports what happened.

#### The decisive test

If the method captures anything real, then mechanisms the AI flagged as **good for the
business** should be followed by *higher* abnormal returns than mechanisms flagged as **bad**.
That is the one prediction the whole approach must make. Here it is, across all five
companies, using the standard market-model specification:

| Company | Difference (positive vs negative) | p-value |
|---|---|---|
| Michelin | +0.04% | 0.94 — flat |
| TotalEnergies | +0.12% | 0.77 — flat |
| L'Oréal | -0.86% | 0.43 — flat |
| LVMH | +4.70% | 0.18 — right way, not significant |
| Sanofi | -1.24% | **0.003 — wrong way** |
| **Pooled** | **-1.30%** | **<0.0001 — wrong way** |

**Three companies flat, one significantly backwards, one inconclusive.** There is no
consistent directional signal. The pooled "significant" result points the *opposite* way to
the prediction and is driven mostly by one company — most plausibly mean reversion (claims
about problems appear in troughs, which are followed by rebounds), not mechanism information.

#### Three things that broke under scrutiny

**1. The original specification was removing the wrong thing.** The first version compared each
stock to *its own past average return*, which leaves all market movement inside the result.
Switching to the standard market model (removing the CAC 40's move) changed the number of
"significant" mechanisms chaotically — up for Michelin (10 → 83), *down* for L'Oréal (24 → 14)
and TotalEnergies (40 → 33). Real signal does not behave like that; specification artefacts do.

**2. There are far fewer independent observations than there appear to be.** Michelin shows 427
testable mechanisms — but they rest on only **55 distinct publication dates**. LVMH: 142
mechanisms, **18 dates**. Many mechanisms are extracted from the *same document*, so they share
an identical 30-day window. Counting them as independent makes everything look more certain
than it is.

**3. Correcting for multiple testing removes most of it.** Testing hundreds of mechanisms at
once guarantees false positives. After Benjamini-Hochberg correction, Michelin's 42 raw
"significant" results become 10 — and pure chance alone would have produced about 21 of the 42.

#### So why publish it at all?

Because the **map** is still real and useful: thousands of causal claims, extracted and
grouped, each traceable to the exact sentence in the exact document. What is *not* supported
is the leap from that map to "this predicts the share price".

That distinction — and being able to demonstrate it with falsification tests rather than
assert it — is the point of the project. A finding that survives no scrutiny is worth less
than a null result that was obtained honestly.
"""
    )

with tab_method:
    st.markdown(
        """
#### The pipeline, end to end

| Step | What happens | Key tools |
|---|---|---|
| **1 · Collect** | Company PDFs from the investor-relations page; press articles from the GDELT news index | Python, GDELT DOC API |
| **2 · Extract** | An LLM reads each document in chunks and returns structured causal claims; malformed output is dropped, never repaired | Amazon Bedrock (Nova Pro), strict validation |
| **3 · Group** | Claims are embedded as vectors and clustered into mechanisms; company and press clusters are cross-compared | Titan embeddings, HDBSCAN |
| **4 · Test** | Each forward-looking mechanism's mention dates become events; abnormal returns over the next 30 trading days are averaged and t-tested | Event-study methodology, daily prices |

#### Design choices that matter

- **Publication date is the anchor.** A claim is dated by when it was *published* — the first
  moment the market could react — never by when the underlying event happened. Using the
  event date would smuggle future knowledge into the past (look-ahead bias).
- **Forward-looking claims only** enter the market test. A claim about something already
  realised has its impact (if any) already in the price.
- **Every claim keeps its verbatim quote.** Nothing in this app is more than two clicks from
  the exact sentence it came from.
- **One claim, one story.** Compound statements are split, so mechanisms stay clean.

#### The typology behind the colors

A mechanism told by **both** the company and the press (*triangulated*) is generally more
credible than a story only one side tells. A mechanism only the press tells might be
speculation — or something management prefers not to discuss. A mechanism only the company
tells might be early information — or wishful framing. Keeping the source visible is
deliberately part of the method.
"""
    )

with tab_limits:
    st.markdown(
        """
#### Read these before drawing conclusions

1. **Small samples.** Many mechanisms have fewer than ten dated events. Averages over few
   events are noisy; the confidence bands in Step 4 are wide because they should be.
2. **Correlation, not causation.** An unusual price move *after* a mention does not prove
   the mention (or the mechanism) caused it. Markets react to many things at once, and big
   corporate events generate both headlines and price moves together.
3. **Coverage is uneven.** Free press sources under-sample paywalled outlets and older
   years. Company PDFs skew to recent years. A mechanism that looks "company-only" may
   simply be one the free press corpus missed.
4. **Keyword noise.** A press search sometimes catches articles that merely mention the
   company; a few off-topic claims slip through. They are visible — which is the point of
   keeping everything traceable — but they exist.
5. **The AI extractor has limits.** Validation catches malformed claims, but a claim the AI
   *missed* is invisible. Recall is unmeasured.
6. **No trading claim.** No transaction costs, no liquidity, no out-of-sample portfolio
   test. This is a lens on corporate narrative — **not investment advice**.

#### Why publish it anyway

The interesting output is not a p-value — it is the **map**: which stories a company and its
press tell, who tells them, and which ones the market appears to shrug at. The statistics
are a filter on that map, applied honestly.
"""
    )

with tab_sources:
    st.markdown(
        """
#### Data sources used by this demo

| Source | Used for | Access |
|---|---|---|
| [Michelin investor relations](https://www.michelin.com/en/finance/regulated-information/) | Company documents (annual results, registration documents, quarterly sales) | Public |
| [GDELT Project](https://www.gdeltproject.org/) | Press articles (finance-filtered news index) | Free, open |
| Yahoo Finance (via `yfinance`) | Daily share prices | Free |
| [Amazon Bedrock](https://aws.amazon.com/bedrock/) | Claim extraction (Nova Pro) and embeddings (Titan v2) | Paid (~$12 for this corpus) |

The full document inventory — every PDF and article behind this demo — is in **Step 1 · Sources**.

#### Research this builds on

- Mariko, El-Haj *et al.* — **FinCausal** shared tasks (2020-2023): causal claim extraction
  from financial text. [ACL Anthology](https://aclanthology.org/2020.fnp-1.3/)
- Tetlock (2007) — *Giving Content to Investor Sentiment*, Journal of Finance: media tone
  and market movements.
- Loughran & McDonald (2011) — *When Is a Liability Not a Liability?*, Journal of Finance:
  finance-specific text analysis.
- Shiller (2017) — *Narrative Economics*, American Economic Review: stories as economic forces.
- Mahadevan (2025) — *Large Causal Models from Large Language Models* (DEMOCRITUS).
  [arXiv:2512.07796](https://arxiv.org/abs/2512.07796)

The contribution here is the **combination** — typed causal extraction + company-vs-press
triangulation + per-mechanism market testing in one open pipeline — not any single component.

#### The code

The entire pipeline and this app are open source — every number shown here can be
regenerated from scratch with one command per company.
"""
    )

st.divider()
ui.link("streamlit_app.py", "← Back to the overview", icon="🏠")
