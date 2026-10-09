# Signal Intelligence

Structured causal mechanism extraction and statistical validation from corporate documents — with an interactive Streamlit explorer for the results.

> Active development. APIs unstable.

---

## What it does

Signal Intelligence extracts **typed causal claims** — called PTCs (Points To Correlate) — from two heterogeneous corpora about a focal company: **management disclosures** (earnings calls, press releases, annual reports) and **press articles**. It clusters those claims into canonical mechanisms, triangulates them across sources, and validates their predictive relationship to stock returns using Granger causality and event studies.

The core insight: management signals specific causal mechanisms ("Red Sea disruptions will elevate logistics costs into H1 2025") that are directional and signed. Extracting and typologising these — rather than mapping to scalar sentiment — enables a richer causal model of earnings surprise.

---

## Architecture

```
Documents (PDF, HTML)
    │
    ▼
[ptc_extraction]          ─── Standalone library, packages/ptc-extract/ ────────
    ├── split_document()              Deterministic chunking (offset-safe)
    ├── BedrockBackend.complete()     AWS Bedrock (Converse API)
    ├── validate_raw_ptc_dict()       Pydantic validation, drop-on-error
    └── PTCPool (JSONL)               Resume-safe persistence

[signal_intelligence]     ─── Analytics pipeline ─────────────────────────────
    Phase 0   build_sector_ontology()   Peer discovery → press fetch → sector vectors
    Phase 1'  acquire_corpus()          IR PDFs + GDELT/Exa/FMP press → data/raw/{ticker}/
    Phase 2   embed_ptcs()              Bedrock Titan embeddings (1024-dim)
    Phases 3-5 cluster_ptcs()          HDBSCAN per corpus, cross-map, sector-tag
    Phase 6   build_timeseries()        Quarterly mention matrix + earnings alignment
    Phase 7A  run_granger_analysis()    Granger causality (ADF + sparsity + BH FDR)
    Phase 7B  run_event_study()         CAR [+1, +30] trading days + daily CAR profiles
    │
    └─▶ canonical_df  granger_df  eventstudy_df  car_profiles_df
    │
    ▼
[app/]                    ─── Guided Streamlit demo ───────────────────────────
    Overview → Sources → Claims → Mechanisms → Market impact → Method & limits
```

---

## The demo app

**Live demo:** https://signal-intelligence-demo.kxe2mskxerr3a.eu-west-3.cs.amazonlightsail.com

A guided, non-expert-friendly walkthrough of the whole pipeline on real Michelin
data: the raw documents, the extracted causal claims (with their verbatim quotes),
the clustered mechanisms, and each mechanism's measured abnormal-return profile
aligned with the stock price. Every number in the app is two clicks from the exact
sentence it came from.

The app is fully self-contained — its dataset is baked in at build time by
`scripts/build_demo_bundle.py`, so it runs with no credentials and no network.

```bash
# run it locally
uv run streamlit run app/streamlit_app.py

# refresh (or add) a company's demo data after a pipeline run
uv run python scripts/build_demo_bundle.py ML.PA
```

**Deployment** is automated: pushing to `main` triggers a GitHub Actions workflow
(`.github/workflows/deploy.yml`) that builds the Docker image, pushes it to Amazon
ECR via OIDC (no stored AWS keys), and rolls it out to an AWS Lightsail container
service. Total infra: one ECR repo, one Lightsail container service (`nano`,
~$7/mo). Lightsail is used rather than App Runner because Streamlit needs
WebSockets, which App Runner does not support.

---

## Running the pipeline end-to-end

```bash
# one company: fetch corpus -> extract PTCs (Bedrock) -> analytics -> S3 sync
uv run python scripts/run_ticker.py ML.PA

# a batch (per-ticker isolation, cost summary at the end)
uv run python scripts/run_universe.py --tickers ML.PA AAPL MSFT
```

The company universe lives in `configs/universe.yaml`. Every step is resume-safe:
re-running skips documents already fetched, chunks already extracted, and
embeddings already computed. The extraction runner prints token usage and an
estimated cost per run (~$2.50/company on the default Amazon Nova Pro model).

---

## PTC schema

Each PTC is a typed, signed causal claim:

| Field | Type | Description |
|-------|------|-------------|
| `mechanism` | str | LLM-drafted causal sentence (10–500 chars) |
| `raw_text` | str | Verbatim supporting passage from the document |
| `direction` | `precursor` \| `consequence` | Precursor = forward-looking signal; consequence = realised impact |
| `polarity` | `+1` \| `-1` | Direction of the financial impact |
| `source_date` | date | Document publication date — the tradable anchor |
| `event_date` | date \| None | When the underlying event occurred (must be ≤ source_date) |
| `source_type` | `management` \| `press` | Corpus origin |
| `source_id` | str | Unique document identifier |
| `confidence` | float \| None | LLM self-assessed confidence (0–1) |
| `content_hash()` | method → str | SHA-256 of (mechanism, direction, polarity, source_id) — dedup key |

**Why source_date, not event_date, anchors the analysis:** the event may predate the document. Using event_date as an anchor would create look-ahead bias in any market analysis. source_date is the earliest date the market could have reacted.

---

## Five canonical families

After clustering and cross-mapping, each canonical mechanism falls into one of five families:

| Family | corpus_presence | direction | What it means |
|--------|-----------------|-----------|---------------|
| **Gold** | triangulated | precursor | Both corpora see it, market hasn't priced it in — highest-value signal |
| **Privileged** | insider_only | precursor | Management signals it; press misses it — potential information edge |
| **Management blind spot** | insider_only | consequence | Management frames it as past; still operationally significant |
| **Sector baseline** | triangulated | any | Industry-wide theme — already priced in, use as control |
| **Narrative inflation** | outsider_only | precursor | Press speculates; management doesn't confirm — treat sceptically |

---

## Repository layout

```
signal_intelligence/
├── packages/
│   └── ptc-extract/               Standalone, independently publishable library
│       ├── pyproject.toml         Distribution name: ptc-extract
│       ├── src/ptc_extraction/
│       │   ├── backends/
│       │   │   ├── base.py        LLMBackend abstract base
│       │   │   ├── bedrock.py     BedrockBackend  — extra: [bedrock]
│       │   │   └── anthropic.py   AnthropicBackend — extra: [anthropic]
│       │   ├── chunking.py        Deterministic paragraph-aware splitter
│       │   ├── extractor.py       PTCExtractor — chunks, calls LLM, validates, persists
│       │   ├── hub.py             Hugging Face pool load/save — extra: [hub]
│       │   ├── io.py              load_pool, append_to_pool, dedup_pool
│       │   ├── schema.py          PTC, Direction, Polarity, SourceType (Pydantic v2)
│       │   ├── prompts.py         Injectable prompts; finance default (v1.1)
│       │   └── validation.py      validate_raw_ptc_dict, filter_valid_ptcs
│       └── tests/                 111 tests, 92% coverage
│
├── src/
│   └── signal_intelligence/       Analytics pipeline
│       ├── acquisition.py         Corpus fetchers (IR PDFs, GDELT, Exa, FMP) + S3 sync
│       ├── clustering.py          HDBSCAN, direction×polarity split, cross-map, sector-tag
│       ├── correlation.py         Granger causality + ADF + sparsity filter + BH FDR
│       ├── embedding.py           Bedrock Titan embed, .npy + index CSV, resume-safe
│       ├── eventstudy.py          compute_car, run_event_study, t-test aggregation
│       ├── ingestion.py           load_documents (PDF via pdfplumber, HTML)
│       ├── market.py              get_prices (yfinance), get_earnings (FMP API)
│       ├── pipeline.py            PipelineConfig, PipelineResult, run_pipeline
│       ├── sector_ontology.py     Peer discovery, Exa fetch, sector vector building
│       └── timeseries.py          build_mention_matrix, align_with_earnings
│
├── app/                           Streamlit results explorer (see Running end-to-end)
│   ├── streamlit_app.py           Home: company selector, impact map, top signals
│   └── pages/                     Mechanism Explorer, Statistics, Method
│
├── scripts/
│   ├── run_ticker.py              Fetch → extract → pipeline → S3 for one company
│   └── run_universe.py            Batch runner over configs/universe.yaml
│
├── tests/
│   └── signal_intelligence/       10 test modules — all pipeline phases + acquisition
│
├── notebooks/
│   └── course/                    7-lesson Kaggle-style course
│       ├── 01_ptc_schema.ipynb    PTC construction, source/event date distinction, atomicity
│       ├── 02_chunking.ipynb      Offset invariant, overlap, size limits
│       ├── 03_extraction.ipynb    Mock backend, validation, resume-safety, dedup
│       ├── 04_embedding_clustering.ipynb  Titan embeddings, HDBSCAN, cross-map, typology
│       ├── 05_timeseries.ipynb    Quarter alignment, mention matrix, earnings join
│       ├── 06_statistics.ipynb    ADF/BH/Granger (3 fixes), CAR computation, t-test
│       └── 07_pipeline.ipynb      PipelineConfig, phase toggles, end-to-end mock run
│
└── data/
    ├── raw/                       Input documents (PDF, HTML) — gitignored
    ├── market/                    Cached price and earnings CSVs — gitignored
    ├── interim/                   Embeddings (.npy) and index CSVs — gitignored
    └── processed/                 Final canonical + analysis CSVs — gitignored
```

---

## Setup

Requires Python 3.12 and [uv](https://github.com/astral-sh/uv).

```bash
uv sync --extra dev
```

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
# set FMP_API_KEY=<your Financial Modeling Prep key>
# set EXA_API_KEY=<your Exa key>  (only needed for sector_ontology phase)
```

AWS credentials for `eu-west-3` must be configured separately (used for Bedrock Titan and Claude).

Run tests:

```bash
uv run pytest
```

---

## Quick start

```python
from pathlib import Path
from signal_intelligence.pipeline import PipelineConfig, run_pipeline

cfg = PipelineConfig(
    ticker="ML.PA",
    company_name="Michelin",
    sector="Specialty Chemicals & Tires",
    pool_path=Path("data/processed/ptc_pool.jsonl"),
)
result = run_pipeline(cfg)

print(result.canonical_df[["direction", "polarity", "corpus_presence", "n_ptcs"]])
print(result.eventstudy_df[["canonical_id", "n_events", "mean_car", "p_value"]])
```

To run only specific phases (e.g. re-run statistics after adding new documents):

```python
cfg = PipelineConfig(
    ticker="ML.PA",
    pool_path=Path("data/processed/ptc_pool.jsonl"),
    run_embedding=False,    # skip — embeddings already on disk
    run_clustering=False,   # skip — loads ML.PA_canonicals.csv from disk
    run_timeseries=True,
    run_granger=True,
    run_eventstudy=True,
)
result = run_pipeline(cfg)
```

---

## Statistical methodology

### Track A — Granger causality (quarterly, N ≈ 22)

Low statistical power. Illustrative diagnostic only.

Three mandatory fixes are built in:

1. **ADF stationarity test**: each series is tested with `adfuller` before Granger. Non-stationary series are first-differenced automatically. This prevents spurious regression.
2. **Sparsity filter**: canonicals with fewer than 30% non-zero quarters are excluded. Running Granger on a mostly-zero series fits noise.
3. **Benjamini-Hochberg FDR correction**: raw p-values are never reported. BH controls the false discovery rate across all tested canonicals. Bonferroni is the fallback if BH errors.

### Track B — Event study (daily, N ≈ events)

Higher statistical power. Primary result.

- **Estimation window**: [-120, -11] trading days before each source_date
- **Normal return**: mean daily return over the estimation window
- **Event window**: [+1, +30] trading days after source_date
- **CAR**: sum of (actual − normal) returns over the event window
- **Aggregation**: mean CAR per canonical, two-tailed t-test

Consequence canonicals are excluded from Track B (the market impact predates the document publication).

---

## Key design decisions

**Resume-safe everywhere.** Long runs (500+ documents, thousands of PTCs) are interrupted and continued without re-processing. The PTC pool deduplicates by `content_hash`; the embedding index deduplicates by `content_hash`; the extractor skips `source_id`s already in the pool.

**Drop-on-error validation.** The LLM occasionally returns malformed JSON or PTCs with invalid fields. These are silently dropped and counted. One bad PTC does not abort a 500-document corpus run.

**Backend abstraction.** `LLMBackend` is an abstract class with a single `complete(system, user) → str` method. Tests use `MagicMock(spec=LLMBackend)`. Production uses `BedrockBackend`; `AnthropicBackend` targets the direct API. No LangChain, no sentence-transformers.

**source_date anchoring.** Every market analysis (event study, Granger) anchors on `source_date` (document publication date), never `event_date` (when the event happened). This eliminates look-ahead bias.

**ptc_extraction is independent.** It lives in `packages/ptc-extract/` as a self-contained,
independently publishable library with its own `pyproject.toml`, tests and CI, consumed here
as a uv workspace member. It has zero imports from `signal_intelligence` and is usable
standalone for causal-claim extraction in any domain — `source_type` accepts any string and
the prompt is injectable. See `packages/ptc-extract/README.md`.

```bash
cd packages/ptc-extract && uv sync --extra dev --extra all && uv run pytest
```

---

## External services

| Service | Used for | Required |
|---------|----------|----------|
| AWS Bedrock (`eu-west-3`) | Claude Sonnet 4.6 — PTC extraction and peer discovery | Yes |
| AWS Bedrock (`eu-west-3`) | Titan Embed Text v2 — 1024-dim embeddings | Yes |
| Financial Modeling Prep | Earnings surprise data (`FMP_API_KEY`) | For Track A/B |
| yfinance | Daily adjusted close prices | For Track B |
| Exa | Peer press article retrieval (`EXA_API_KEY`) | For sector ontology only |

---

## References

This project builds on established components; the contribution is their combination at the level of the individual causal claim.

- Mariko, D., El-Haj, M., et al. — *The Financial Document Causality Detection Shared Task (FinCausal)*, FNP Workshop series, 2020–2023. [ACL Anthology](https://aclanthology.org/2020.fnp-1.3/) — prior art on financial cause→effect extraction; PTCs add directional typing (precursor/consequence) and signed polarity.
- Tetlock, P. C. (2007). *Giving Content to Investor Sentiment: The Role of Media in the Stock Market.* Journal of Finance — canonical media-tone → returns Granger analysis.
- Loughran, T., & McDonald, B. (2011). *When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks.* Journal of Finance — foundational financial sentiment dictionaries.
- Shiller, R. J. (2017). *Narrative Economics.* American Economic Review — theoretical frame for narrative-driven price dynamics.
- Mahadevan, S. (2025). *Large Causal Models from Large Language Models (DEMOCRITUS).* [arXiv:2512.07796](https://arxiv.org/abs/2512.07796) — LLM-based causal claim extraction at scale, beyond finance.

---

## License

MIT
