# ptc-extract

Extract typed, signed causal claims from text with an LLM. A **PTC** (Point To Correlate) is
one atomic claim — a mechanism, which way it pushes, whether the effect is still ahead or
already realised, when it was published, and the passage it came from. The library handles
chunking, parallel model calls, strict validation and deduplicated JSONL persistence; you
supply a prompt and a backend. It is domain-agnostic — the reference use case is corporate
finance, and a finance prompt ships as the default, but nothing below is finance-specific.

> **Context.** This is a tool carved out of another wider personnal exploration project,
> asking whether the causal stories a company and its press tell carry information about its
> share price, the goal being to extract the claims as structured data, cluster them into recurring 
> mechanisms, and test those against subsequent returns. The library is the extraction layer of that
> project, useful on its own and released separately for that reason. **Both are work in
> progress**.

## Install

**Not on PyPI** (yet..). Install straight from this repository:

```bash
REPO=git+https://github.com/tomPrdd/ptc-extract

pip install "ptc-extract @ $REPO"               # core: schema, chunking, validation, pools
pip install "ptc-extract[bedrock] @ $REPO"      # + Amazon Bedrock backend
pip install "ptc-extract[anthropic] @ $REPO"    # + direct Anthropic API backend
pip install "ptc-extract[hub] @ $REPO"          # + load/save pools on the Hugging Face Hub
pip install "ptc-extract[all] @ $REPO"          # everything
```

There is no `pip install ptc-extract`. The name is unclaimed on PyPI and this project makes
no reservation on it, so a future package under that name need not be this one. If you
depend on this library, pin the URL and a ref.

The base install pulls no vendor SDK. Importing the package never imports `boto3`,
`anthropic` or `huggingface_hub`; you get an `ImportError` naming the extra only when you
actually reach for that feature.

## Quickstart: extract from a string

```python
from datetime import date
from ptc_extraction import BedrockBackend, PTCExtractor, SourceType

extractor = PTCExtractor(backend=BedrockBackend())

ptcs = extractor.extract_from_text(
    text="Raw material costs surged in Q3 2024, pressuring our operating margins, "
         "and management does not expect input prices to normalise before mid-2025.",
    source_id="acme-2024-q3-results",
    source_type=SourceType.MANAGEMENT,
    source_date=date(2024, 11, 14),
)

for ptc in ptcs:
    print(ptc.direction, ptc.polarity, ptc.mechanism)
```

## Quickstart: load a pre-extracted pool

Extraction costs real money, so a reference pool is published to read instead —
**[`tomPrdd/ptc-pool-eu-largecap`](https://huggingface.co/datasets/tomPrdd/ptc-pool-eu-largecap)**
on the Hugging Face Hub. It is what this library produced on the Signal Intelligence corpus:

- **35,582 validated claims** from **567 documents** about five European large-caps —
  Michelin, L'Oréal, Sanofi, LVMH, TotalEnergies — spanning **2012–2026**.
- Two corpora: the companies' own filings (`management`) and news coverage (`press`).
- Every schema field below, plus derived columns — company, sector, language, near-duplicate
  clusters and document-grouped train/validation/test splits.
- Optional extra configs: **Titan v2 embeddings** (1024-dim), and the **full source text** of
  the management documents, so the `span` offsets resolve against real text.

Caveats worth knowing before you use it — press `raw_text` is stripped for copyright, spans
are absent on about half the rows, and the market-prediction result was negative — are
documented on the dataset page. Read it before drawing conclusions.

```python
from ptc_extraction.hub import load_pool_from_hub

ptcs = load_pool_from_hub(
    "tomPrdd/ptc-pool-eu-largecap",
    filters={"source_type": "management", "direction": "precursor"},
)
```

Every row is validated against the schema on load; rows that fail are dropped and counted,
never coerced.

## The PTC schema

| Field | Type | Meaning |
|---|---|---|
| `mechanism` | `str` | One-sentence causal statement, in the model's own words. A reformulation, not a quotation. |
| `raw_text` | `str` | The verbatim passage the claim was read from. |
| `direction` | `precursor` \| `consequence` | Whether the effect lies ahead of the publication date, or is already realised. |
| `polarity` | `+1` \| `-1` | The claimed direction of the effect on the outcome variable your prompt defines. |
| `source_date` | `date` | Publication date of the document. The anchor for all time-based analysis. |
| `event_date` | `date \| None` | When the underlying event occurred, if the text dates it. Never after `source_date`. |
| `source_type` | `str` | Corpus label — any non-empty string. |
| `source_id` | `str` | Stable identifier of the source document. |
| `span` | `(int, int) \| None` | Character offsets of `raw_text` in the document. `None` when the model paraphrased. |
| `extracted_by` | `str` | Model identifier, for audit. |
| `extraction_version` | `str` | Prompt version, for audit. |
| `confidence` | `float \| None` | Model's self-reported confidence, 0–1. Treat with suspicion. |

`PTC.content_hash()` is a stable 16-character identity over
`mechanism | direction | polarity | source_id`, used for deduplication. It deliberately
excludes `raw_text` and the dates: the same mechanism quoted from two passages of one
document is one claim.

`source_type` is a plain string. `SourceType.MANAGEMENT` and `SourceType.PRESS` exist as
documented constants for the common "subject's own words versus what others write about it"
split, but `"abstract"`, `"statute"`, `"ruling"` or anything else works identically.

## Three design rules

These are the opinionated parts. They exist because of specific failure modes.

**1. Source-date anchoring.** A claim is dated by when the document was *published*, never by
when the underlying event happened. A March document may describe a January event; dating the
claim to January lets a downstream analysis "know" things before they were public. That is
look-ahead bias, and it is the standard way to accidentally manufacture a predictive signal
that evaporates out of sample. `event_date` is recorded separately and constrained to be no
later than `source_date`.

**2. Atomicity.** One mechanism per claim. When a passage contains both a realised impact and
an expected continuation of the same cause, that is two PTCs, not one — they have different
temporal stances and belong on different sides of any forward-looking analysis. Compound
claims are unusable downstream because they cannot be assigned a single direction.

**3. No repair.** Malformed model output is dropped and counted, never patched. Any corpus of
a few hundred documents produces some garbage — bad JSON, an invalid polarity, a mechanism
below the length floor. Coercing it into shape is how unreliable claims enter results
wearing the same clothes as good ones. `extract_from_text_with_stats()` and
`extract_from_documents()` return the dropped count so the loss is visible rather than
silent.

A fourth, quieter rule: **offsets are computed, not requested.** Models are unreliable at
character arithmetic, so `span` is derived by locating `raw_text` inside the chunk that
produced it and adding the chunk's document offset. A `None` span means the model paraphrased
instead of quoting — useful signal about that claim's provenance rather than a failure.

## Writing a domain prompt

Pass your own prompt and a version label:

```python
extractor = PTCExtractor(
    backend=backend,
    system_prompt=MY_SYSTEM_PROMPT,
    user_prompt_template="Published {source_date}. Corpus: {source_type}.\n\n{chunk_text}",
    extraction_version="clinical-v1",
)
```

The template is formatted with `source_date`, `source_type` and `chunk_text`; using only a
subset is fine. `extraction_version` is stamped on every claim, so pools from different
prompts stay separable — bump it whenever the prompt text changes.

Your system prompt must instruct the model to return **a JSON array** of objects, empty when
there is nothing to extract, with these keys:

```json
[
  {
    "mechanism": "one or two sentences describing the causal mechanism",
    "raw_text": "verbatim excerpt from the source",
    "direction": "precursor",
    "polarity": -1,
    "event_date": "2024-07-01",
    "confidence": 0.9
  }
]
```

`event_date` may be `null`; `confidence` is optional. Do not ask for `span` — it is computed.
Code fences are tolerated and stripped, but instruct the model not to emit them.

Cover the three rules explicitly in your prompt, because the model, not the library, is what
enforces them:

- Define **what `polarity` is signed against** in your domain — the library is agnostic. "+1
  pushes the outcome up" is meaningless until you say what the outcome is.
- Define **`direction` relative to the publication date**, and say what to do with ongoing
  effects. The finance prompt classifies "already visible and expected to continue" as
  `precursor`, since it still carries forward information.
- Spell out **atomicity** with an example of a passage that should split into two claims.
  Models under-split without one.

The shipped finance prompt (`FINANCE_SYSTEM_PROMPT_V1_1`) is a worked example of all three,
including a mapping from partial temporal expressions ("late 2023", "Q3") to ISO dates.

## Implementing a custom backend

```python
from ptc_extraction.backends.base import LLMBackend

class MyBackend(LLMBackend):
    @property
    def model_id(self) -> str:
        return "my-org/my-model"          # recorded on every PTC.extracted_by

    def complete(self, system: str, user: str, max_tokens: int = 4096) -> str:
        return my_client.generate(system=system, user=user, max_tokens=max_tokens)
```

Two methods. `complete` returns the raw text of the completion; the extractor handles JSON
parsing, fence stripping and validation. Raise
`ptc_extraction.exceptions.LLMBackendError` for failures the extractor should count rather
than crash on. `complete` is called concurrently from a thread pool, so keep it thread-safe.

`BedrockBackend` and `AnthropicBackend` are the two shipped implementations; both expose a
`.usage` dict of cumulative input/output tokens for cost tracking.

## Origin

Built for **Signal Intelligence**, a research project testing whether the causal stories a
company and its press tell predict its share price. That project ran this library over ~570
corporate documents and extracted ~35,600 validated claims.

The market-prediction result was **negative**: the extracted polarity does not predict the
direction of subsequent abnormal returns, and the apparent significance in early versions
came from specification artefacts, non-independent observations and uncorrected multiple
testing. Details are in that repository, which reports the null result rather than burying
it.

The extraction layer is a separate question from what the claims were then used for, and it
stands on its own — which is why it lives here.

## License

MIT — see [LICENSE](LICENSE).
