# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — unreleased

**Nothing has been released yet.** There are no tags, and the package is not on PyPI —
install from the repository, as the README describes. This repository starts from a single
commit; the per-change history below predates it and is reconstructed from the project this
library was carved out of, so it reads as a description of the initial state rather than as
a diff you can inspect here.

Extracted from an exploratory research project testing whether the
causal stories a company and its press tell carry information about its share price, where
this library ran over ~570 corporate documents and produced ~35,600 validated claims. That
project is private and still in progress.

### The library at 0.1.0

- `PTCExtractor` — chunk a document, call an LLM per chunk in parallel, validate, dedup.
- `PTC` schema (pydantic v2) with `content_hash()` for stable deduplication.
- Deterministic, offset-safe `split_document()` chunker with sentence-aware overlap.
- Drop-on-error validation: malformed LLM output is counted and discarded, never repaired.
- JSONL pool persistence with resume support (`existing_source_ids`, `dedup_pool`).
- `BedrockBackend` — Amazon Bedrock Converse API, with token accounting.
- `AnthropicBackend` — direct Anthropic Messages API, behind the `anthropic` extra.
- `ptc_extraction.hub` — load and save pre-extracted PTC pools on the Hugging Face Hub,
  behind the `hub` extra. Parquet preferred, JSONL accepted. `DEFAULT_POOL_REPO` points at
  [`tomPrdd/ptc-pool-eu-largecap`](https://huggingface.co/datasets/tomPrdd/ptc-pool-eu-largecap),
  35,582 claims with separate embeddings and source-document configs.
- Injectable prompts: `PTCExtractor` accepts `system_prompt`, `user_prompt_template` and
  `extraction_version`, so the library is usable outside finance.
- `span` — character offsets of the supporting passage within its source document.
  **Computed by locating `raw_text` in the chunk, never requested from the model**, which is
  unreliable at character arithmetic. A `None` span means the model paraphrased rather than
  quoted.

### Differences from the in-project version

Relevant only if you are migrating code that imported `ptc_extraction` from the Signal
Intelligence repository. None of these were ever published, so none is a breaking change to
a released API.

- `source_type` accepts any non-empty string. `SourceType.MANAGEMENT` / `SourceType.PRESS`
  remain as documented constants for the two-corpus case; existing values validate and
  serialise identically.
- `boto3` is no longer a base dependency — install the `bedrock` extra for the Bedrock
  backend. The base package imports with no extras installed and pulls in no vendor SDK.
- Prompt constants renamed to `FINANCE_SYSTEM_PROMPT_V1_1` / `FINANCE_USER_PROMPT_TEMPLATE`
  with `FINANCE_PROMPT_VERSION`. The previous `SYSTEM_PROMPT_V1`, `USER_PROMPT_TEMPLATE_V1`
  and `PROMPT_VERSION` names remain as aliases.

`PTC` field names, `content_hash()` and the JSONL serialisation format are unchanged, so
pools written by the in-project version load without migration.
