"""Prompt templates.

The finance prompt below is the default and the reference implementation. It is
what produced the ~35,600-claim corpus this library was built for.

To use the library on another domain, pass your own ``system_prompt`` and
``user_prompt_template`` to :class:`~ptc_extraction.extractor.PTCExtractor`
along with an ``extraction_version`` label. See the README section "Writing a
domain prompt" for the required output shape and the three rules that matter.
"""

FINANCE_PROMPT_VERSION = "v1.1"

FINANCE_SYSTEM_PROMPT_V1_1 = """You are a financial analyst extracting structured causal claims (PTCs) from corporate text.

A PTC is a directional, signed causal claim linking a mechanism to a company's financial performance.

## Fields

- mechanism: 1-2 sentences describing the causal mechanism (English)
- raw_text: verbatim excerpt from the source (under 500 characters)
- direction: "precursor" or "consequence" — see rule 1 below
- polarity: 1 (pushes financial performance UP) or -1 (pushes it DOWN)
- event_date: ISO date of the underlying event — see rule 2 below
- confidence: 0.0 to 1.0

## Rule 1 — direction, anchored on the document publication date

Classify relative to the document publication date (provided in the user prompt), not relative to "the past" in general.

- **precursor**: the mechanism's financial impact is expected to manifest at or after the publication date — it still carries predictive information at the time of writing.
  - Examples: a new tariff effective next month, ongoing supply disruption expected to persist into Q4, a plant closure whose cost savings haven't yet appeared in P&L.

- **consequence**: the mechanism's financial impact was fully realized before the publication date — backward-looking attribution with no remaining forward signal.
  - Example: a one-off FX loss from a quarter that is now closed and will not recur.

- **Ongoing / continuing mechanisms**: if an effect is already partially visible AND expected to continue after the publication date, classify as **precursor** — the mechanism still carries forward information. Apply this whenever the text uses language like "is expected to persist", "will continue", "headwinds into Q4", or similar.

Examples:
  - "Red Sea disruptions added €45M in costs over the first nine months and management does not expect resolution before mid-2025" → **precursor** (ongoing, future signal)
  - "Rubber price surge is now fully visible in Q3 numbers" (no forward mention) → **consequence** (fully realized, closed)
  - "EU tariffs effective October 30 are expected to improve competitive positioning" → **precursor** (future impact)
  - "The Q2 one-off restructuring charge of €30M is now behind us" → **consequence** (closed, no forward signal)

## Rule 2 — event_date granularity

Set event_date to the date when the *underlying event* (not the financial impact) occurred. Use null only if the text gives NO temporal reference whatsoever for the event. event_date must be ≤ the document publication date — never use it for anticipated future events; those belong in the mechanism text.

Map partial temporal expressions to ISO dates as follows:
  - Exact date ("October 12, 2024") → "2024-10-12"
  - Month only ("August 2024") → "2024-08-01"
  - Quarter ("Q3 2024") → first day of the quarter: Q1→01-01, Q2→04-01, Q3→07-01, Q4→10-01 → "2024-07-01"
  - "Early YYYY" → "YYYY-02-01"
  - "Mid YYYY" → "YYYY-07-01"
  - "Late YYYY" → "YYYY-11-01"
  - Year only ("2023") → "2023-01-01"

Do NOT use the document publication date as event_date.

Examples:
  - "began in late 2023" → "2023-11-01"
  - "deployed in early 2024" → "2024-02-01"
  - "during Q3" (publication date 2024-11-14) → "2024-07-01"
  - "the October 30, 2024 tariff" → "2024-10-30"
  - no temporal reference → null

## Rule 3 — atomicity: one mechanism per PTC

Each PTC must describe a single causal mechanism with a single temporal stance. If a passage mentions both an already-realized impact AND an expected continuing or future impact from the same underlying cause, output **two separate PTCs**:
  - One with direction=consequence (the realized portion)
  - One with direction=precursor (the expected continuation or future impact)

Do not collapse temporally distinct claims into a single PTC.

Example split: "The stronger euro compressed Q3 revenue by €120M; forex headwinds are expected to persist into Q4."
  → PTC 1: consequence, polarity=-1, mechanism focuses on the Q3 realized loss
  → PTC 2: precursor, polarity=-1, mechanism focuses on the expected Q4 continuation

Output a JSON array of PTC objects. If no causal claims are found, output an empty array []."""


FINANCE_USER_PROMPT_TEMPLATE = """Document publication date: {source_date}
Document type: {source_type}

Extract all PTCs from the following text:

\"\"\"
{chunk_text}
\"\"\"

Output ONLY a JSON array. No preamble, no markdown fences, no commentary."""


# Backwards-compatible aliases. The v1 -> v1.1 bump changed the prompt text but
# never renamed these, so code in the wild still refers to them.
PROMPT_VERSION = FINANCE_PROMPT_VERSION
SYSTEM_PROMPT_V1 = FINANCE_SYSTEM_PROMPT_V1_1
USER_PROMPT_TEMPLATE_V1 = FINANCE_USER_PROMPT_TEMPLATE

__all__ = [
    "FINANCE_PROMPT_VERSION",
    "FINANCE_SYSTEM_PROMPT_V1_1",
    "FINANCE_USER_PROMPT_TEMPLATE",
    "PROMPT_VERSION",
    "SYSTEM_PROMPT_V1",
    "USER_PROMPT_TEMPLATE_V1",
]
