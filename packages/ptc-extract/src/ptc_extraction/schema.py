"""The PTC schema — a typed, signed causal claim.

The field names, the ``content_hash`` algorithm and the JSON serialisation are a
stable contract: pools written by earlier versions load unchanged, and pools
written by this version load in earlier ones.
"""

import hashlib
from datetime import date
from enum import IntEnum, StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Direction(StrEnum):
    """When the claimed effect lands, relative to the publication date."""

    PRECURSOR = "precursor"
    """The effect lies ahead — the claim still carries forward information."""

    CONSEQUENCE = "consequence"
    """The effect is already realised — backward-looking attribution."""


class Polarity(IntEnum):
    """The sign of the claimed effect."""

    POSITIVE = 1
    NEGATIVE = -1


class SourceType(StrEnum):
    """Conventional corpus labels for the two-corpus case.

    These are *documented constants, not a constraint*. :attr:`PTC.source_type`
    is a plain string and accepts any non-empty value, so a corpus split like
    ``"abstract"`` / ``"review"``, or ``"statute"`` / ``"ruling"``, works without
    subclassing anything.

    The two values here exist because the reference use case — a company's own
    disclosures versus what third parties write about it — is a useful split:
    agreement between the two is corroboration, and disagreement is interesting
    on its own.
    """

    MANAGEMENT = "management"
    """Published by the subject of the claim (annual reports, filings, calls)."""

    PRESS = "press"
    """Published by third parties about the subject (news, analyst notes)."""


class PTC(BaseModel):
    """A Point To Correlate: one atomic, signed, dated causal claim."""

    mechanism: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="One-sentence statement of the causal mechanism, in the extractor's "
        "own words. A reformulation, not a quotation.",
    )
    raw_text: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="The verbatim passage the claim was read from.",
    )
    direction: Direction = Field(
        ...,
        description="Whether the claimed effect lies ahead of the publication date "
        "(precursor) or is already realised (consequence).",
    )
    polarity: Polarity = Field(
        ...,
        description="The claimed direction of the effect: +1 for an effect that pushes "
        "the outcome variable up, -1 for one that pushes it down. In the reference "
        "finance use case the outcome variable is the subject's financial performance; "
        "in another domain it is whatever the prompt defines it to be.",
    )
    source_date: date = Field(
        ...,
        description="Publication date of the document — the anchor for every "
        "downstream time-based analysis.",
    )
    event_date: date | None = Field(
        None,
        description="When the underlying event occurred, if the text dates it at all. "
        "Never after source_date. None when the text gives no temporal reference.",
    )
    source_type: str = Field(
        ...,
        min_length=1,
        description="Corpus label. Any non-empty string; see SourceType for the "
        "conventional two-corpus values.",
    )
    source_id: str = Field(
        ..., min_length=1, description="Stable identifier of the source document."
    )
    span: tuple[int, int] | None = Field(
        None,
        description="Character offsets of raw_text within the source document, as "
        "(start, end). None when the passage could not be located verbatim, which "
        "usually means the model paraphrased instead of quoting.",
    )
    extracted_by: str = Field(..., description="Model identifier, recorded for audit.")
    extraction_version: str = Field(..., description="Prompt version, recorded for audit.")
    confidence: float | None = Field(None, ge=0.0, le=1.0)

    @field_validator("event_date")
    @classmethod
    def event_not_after_source(cls, v: date | None, info) -> date | None:
        if v is not None and "source_date" in info.data and v > info.data["source_date"]:
            raise ValueError("event_date must not be after source_date")
        return v

    @field_validator("span")
    @classmethod
    def span_is_a_sane_range(cls, v: Any) -> tuple[int, int] | None:
        if v is None:
            return None
        start, end = v
        if start < 0 or end < start:
            raise ValueError(f"span must be a non-negative (start, end) range, got {v!r}")
        return (start, end)

    def content_hash(self) -> str:
        """Stable identity of the claim, used for deduplication.

        Deliberately excludes raw_text, dates and confidence: the same mechanism
        read from two different passages of the same document is one claim.
        """
        key = f"{self.mechanism}|{self.direction.value}|{self.polarity.value}|{self.source_id}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
