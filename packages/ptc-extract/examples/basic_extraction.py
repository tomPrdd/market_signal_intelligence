"""Extract causal claims from a string.

    pip install 'ptc-extract[bedrock] @ git+https://github.com/tomPrdd/ptc-extract'
    python examples/basic_extraction.py

Requires AWS credentials with Bedrock access. Swap in AnthropicBackend (install
the `anthropic` extra instead, export ANTHROPIC_API_KEY) to run against the
direct API instead.
"""

from datetime import date

from ptc_extraction import BedrockBackend, PTCExtractor, SourceType

TEXT = """
Raw material costs surged in Q3 2024, pressuring our operating margins, and
management does not expect input prices to normalise before mid-2025.

The divestiture of the specialty division, completed in October 2024, removed a
persistent drag on group profitability.
"""


def main() -> None:
    extractor = PTCExtractor(backend=BedrockBackend(), max_workers=4)

    ptcs = extractor.extract_from_text(
        text=TEXT,
        source_id="acme-2024-q3-results",
        source_type=SourceType.MANAGEMENT,
        source_date=date(2024, 11, 14),
    )

    print(f"{len(ptcs)} claims extracted\n")
    for ptc in ptcs:
        sign = "+" if ptc.polarity > 0 else "-"
        print(f"[{ptc.direction.value:11s}] {sign} {ptc.mechanism}")
        print(f"              event_date={ptc.event_date}  span={ptc.span}")
        print(f"              > {ptc.raw_text[:90]}...\n")


if __name__ == "__main__":
    main()
