from ptc_extraction.chunking import split_document


def test_empty_string_returns_empty():
    assert split_document("") == []


def test_short_text_returns_single_chunk():
    text = "This is a short paragraph. It fits in one chunk easily."
    chunks = split_document(text, max_chars=8000)
    assert len(chunks) == 1
    assert chunks[0].start == 0
    assert chunks[0].end == len(text)
    assert chunks[0].text == text


def test_short_text_offsets_invariant():
    text = "Hello world. This is a test document."
    chunks = split_document(text, max_chars=8000)
    for chunk in chunks:
        assert text[chunk.start : chunk.end] == chunk.text


def test_long_text_produces_multiple_chunks():
    para = "Alpha beta gamma delta epsilon. " * 30  # ~960 chars per para
    text = "\n\n".join([para] * 12)  # ~11500 chars total
    chunks = split_document(text, max_chars=3000, overlap_chars=100)
    assert len(chunks) > 1


def test_offsets_invariant_multi_chunk():
    para = "The company reported strong growth in emerging markets. " * 25
    text = "\n\n".join([para] * 10)
    chunks = split_document(text, max_chars=3000, overlap_chars=200)
    for chunk in chunks:
        assert text[chunk.start : chunk.end] == chunk.text, (
            f"Invariant violated: chunk[{chunk.start}:{chunk.end}] "
            f"!= stored text (first 50 chars: {chunk.text[:50]!r})"
        )


def test_overlap_exists_between_consecutive_chunks():
    para = "Revenue declined due to supply chain disruptions. " * 30
    text = "\n\n".join([para] * 8)
    overlap_chars = 200
    chunks = split_document(text, max_chars=2000, overlap_chars=overlap_chars)
    assert len(chunks) >= 2
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        curr = chunks[i]
        # Chunks must overlap in the original text
        overlap_len = prev.end - curr.start
        assert overlap_len >= overlap_chars * 0.5, (
            f"Chunk {i}: expected >=50% overlap ({overlap_chars * 0.5:.0f} chars), "
            f"got {overlap_len}"
        )


def test_huge_single_paragraph_split():
    # One paragraph that exceeds max_chars significantly
    sentence = "Operating costs increased due to inflationary pressure on wages. "
    text = sentence * 200  # ~13000 chars, no paragraph breaks
    chunks = split_document(text, max_chars=4000, overlap_chars=200)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 4000 * 1.1  # allow 10% slack


def test_no_chunk_exceeds_max_chars_with_normal_paragraphs():
    para = "Supply chain pressures weighed on gross margins during the quarter. " * 20
    text = "\n\n".join([para] * 6)
    max_chars = 2000
    chunks = split_document(text, max_chars=max_chars, overlap_chars=200)
    for chunk in chunks:
        # Overlap can push slightly past max_chars but body shouldn't
        assert len(chunk.text) <= max_chars * 1.5


def test_single_chunk_below_min_chunk_chars_not_dropped():
    text = "Short but meaningful."
    chunks = split_document(text, max_chars=8000, min_chunk_chars=200)
    assert len(chunks) == 1
    assert chunks[0].text == text
