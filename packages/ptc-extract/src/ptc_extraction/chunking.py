from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    start: int
    end: int


def _find_sentence_boundary(text: str, target_pos: int) -> int:
    """Search backwards from target_pos for a sentence-ending boundary."""
    for i in range(target_pos, max(0, target_pos - 200), -1):
        if i < len(text) - 1 and text[i] in ".?!" and text[i + 1] == " ":
            return i + 2
    return target_pos


def _split_text_into_units(text: str, max_chars: int) -> list[str]:
    """Split text into paragraph units, further splitting oversized ones."""
    paragraphs = text.split("\n\n")
    units: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            units.append(para)
        else:
            # Split on single newlines first
            lines = para.split("\n")
            sub_units: list[str] = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if len(line) <= max_chars:
                    sub_units.append(line)
                else:
                    # Split on sentence boundaries
                    remaining = line
                    while len(remaining) > max_chars:
                        split_pos = max_chars
                        # Walk back to find sentence boundary
                        boundary = _find_sentence_boundary(remaining, split_pos)
                        if boundary <= 0 or boundary >= len(remaining):
                            boundary = split_pos
                        sub_units.append(remaining[:boundary].strip())
                        remaining = remaining[boundary:].strip()
                    if remaining:
                        sub_units.append(remaining)
            units.extend(sub_units)
    return units


def _build_overlap_prefix(prev_text: str, overlap_chars: int) -> tuple[str, int]:
    """
    Return (overlap_text, chars_from_end_of_prev) where overlap_text is taken
    from the tail of prev_text, respecting a sentence boundary if found.
    """
    if len(prev_text) <= overlap_chars:
        return prev_text, len(prev_text)
    raw_start = len(prev_text) - overlap_chars
    boundary = _find_sentence_boundary(prev_text, raw_start + 50)
    start = boundary if boundary > raw_start and boundary < len(prev_text) else raw_start
    return prev_text[start:], len(prev_text) - start


def split_document(
    text: str,
    max_chars: int = 8000,
    overlap_chars: int = 400,
    min_chunk_chars: int = 200,
) -> list[Chunk]:
    """
    Hierarchical splitter.

    1. If len(text) <= max_chars: return a single Chunk(text, 0, len(text)).
    2. Split on '\\n\\n' (paragraphs). Accumulate paragraphs until adding the next
       would exceed max_chars; emit chunk; start next.
    3. If a single paragraph exceeds max_chars: split it on '\\n' first, then on
       sentence boundaries ('. ', '? ', '! '). Same accumulation logic.
    4. Maintain a sliding overlap: each non-first chunk starts with the last
       `overlap_chars` of the previous chunk's text. Overlap must respect a
       sentence boundary if one is found within the overlap window; otherwise
       use the raw character offset.
    5. Drop final chunks shorter than `min_chunk_chars` only if there are
       multiple chunks. If the whole document is shorter, return it as one chunk
       even if below min_chunk_chars.
    6. start/end offsets refer to the position in the original `text`.
       For the overlap portion of a chunk, `start` is the offset of the overlap
       beginning, not the new content beginning. Invariant: text[chunk.start:chunk.end]
       equals chunk.text.
    """
    if not text:
        return []

    if len(text) <= max_chars:
        return [Chunk(text=text, start=0, end=len(text))]

    units = _split_text_into_units(text, max_chars)
    if not units:
        return []

    # Accumulate units into chunks, tracking positions back to original text
    chunks: list[Chunk] = []
    current_units: list[str] = []
    current_len = 0
    prev_chunk_text: str | None = None

    def _emit_chunk(units_in_chunk: list[str], prev_text: str | None) -> Chunk:
        chunk_body = "\n\n".join(units_in_chunk)
        if prev_text is not None:
            overlap_text, _ = _build_overlap_prefix(prev_text, overlap_chars)
            full_text = overlap_text + "\n\n" + chunk_body if overlap_text else chunk_body
        else:
            full_text = chunk_body
            overlap_text = ""

        # Locate full_text in original
        search_start = chunks[-1].start if chunks else 0
        # For overlap chunks, search from slightly before the previous start
        if prev_text is not None and chunks:
            search_start = max(0, chunks[-1].end - overlap_chars - 100)

        pos = text.find(full_text, search_start)
        if pos == -1:
            # Overlap prefix may not be verbatim in original; find body instead
            body_pos = text.find(chunk_body, search_start)
            if body_pos == -1:
                # Fall back: find first unit of body
                body_pos = text.find(units_in_chunk[0], search_start)
            if body_pos != -1 and prev_text is not None and overlap_text:
                # Start at the overlap boundary
                overlap_start = max(0, body_pos - len(overlap_text) - 10)
                overlap_pos = text.find(overlap_text, overlap_start)
                if overlap_pos != -1 and overlap_pos < body_pos:
                    start = overlap_pos
                    end = start + len(full_text)
                    if end > len(text):
                        end = len(text)
                    actual_text = text[start:end]
                    return Chunk(text=actual_text, start=start, end=end)
                start = body_pos
                end = body_pos + len(chunk_body)
                if end > len(text):
                    end = len(text)
                return Chunk(text=text[start:end], start=start, end=end)
            if body_pos != -1:
                end = body_pos + len(chunk_body)
                if end > len(text):
                    end = len(text)
                return Chunk(text=text[body_pos:end], start=body_pos, end=end)
            # Last resort
            start = search_start
            end = min(len(text), start + len(full_text))
            return Chunk(text=text[start:end], start=start, end=end)
        end = pos + len(full_text)
        if end > len(text):
            end = len(text)
        return Chunk(text=text[pos:end], start=pos, end=end)

    for unit in units:
        sep_len = 2 if current_units else 0  # "\n\n"
        if current_units and current_len + sep_len + len(unit) > max_chars:
            chunk = _emit_chunk(current_units, prev_chunk_text)
            chunks.append(chunk)
            prev_chunk_text = chunk.text
            current_units = [unit]
            current_len = len(unit)
        else:
            current_units.append(unit)
            current_len += sep_len + len(unit)

    if current_units:
        chunk = _emit_chunk(current_units, prev_chunk_text)
        chunks.append(chunk)

    # Drop trailing short chunks only when there are multiple
    if len(chunks) > 1:
        chunks = [c for c in chunks if len(c.text) >= min_chunk_chars]

    return chunks
