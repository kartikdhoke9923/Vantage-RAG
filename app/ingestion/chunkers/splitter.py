from typing import List

import logfire


def _split_long_text(text: str, chunk_size: int) -> List[str]:
    """Hard-split an oversized block into chunk_size-maxed pieces on word boundaries."""
    pieces: List[str] = []
    current: List[str] = []
    current_len = 0

    for word in text.split():
        if len(word) >= chunk_size:
            if current:
                pieces.append(" ".join(current))
                current = []
                current_len = 0
            pieces.extend(word[i : i + chunk_size] for i in range(0, len(word), chunk_size))
            continue
        if current_len + len(word) + 1 > chunk_size:
            pieces.append(" ".join(current))
            current = []
            current_len = 0
        current.append(word)
        current_len += len(word) + 1

    if current:
        pieces.append(" ".join(current))

    return [p for p in pieces if p.strip()]


def chunk_text(text: str, chunk_size: int = 1500) -> List[str]:
    """
    Chunk text by paragraphs, merging short paragraphs up to chunk_size.
    Oversized paragraphs are hard-split so no chunk ever exceeds chunk_size.
    """
    with logfire.span("✂️ Text Chunking", text_length=len(text)):
        if not text.strip():
            return []

        paragraphs = text.split("\n\n")
        chunks: List[str] = []
        current_chunk = ""

        def flush():
            nonlocal current_chunk
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = ""

        for p in paragraphs:
            p = p.strip()
            if not p:
                continue

            if len(current_chunk) + len(p) + 2 < chunk_size:
                current_chunk += p + "\n\n"
                continue

            flush()

            if len(p) <= chunk_size:
                current_chunk = p + "\n\n"
                continue

            sub_chunks = _split_long_text(p, chunk_size)
            for sc in sub_chunks[:-1]:
                chunks.append(sc)
            current_chunk = sub_chunks[-1] + "\n\n"

        flush()

        valid_chunks = [c for c in chunks if c.strip()]
        logfire.info(f"✅ Generated {len(valid_chunks)} chunks")
        return valid_chunks
