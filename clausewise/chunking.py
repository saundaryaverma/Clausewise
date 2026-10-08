"""Split a contract into overlapping chunks, keeping each chunk's character offsets
so any retrieved text can be cited back to its exact place in the original."""


def chunk(text, size=600, overlap=100):
    if size <= overlap:
        raise ValueError("size must be larger than overlap")
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:  # don't cut words in half
            space = text.rfind(" ", start + size // 2, end)
            end = space if space > start else end
        chunks.append((start, end))
        if end >= n:
            break
        next_start = end - overlap
        space = text.find(" ", next_start, end)
        start = space + 1 if space != -1 else next_start
    return chunks


def overlaps(chunk_span, gold_spans):
    s, e = chunk_span
    return any(s < ge and gs < e for gs, ge in gold_spans)
