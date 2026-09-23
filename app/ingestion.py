from bisect import bisect_right


def split_text(
    full_text: str,
    page_boundaries: list[tuple[int, int]],
    source: str,
    chunk_size: int = 500,
    overlap: int = 100,
):
    chunks = []

    page_numbers = [page for page, offset in page_boundaries]
    page_starts = [offset for page, offset in page_boundaries]

    start = 0

    while start < len(full_text):
        end = min(start + chunk_size, len(full_text))
        chunk_text = full_text[start:end]

        start_page_index = bisect_right(page_starts, start) - 1
        end_page_index = bisect_right(page_starts, end - 1) - 1

        chunk_pages = page_numbers[start_page_index : end_page_index + 1]

        chunks.append(
            {
                "text": chunk_text,
                "source": source,
                "page_numbers": chunk_pages,
                "start": start,
                "end": end,
            }
        )
        start += chunk_size - overlap

    return chunks
