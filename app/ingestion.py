from bisect import bisect_right
from fastapi import UploadFile, HTTPException
from pypdf import PdfReader

from app.embeddings import generate_embeddings
from app.vector_store import store_chunks


async def extract_text(file: UploadFile):
    contents = await file.read()

    with open(file.filename, "wb") as f:
        f.write(contents)
    reader = PdfReader(file.filename)

    full_text = ""
    page_boundaries = []

    for pg_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text is None or not text.strip():
            print(f"Warning: Page {pg_no} contains no extractable text. Skipping.")
            continue

        text = text.strip()

        page_boundaries.append((pg_no, len(full_text)))

        full_text += text + "\n"

    if not full_text.strip():
        raise HTTPException(status_code=400, detail="No text found.")

    chunks = split_text(full_text, page_boundaries, file.filename)
    embeddings = generate_embeddings(chunks)

    store_result = store_chunks(chunks, embeddings)
    return chunks, page_boundaries, store_result


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
