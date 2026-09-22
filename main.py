import os
import numpy as np
import faiss
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import Annotated
from pypdf import PdfReader
 
load_dotenv(override=True)
app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

vector_index = None
stored_chunks = [] 

@app.post("/files")
async def upload_file(file: UploadFile = File()):
    global vector_index, stored_chunks

    contents = await file.read()
    with open(file.filename, "wb") as f:
        f.write(contents)

    reader = PdfReader(file.filename)
    pages = []
    for pg_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text is None or not text.strip():
            print(f"Warning: Page {pg_no} contains no extarctable text.Skipping.")
            continue

        pages.append({"text": text, "page_number": pg_no})

    if not pages:
        raise HTTPException(status_code=400, detail="No text found.")

    stored_chunks = split_text(pages, file.filename)
    vector_index, stored_chunks = store_embeddings(stored_chunks)
    print("VECTOR INDEX:", vector_index)
    return {"filename": file.filename, "chunks": len(stored_chunks)}


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(request: AskRequest):
    question_embedding = client.embeddings.create(
        input=request.question, model="text-embedding-3-small"
    )
    embedding = question_embedding.data[0].embedding
    question_np = np.array([embedding]).astype("float32")
    distances, indices = vector_index.search(question_np, k=3)
    relevant_chunks = []

    for idx in indices[0]:
        if idx != -1:
            relevant_chunks.append(stored_chunks[idx])

    context = "\n\n".join(chunk["text"] for chunk in relevant_chunks)

    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {
                "role": "system",
                "content": "Answer questions based only on the provided context.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:{request.question}",
            },
        ],
    )
    return {"answer": response.choices[0].message.content}


def split_text(pages, source, chunk_size=500, overlap=100):
    chunks = []

    for page in pages:
        text = page["text"]
        page_number = page["page_number"]

        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append(
                {"text": chunk_text, "source": source, "page_number": page_number}
            )
            start += chunk_size - overlap

    return chunks


def store_embeddings(chunks: list[str]):
    embeddings = []

    for chunk in chunks:
        print("CHUNK:", chunk)
        print("TEXT:", chunk["text"])
        print("TEXT TYPE:", type(chunk["text"]))

        response = client.embeddings.create(
            input=chunk["text"], model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        embeddings.append(embedding)

    embeddings_np = np.array(embeddings).astype("float32")
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings_np)

    return index, chunks
