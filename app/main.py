import os
import numpy as np
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from pypdf import PdfReader
from app.ingestion import split_text
from app.embeddings import store_embeddings
from app.retrieval import retrieve_chunks

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

    full_text = ""
    page_boundaries = []

    for pg_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text is None or not text.strip():
            print(f"Warning: Page {pg_no} contains no extractable text.Skipping.")
            continue

        text = text.strip()

        page_boundaries.append((pg_no, len(full_text)))

        full_text += text + "\n"

    if not full_text.strip():
        raise HTTPException(status_code=400, detail="No text found.")

    stored_chunks = split_text(full_text, page_boundaries, file.filename)
    vector_index, stored_chunks = store_embeddings(stored_chunks)

    return {
        "filename": file.filename,
        "chunks": len(stored_chunks),
        "page_boundaries": page_boundaries,
    }


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(request: AskRequest):
    question = request.question

    relevant_chunks = retrieve_chunks(question)

    return relevant_chunks
