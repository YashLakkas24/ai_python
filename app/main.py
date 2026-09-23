import os
import numpy as np
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from app.retrieval import retrieve_chunks
from app.ingestion import extract_text

load_dotenv(override=True)
app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

vector_index = None
stored_chunks = []


@app.post("/files")
async def upload_file(file: UploadFile = File()):
    global vector_index, stored_chunks

    vector_index, stored_chunks, page_boundaries = extract_text(file)

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
