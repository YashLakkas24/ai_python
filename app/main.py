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


@app.post("/files")
async def upload_file(file: UploadFile = File()):

    chunks, page_boundaries, store_result = extract_text(file)

    return {
        "filename": file.filename,
        "chunks": len(chunks),
        "page_boundaries": page_boundaries,
        "store_result": store_result,
    }


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(request: AskRequest):
    question = request.question

    relevant_chunks = retrieve_chunks(question)

    return relevant_chunks
