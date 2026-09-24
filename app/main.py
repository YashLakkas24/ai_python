from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
from app.retrieval import retrieve_chunks
from app.generation import generate_answer
from app.ingestion import extract_text

app = FastAPI()


@app.post("/files")
async def upload_file(file: UploadFile = File()):

    chunks, page_boundaries, store_result = await extract_text(file)

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

    answer = generate_answer(question, relevant_chunks)
    return {"answer": answer, "sources": relevant_chunks}
