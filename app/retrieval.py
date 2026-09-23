import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

vector_index = None
stored_chunks = []


def retrieve_chunks(question: str):
    if vector_index is None or vector_index.ntotal == 0:
        return {"answer": "Vector index is empty or not initialized."}

    question_embedding = client.embeddings.create(
        input=question, model="text-embedding-3-small"
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
                "content": f"Context:\n{context}\n\nQuestion:{question}",
            },
        ],
    )
    return {"answer": response.choices[0].message.content}
