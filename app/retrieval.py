from app.embeddings import generate_embeddings
from app.vector_store import search_similar


def retrieve_chunks(question: str, k: int = 3):
    question_embedding = generate_embeddings([{"text": question}])[0]

    relevant_chunks = search_similar(question_embedding, k=k)

    return relevant_chunks
