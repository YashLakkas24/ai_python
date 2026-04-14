import os
import numpy as np
import faiss
from PyPDF2 import PdfReader
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------- LOAD DOCUMENT -----------


def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


# ----------- CHUNKING -----------


def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)

    return chunks


# ----------- EMBEDDINGS -----------


def get_embedding(text):
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.array(response.data[0].embedding, dtype=np.float32)


# ----------- VECTOR STORE -----------


class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add(self, embeddings, texts):
        self.index.add(np.array(embeddings))
        self.texts.extend(texts)

    def search(self, query_embedding, k=3):
        distances, indices = self.index.search(np.array([query_embedding]), k)
        return [self.texts[i] for i in indices[0]]


# ----------- BUILD INDEX -----------


def build_index(chunks):
    embeddings = [get_embedding(chunk) for chunk in chunks]
    dim = len(embeddings[0])

    store = VectorStore(dim)
    store.add(embeddings, chunks)
    return store


# ----------- ASK QUESTION -----------


def ask_question(store, question):
    query_embedding = get_embedding(question)
    relevant_chunks = store.search(query_embedding)

    context = "\n\n".join(relevant_chunks)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Answer based only on the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )

    return response.choices[0].message.content


# ----------- MAIN -----------

if __name__ == "__main__":
    file_path = "sample.pdf"  # your document

    print("Loading document...")
    text = load_pdf(file_path)

    print("Chunking...")
    chunks = chunk_text(text)

    print("Building index...")
    store = build_index(chunks)

    print("Ready! Ask questions (type 'exit' to quit)\n")

    while True:
        q = input("You: ")
        if q.lower() == "exit":
            break

        answer = ask_question(store, q)
        print("AI:", answer)
