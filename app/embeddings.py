import faiss
import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def store_embeddings(chunks: list[dict]):
    embeddings = []

    for chunk in chunks:

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
