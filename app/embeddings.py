import faiss
import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_embeddings(chunks: list[dict]):

    embeddings = []

    for chunk in chunks:

        response = client.embeddings.create(
            input=chunk["text"], model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        embeddings.append(embedding)

    return embeddings
