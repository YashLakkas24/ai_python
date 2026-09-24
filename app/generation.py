import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_answer(question: str, relevant_chunks: list[dict]) -> str:
    if not relevant_chunks:
        return "I could not find relevant information in the uploaded documents."

    context = "\n\n".join(
        f"Source:{chunk['source']}|"
        f"Pages:{chunk['page_numbers']}\n"
        f"{chunk['text']}"
        for chunk in relevant_chunks
    )

    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the user's question using only the provided context. "
                    "If the context does not contain enough information to answer, "
                    "say that the information is not available in the provided "
                    "documents. Do not invent facts."
                ),
            },
            {
                "role": "user",
                "content": (f"Context:\n{context}\n\n" f"Question: {question}"),
            },
        ],
    )

    return response.choices[0].message.content
