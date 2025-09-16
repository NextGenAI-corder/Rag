import os
import openai
from query_embeddings import get_similar_chunks

openai.api_key = os.getenv("OPENAI_API_KEY")

def ask_bot(question):
    context = "\n".join(get_similar_chunks(question))

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer only based on the context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
    ]

    res = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    return res.choices[0].message.content
