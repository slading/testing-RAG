import os

import chromadb
from chromadb.utils import embedding_functions

FACTS = [
    "Capital.com is a fintech company that provides online trading and investment services.",
    "Capital.com headquarters is located in Limassol, Cyprus.",
    "Warsaw is the capital of Poland.",
    "Python is a programming language.",
]

QUERY = "Where is the headquarters of the fintech company?"


def build_collection():
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name="facts",
        embedding_function=embedding_fn,
    )
    if collection.count() == 0:
        collection.add(
            documents=FACTS,
            ids=[str(i) for i in range(len(FACTS))],
        )
    return collection


def retrieve(collection, query: str, top_k: int = 2):
    results = collection.query(query_texts=[query], n_results=top_k)
    docs = results["documents"][0]
    distances = results["distances"][0]
    return list(zip(docs, distances))


def generate_answer(query: str, context_docs: list[str]) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return fallback_answer(context_docs)

    from groq import Groq

    context = "\n".join(f"- {doc}" for doc in context_docs)
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer in English, briefly and to the point. "
                    "Use only the provided context. "
                    "If the context does not contain the answer, say so."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def fallback_answer(context_docs: list[str]) -> str:
    for doc in context_docs:
        if "headquarters" in doc.lower():
            return doc
    return context_docs[0]


def main():
    collection = build_collection()
    hits = retrieve(collection, QUERY, top_k=2)

    print("Query:", QUERY)
    print("\nRetrieved facts (lower distance = closer match):")
    for i, (doc, distance) in enumerate(hits, start=1):
        print(f"  {i}. [{distance:.3f}] {doc}")

    context_docs = [doc for doc, _ in hits]
    answer = generate_answer(QUERY, context_docs)

    print("\nAnswer:", answer)
    if not os.environ.get("GROQ_API_KEY"):
        print("\n(Tip: set GROQ_API_KEY for a more natural LLM-generated answer)")


if __name__ == "__main__":
    main()
