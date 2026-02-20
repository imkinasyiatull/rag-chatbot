import chromadb
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def __call__(self, input):
        return self.model.encode(input).tolist()

    def embed_query(self, input):
        return self.model.encode(input).tolist()

    def embed_documents(self, input):
        return self.model.encode(input).tolist()

    def name(self):
        return "all-MiniLM-L6-v2"


def build_db(chunks, metas, ids, collection_name="docs", reset=True):
    client = chromadb.Client()
    emb = Embedder()

    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    col = client.get_or_create_collection(
        name=collection_name,
        embedding_function=emb
    )

    col.add(documents=chunks, metadatas=metas, ids=ids)
    return col

def retrieve(col, q):
    r = col.query(
        query_texts=[q],
        n_results=3,
        include=["documents", "metadatas"]
    )
    return list(zip(r["documents"][0], r["metadatas"][0]))

def ask_llm(context, q):
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    r = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Answer from context only."},
            {"role": "user", "content": f"{context}\n\n{q}"}
        ]
    )
    return r.choices[0].message.content
