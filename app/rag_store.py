"""
RAG layer: builds a small vector store containing
  (a) domain glossary docs (TB, MCP, ACP, DAM, RTM, OnBar/OffBar, etc.)
  (b) auto-generated daily plant summaries in plain English

so the agent can *retrieve* context (definitions, narrative summaries)
instead of relying only on structured queries. This is what makes it a
genuine RAG + agentic system rather than just an API wrapper.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

GLOSSARY = [
    "TB (Time Block) refers to a 15-minute interval used in Indian power "
    "market scheduling. There are 96 time blocks in a day.",
    "MCP (Market Clearing Price) is the price at which supply and demand "
    "for electricity are matched in the day-ahead or real-time market.",
    "ACP (Area Clearing Price) is the clearing price adjusted for "
    "transmission congestion within a specific bidding area.",
    "DAM (Day Ahead Market) is a market where electricity is traded a day "
    "in advance of delivery, based on forecasted demand and supply.",
    "RTM (Real Time Market) is a market for electricity trading closer to "
    "actual delivery time, used to balance real-time grid fluctuations.",
    "OnBar status means a generating unit is scheduled and available to "
    "supply power during that time block; OffBar means it is not.",
    "SRLDC (Southern Regional Load Despatch Centre) and IEX (Indian Energy "
    "Exchange) are the primary sources of power market scheduling and "
    "price data referenced in this system.",
    "Revision number tracks how many times a given time block's schedule "
    "or price data has been revised/updated after initial publication.",
]


def build_daily_summaries(df: pd.DataFrame) -> list[str]:
    """Turn raw rows into short natural-language summaries per plant/day/market —
    good RAG content because it's the kind of narrative a human analyst would write.
    """
    docs = []
    grouped = df.groupby(["date", "market", "plant"])
    for (date, market, plant), g in grouped:
        avg_mcp = g["mcp"].mean()
        max_mcp = g["mcp"].max()
        min_mcp = g["mcp"].min()
        onbar_pct = g["onbar"].mean() * 100
        docs.append(
            f"On {date} in the {market} market, plant {plant} had an average "
            f"MCP of {avg_mcp:.2f} INR/MWh, ranging from {min_mcp:.2f} to "
            f"{max_mcp:.2f}. The plant was on-bar {onbar_pct:.1f}% of the day."
        )
    return docs


def _get_free_embeddings():
    """Local, free embedding model — runs on CPU, no API key, no cost."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


class SimpleVectorStore:
    """A tiny pure-Python vector store using numpy cosine similarity.

    Avoids chromadb/hnswlib, which need a C++ compiler to install on
    Windows. For a dataset this small (glossary + daily summaries — a few
    hundred short docs) a brute-force numpy search is plenty fast and
    needs zero native build tools.
    """

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.texts: list[str] = []
        self.vectors = None  # numpy array, shape (n_docs, dim)

    def add_texts(self, texts: list[str]):
        import numpy as np

        vecs = self.embeddings.embed_documents(texts)
        vecs = np.array(vecs, dtype="float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        vecs = vecs / norms  # pre-normalize for cosine similarity

        self.texts.extend(texts)
        self.vectors = vecs if self.vectors is None else __import__("numpy").vstack(
            [self.vectors, vecs]
        )

    def similarity_search(self, query: str, k: int = 3) -> list[str]:
        import numpy as np

        q = np.array(self.embeddings.embed_query(query), dtype="float32")
        q = q / (np.linalg.norm(q) + 1e-9)
        scores = self.vectors @ q
        top_idx = np.argsort(-scores)[:k]
        return [self.texts[i] for i in top_idx]

    def save(self, path: str):
        import pickle

        with open(path, "wb") as f:
            pickle.dump({"texts": self.texts, "vectors": self.vectors}, f)

    def load(self, path: str):
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)
        self.texts = data["texts"]
        self.vectors = data["vectors"]
        return self


def build_vectorstore(csv_path: str, persist_path: str = "vector_store.pkl"):
    df = pd.read_csv(csv_path)
    summaries = build_daily_summaries(df)
    all_docs = GLOSSARY + summaries

    store = SimpleVectorStore(_get_free_embeddings())
    store.add_texts(all_docs)
    store.save(persist_path)
    return store


def load_vectorstore(persist_path: str = "vector_store.pkl"):
    store = SimpleVectorStore(_get_free_embeddings())
    store.load(persist_path)
    return store


def make_retriever_tool(vectorstore: SimpleVectorStore):
    from langchain.tools import StructuredTool
    from pydantic import BaseModel, Field

    class SearchInput(BaseModel):
        query: str = Field(..., description="What to search for in the domain knowledge base")

    def _run(query: str) -> str:
        results = vectorstore.similarity_search(query, k=3)
        return "\n---\n".join(results)

    return StructuredTool.from_function(
        func=_run,
        name="domain_knowledge_search",
        description=(
            "Search domain glossary definitions (TB, MCP, ACP, DAM, RTM, "
            "OnBar/OffBar, SRLDC, IEX) and plain-English daily plant summaries. "
            "Use this for definitional questions or general narrative context, "
            "not for precise numeric lookups (use market_data_query for that)."
        ),
        args_schema=SearchInput,
    )
