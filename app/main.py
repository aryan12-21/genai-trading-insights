"""
FastAPI backend exposing the GenAI trading insights agent as a REST API.

Run with: uvicorn app.main:app --reload
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel

from app.agent import build_agent

app = FastAPI(
    title="GenAI Trading Insights API",
    description=(
        "Agentic RAG assistant over power market (RTM/DAM) data — "
        "combines structured queries, chart generation, and domain "
        "knowledge retrieval."
    ),
    version="1.0.0",
)


@lru_cache(maxsize=1)
def get_agent():
    return build_agent()


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    agent = get_agent()
    result = agent.invoke({"input": req.question})
    return AskResponse(answer=result["output"])
