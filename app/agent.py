"""
Agent orchestration layer.

Wires together three tools into a single LangChain agent that can plan
multi-step actions: e.g. "compare DAM prices for NLCIIST2 and NNTPP last
week and plot the trend" requires the agent to (1) retrieve domain context
if needed, (2) run structured numeric queries, (3) generate a chart, and
(4) summarize the findings in plain English — genuine multi-step
orchestration rather than a single prompt-response call.
"""

from __future__ import annotations

import os

from app.chart_tool import make_chart_tool
from app.query_tool import MarketDataStore
from app.rag_store import load_vectorstore, make_retriever_tool

SYSTEM_PROMPT = """You are a power trading insights assistant for an Indian \
electricity market analyst. You have access to:
1. market_data_query — precise numeric lookups (MCP, ACP, on-bar %) for a \
plant/market/date/time-block.
2. price_trend_chart — generates a trend chart image for a plant/market/date.
3. domain_knowledge_search — glossary + narrative daily summaries.

Rules:
- Always use market_data_query for actual numbers. Never invent prices.
- Use domain_knowledge_search for definitions or general context (e.g. "what is ACP").
- Use price_trend_chart only when the user asks to see/plot/visualize a trend.
- Give clear, concise answers a non-technical stakeholder could understand.
- If a question requires multiple steps (e.g. compare two plants and chart both), \
plan and execute them in sequence before giving your final answer.
"""


def build_agent(csv_path: str = "data/market_data.csv", vector_store_path: str = "vector_store.pkl"):
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_groq import ChatGroq

    store = MarketDataStore(csv_path)
    vectorstore = load_vectorstore(vector_store_path)

    tools = [
        store_query_tool(store),
        make_chart_tool(store),
        make_retriever_tool(vectorstore),
    ]

    # Groq offers a free API tier (no credit card needed) with fast open-weight
    # models like Llama 3.3 — this keeps the whole project cost-free to run.
    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


def store_query_tool(store: MarketDataStore):
    from app.query_tool import make_query_tool

    return make_query_tool(store)


if __name__ == "__main__":
    agent_executor = build_agent()
    result = agent_executor.invoke(
        {"input": "What was the average MCP for NLCIIST2 in the DAM market on 2026-08-01?"}
    )
    print(result["output"])
