"""
Streamlit chat UI for the GenAI Trading Insights Assistant.

Run with: streamlit run streamlit_app.py
"""

import os
from pathlib import Path

import streamlit as st

from app.agent import build_agent

st.set_page_config(page_title="Trading Insights Assistant", page_icon="⚡", layout="centered")
st.title("⚡ GenAI Power Trading Insights Assistant")
st.caption(
    "Agentic RAG assistant over synthetic RTM/DAM power market data "
    "(NLCIIST2, NLCIIST1, NLCTS2EXP, NLCEXP, NNTPP, NTPL)."
)

if not os.getenv("GROQ_API_KEY"):
    st.warning(
        "No GROQ_API_KEY found in environment. Get a free key at "
        "console.groq.com and set it in a .env file before running this app.",
        icon="⚠️",
    )

if "agent" not in st.session_state:
    with st.spinner("Loading agent..."):
        st.session_state.agent = build_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("chart"):
            st.image(msg["chart"])

example_qs = [
    "What was the average MCP for NLCIIST2 in DAM on 2026-08-01?",
    "Compare NLCIIST2 and NNTPP RTM prices on 2026-08-02 and plot the trend for NLCIIST2",
    "What does ACP mean?",
]
st.write("**Try asking:**")
for q in example_qs:
    st.code(q, language=None)

if question := st.chat_input("Ask about market prices, trends, or definitions..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.agent.invoke({"input": question})
            answer = result["output"]
            st.markdown(answer)

            # If the agent generated a chart, try to surface it
            chart_path = None
            for line in answer.splitlines():
                if "charts/" in line and line.strip().endswith(".png"):
                    candidate = line.strip().split()[-1]
                    if Path(candidate).exists():
                        chart_path = candidate
            if chart_path:
                st.image(chart_path)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "chart": chart_path}
    )
