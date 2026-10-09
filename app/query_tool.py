"""
Structured query tool: answers precise, numeric questions about the market
data by filtering a pandas DataFrame — no LLM hallucination risk for numbers.

The LLM agent decides *when* to call this and *how to phrase the answer*;
this module guarantees the actual figures are always read straight from data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

VALID_PLANTS = ["NLCIIST2", "NLCIIST1", "NLCTS2EXP", "NLCEXP", "NNTPP", "NTPL"]
VALID_MARKETS = ["DAM", "RTM"]


@dataclass
class QueryFilters:
    plant: Optional[str] = None
    market: Optional[str] = None
    date: Optional[str] = None  # "YYYY-MM-DD"
    time_block: Optional[int] = None


class MarketDataStore:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

    def query(self, filters: QueryFilters) -> pd.DataFrame:
        df = self.df
        if filters.plant:
            df = df[df["plant"] == filters.plant.upper()]
        if filters.market:
            df = df[df["market"] == filters.market.upper()]
        if filters.date:
            df = df[df["date"] == filters.date]
        if filters.time_block:
            df = df[df["time_block"] == filters.time_block]
        return df

    def summary_stats(self, filters: QueryFilters) -> dict:
        subset = self.query(filters)
        if subset.empty:
            return {"error": "No matching data for those filters."}

        return {
            "rows": len(subset),
            "avg_mcp": round(subset["mcp"].mean(), 2),
            "max_mcp": round(subset["mcp"].max(), 2),
            "min_mcp": round(subset["mcp"].min(), 2),
            "avg_acp": round(subset["acp"].mean(), 2),
            "onbar_pct": round(subset["onbar"].mean() * 100, 1),
        }

    def compare_plants(self, plants: list[str], market: str, date: str) -> pd.DataFrame:
        df = self.df[
            (self.df["plant"].isin([p.upper() for p in plants]))
            & (self.df["market"] == market.upper())
            & (self.df["date"] == date)
        ]
        return (
            df.groupby("plant")
            .agg(avg_mcp=("mcp", "mean"), max_mcp=("mcp", "max"), avg_acp=("acp", "mean"))
            .round(2)
            .reset_index()
        )


# --- LangChain-compatible tool wrapper -------------------------------------

def make_query_tool(store: MarketDataStore):
    from langchain.tools import StructuredTool
    from pydantic import BaseModel, Field

    class QueryInput(BaseModel):
        plant: Optional[str] = Field(
            None, description=f"Plant code, one of {VALID_PLANTS}"
        )
        market: Optional[str] = Field(None, description="DAM or RTM")
        date: Optional[str] = Field(None, description="Date as YYYY-MM-DD")
        time_block: Optional[int] = Field(
            None, description="Time block number 1-96 (15-min intervals)"
        )

    def _run(plant=None, market=None, date=None, time_block=None) -> str:
        filters = QueryFilters(plant=plant, market=market, date=date, time_block=time_block)
        stats = store.summary_stats(filters)
        return str(stats)

    return StructuredTool.from_function(
        func=_run,
        name="market_data_query",
        description=(
            "Look up power market price statistics (MCP, ACP, on-bar %) for a "
            "specific plant, market type (DAM/RTM), date, and/or time block. "
            "Use this whenever the user asks for actual numbers or prices."
        ),
        args_schema=QueryInput,
    )
