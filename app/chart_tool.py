"""
Chart tool: renders a price-trend chart for a plant/market/date and saves it
as a PNG, returning the file path so the API/UI layer can display it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import pandas as pd

CHART_DIR = Path("charts")
CHART_DIR.mkdir(exist_ok=True)


def plot_price_trend(df: pd.DataFrame, plant: str, market: str, date: str) -> str:
    subset = df[
        (df["plant"] == plant.upper())
        & (df["market"] == market.upper())
        & (df["date"] == date)
    ].sort_values("time_block")

    if subset.empty:
        raise ValueError(f"No data for {plant} / {market} / {date}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(subset["time_block"], subset["mcp"], label="MCP", linewidth=1.5)
    ax.plot(subset["time_block"], subset["acp"], label="ACP", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Time Block (15-min intervals)")
    ax.set_ylabel("Price (INR/MWh)")
    ax.set_title(f"{plant} — {market} price trend — {date}")
    ax.legend()
    fig.tight_layout()

    out_path = CHART_DIR / f"{plant}_{market}_{date}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return str(out_path)


def make_chart_tool(store):
    from langchain.tools import StructuredTool
    from pydantic import BaseModel, Field

    class ChartInput(BaseModel):
        plant: str = Field(..., description="Plant code, e.g. NLCIIST2")
        market: str = Field(..., description="DAM or RTM")
        date: str = Field(..., description="Date as YYYY-MM-DD")

    def _run(plant: str, market: str, date: str) -> str:
        try:
            path = plot_price_trend(store.df, plant, market, date)
            return f"Chart saved to {path}"
        except ValueError as e:
            return str(e)

    return StructuredTool.from_function(
        func=_run,
        name="price_trend_chart",
        description=(
            "Generate a price trend chart (MCP vs ACP across time blocks) for "
            "a given plant, market, and date. Use this when the user asks to "
            "'show', 'plot', 'chart', or 'visualize' price trends."
        ),
        args_schema=ChartInput,
    )
