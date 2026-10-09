"""
Synthetic RTM/DAM power market data generator.

Mimics the structure of real Grid India SRLDC / IEX market data used in
day-ahead (DAM) and real-time (RTM) power trading — TB (time blocks),
MCP (Market Clearing Price), ACP (Area Clearing Price), OnBar/OffBar status,
across a fixed set of power plants.

This lets the rest of the project (RAG ingestion, agent tools, dashboard)
run end-to-end without needing live API credentials for Grid India / IEX.
Swap this out for a real API client later if you want to point it at
live data.
"""

import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Six fixed plants (same naming convention as real SRLDC feeds)
PLANTS = ["NLCIIST2", "NLCIIST1", "NLCTS2EXP", "NLCEXP", "NNTPP", "NTPL"]

TIME_BLOCKS_PER_DAY = 96  # 15-minute blocks
MARKET_TYPES = ["DAM", "RTM"]

# Rough realistic price bands (INR/MWh) — DAM tends to be a bit smoother than RTM
PRICE_BANDS = {
    "DAM": (2200, 6500),
    "RTM": (1800, 9000),
}


def _time_block_to_clock(tb: int) -> str:
    """Convert a 1-96 time block index into an HH:MM string."""
    minutes = (tb - 1) * 15
    hh = (minutes // 60) % 24
    mm = minutes % 60
    return f"{hh:02d}:{mm:02d}"


def generate_day(date: datetime, market: str, seed_offset: int = 0) -> pd.DataFrame:
    rng = random.Random(hash((date.date(), market, seed_offset)) & 0xFFFFFFFF)
    lo, hi = PRICE_BANDS[market]
    rows = []

    for plant in PLANTS:
        # Give each plant a slightly different baseline so patterns look real
        plant_bias = rng.uniform(-300, 300)
        for tb in range(1, TIME_BLOCKS_PER_DAY + 1):
            # Simple diurnal curve: higher prices in morning/evening peaks
            hour = (tb - 1) * 15 / 60
            peak_factor = 1.0
            if 6 <= hour <= 10 or 18 <= hour <= 22:
                peak_factor = 1.25
            elif 0 <= hour <= 5:
                peak_factor = 0.75

            base = rng.uniform(lo, hi) * peak_factor + plant_bias
            mcp = round(max(500, base), 2)
            acp = round(max(400, mcp - rng.uniform(-150, 250)), 2)
            onbar = rng.random() > 0.15  # ~85% of the time plant is scheduled "on-bar"

            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "market": market,
                    "plant": plant,
                    "time_block": tb,
                    "clock_time": _time_block_to_clock(tb),
                    "mcp": mcp,
                    "acp": acp,
                    "onbar": onbar,
                    "revision_number": rng.randint(1, 3),
                }
            )

    return pd.DataFrame(rows)


def generate_dataset(start_date: str, num_days: int, out_path: str) -> pd.DataFrame:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    frames = []
    for d in range(num_days):
        date = start + timedelta(days=d)
        for market in MARKET_TYPES:
            frames.append(generate_day(date, market, seed_offset=d))

    df = pd.concat(frames, ignore_index=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    df = generate_dataset(
        start_date="2026-08-01",
        num_days=14,
        out_path="data/market_data.csv",
    )
    print(f"Generated {len(df):,} rows -> data/market_data.csv")
    print(df.head())
