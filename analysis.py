"""
Print correlation tables for the report (copy into results section).

Run:  python analysis.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
WEEKLY_PATH = ROOT / "data" / "processed" / "market_weekly_with_trends.csv"

GME_ZOOM_START = "2020-11-01"
GME_ZOOM_END = "2021-06-30"

COLUMNS = ["return", "rolling_volatility", "abnormal_volume", "google_trends"]
LABELS = [
    "Weekly return",
    "Rolling volatility",
    "Abnormal volume",
    "Google Trends",
]


def main() -> None:
    weekly = pd.read_csv(WEEKLY_PATH, parse_dates=["week_end"])
    all_data = weekly.dropna(subset=COLUMNS)
    meme_hype = weekly[
        weekly["ticker"].isin(["GME", "AMC"])
        & (weekly["week_end"] >= GME_ZOOM_START)
        & (weekly["week_end"] <= GME_ZOOM_END)
    ].dropna(subset=COLUMNS)

    for name, df in [("All tickers (2020-2025)", all_data), ("GME & AMC hype window", meme_hype)]:
        corr = df[COLUMNS].corr()
        corr.index = LABELS
        corr.columns = LABELS
        print(f"\n=== {name} (n={len(df)}) ===")
        print(corr.round(2).to_string())


if __name__ == "__main__":
    main()
