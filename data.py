"""
Download and process market data for the meme-stock visualisation project.

Run:  python data.py
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

# --- config ---
TICKERS = ["GME", "AMC", "TSLA", "NVDA", "SPY"]
START = "2020-01-01"
END = "2026-01-01"
ROLLING_WINDOW = 20

ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed"


def download_market_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV from Yahoo Finance and return a long-format DataFrame."""
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError("No data returned from yfinance. Check tickers and date range.")

    # MultiIndex columns: (Ticker, Price) -> one row per date per ticker
    long = raw.stack(level=0).reset_index()
    long.columns = ["date", "ticker", "open", "high", "low", "close", "volume"]
    long["date"] = pd.to_datetime(long["date"]).dt.tz_localize(None)
    long = long.sort_values(["ticker", "date"]).reset_index(drop=True)
    return long


def add_daily_features(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Add returns, rolling volatility, abnormal volume, and drawdown."""
    parts = []
    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date").copy()
        g["return"] = g["close"].pct_change()
        g["rolling_volatility"] = g["return"].rolling(window).std() * (252**0.5)
        vol_mean = g["volume"].rolling(window).mean()
        g["abnormal_volume"] = g["volume"] / vol_mean
        running_max = g["close"].cummax()
        g["drawdown"] = (g["close"] - running_max) / running_max
        g["ticker"] = ticker
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily data to week-ending Friday."""
    weekly_rows = []

    for ticker, g in daily.groupby("ticker"):
        g = g.set_index("date").sort_index()
        w = g.resample("W-FRI").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "return": lambda s: (1 + s).prod() - 1,
                "rolling_volatility": "last",
                "abnormal_volume": "mean",
                "drawdown": "last",
            }
        )
        w = w.dropna(subset=["close"]).reset_index()
        w.rename(columns={"date": "week_end"}, inplace=True)
        w["ticker"] = ticker
        weekly_rows.append(w)

    weekly = pd.concat(weekly_rows, ignore_index=True)
    cols = [
        "week_end",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "return",
        "rolling_volatility",
        "abnormal_volume",
        "drawdown",
    ]
    return weekly[cols]


def save_datasets(daily: pd.DataFrame, weekly: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    daily_path = PROCESSED_DIR / "market_daily.csv"
    weekly_path = PROCESSED_DIR / "market_weekly.csv"
    daily.to_csv(daily_path, index=False)
    weekly.to_csv(weekly_path, index=False)
    print(f"Saved {len(daily):,} daily rows -> {daily_path}")
    print(f"Saved {len(weekly):,} weekly rows -> {weekly_path}")


def main() -> None:
    print(f"Downloading {len(TICKERS)} tickers ({START} to {END})...")
    daily = download_market_data(TICKERS, START, END)
    daily = add_daily_features(daily)
    weekly = to_weekly(daily)
    save_datasets(daily, weekly)

    print("\nSample (GME, last 3 days):")
    sample = daily[daily["ticker"] == "GME"].tail(3)[
        ["date", "ticker", "close", "volume", "return", "rolling_volatility"]
    ]
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
