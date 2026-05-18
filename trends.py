"""
Fetch Google Trends search interest and merge with weekly market data.

Run:  python trends.py
"""

from pathlib import Path
import time

import pandas as pd
from pytrends.request import TrendReq

ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed"
WEEKLY_PATH = PROCESSED_DIR / "market_weekly.csv"

TICKER_KEYWORDS = {
    "GME": "GameStop stock",
    "AMC": "AMC stock",
    "TSLA": "Tesla stock",
    "NVDA": "Nvidia stock",
    "SPY": "S&P 500",
}

START = "2020-01-01"
END = "2026-01-01"
TIMEFRAME = f"{START} {END}"


def fetch_keyword_interest(keyword: str, timeframe: str, geo: str = "") -> pd.DataFrame:
    """Fetch Trends for a single keyword (useful for testing GME first)."""
    pytrends = TrendReq(hl="en-US", tz=360)
    pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo=geo)
    df = pytrends.interest_over_time()

    if df.empty:
        raise ValueError(f"No data found for keyword: {keyword}")

    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])

    df = df.reset_index()
    df = df.rename(columns={keyword: "google_trends"})
    df["google_trends"] = df["google_trends"].astype(float)
    return df[["date", "google_trends"]]


def _interest_over_time_with_retry(pytrends: TrendReq, retries: int = 3) -> pd.DataFrame:
    """Retry on Google 429 rate-limit errors."""
    for attempt in range(retries):
        try:
            return pytrends.interest_over_time()
        except Exception as exc:
            if "429" in str(exc) and attempt < retries - 1:
                wait = 10 * (attempt + 1)
                print(f"    rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            raise


def _fetch_keyword_chunks(
    pytrends: TrendReq, keyword: str, start_year: int, end_year: int, geo: str = ""
) -> pd.DataFrame:
    """
    Fetch one keyword year-by-year so Google returns finer-grained data.

    A single 2020-2026 request is heavily downsampled (~73 points total).
    """
    chunks = []
    for year in range(start_year, end_year):
        timeframe = f"{year}-01-01 {year}-12-31"
        pytrends.build_payload([keyword], timeframe=timeframe, geo=geo)
        df = _interest_over_time_with_retry(pytrends)
        if df.empty:
            print(f"    warning: no data for {keyword} in {year}")
            time.sleep(5)
            continue
        if "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])
        df = df.reset_index()
        df = df.rename(columns={keyword: "google_trends"})
        chunks.append(df[["date", "google_trends"]])
        time.sleep(5)

    if not chunks:
        return pd.DataFrame(columns=["date", "google_trends"])

    out = pd.concat(chunks, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    return out.drop_duplicates(subset=["date"]).sort_values("date")


def fetch_all_trends(ticker_keywords: dict, start_year: int, end_year: int) -> pd.DataFrame:
    """Fetch Trends for each ticker keyword with a pause between requests."""
    all_rows = []
    pytrends = TrendReq(hl="en-US", tz=360)

    for ticker, keyword in ticker_keywords.items():
        print(f"Fetching {ticker}: {keyword} ({start_year}-{end_year - 1})")
        df = _fetch_keyword_chunks(pytrends, keyword, start_year, end_year)
        if df.empty:
            print(f"  warning: no data for {keyword}")
            continue
        df["ticker"] = ticker
        all_rows.append(df)

    if not all_rows:
        raise ValueError("No trends data fetched for any ticker.")

    return pd.concat(all_rows, ignore_index=True)


def to_weekly_trends(daily_trends: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily Trends to week-ending Friday (matches data.py)."""
    parts = []
    for ticker, g in daily_trends.groupby("ticker"):
        g = g.copy()
        g["date"] = pd.to_datetime(g["date"]).dt.tz_localize(None)
        g = g.set_index("date").sort_index()
        w = g["google_trends"].resample("W-FRI").mean().reset_index()
        w = w.rename(columns={"date": "week_end"})
        w["ticker"] = ticker
        parts.append(w)
    return pd.concat(parts, ignore_index=True)


def merge_with_market(trends_weekly: pd.DataFrame) -> pd.DataFrame:
    """Left-join Trends onto market_weekly.csv on week_end + ticker."""
    if not WEEKLY_PATH.exists():
        raise FileNotFoundError(f"{WEEKLY_PATH} not found. Run `python data.py` first.")

    market = pd.read_csv(WEEKLY_PATH, parse_dates=["week_end"])
    trends_weekly = trends_weekly.copy()
    trends_weekly["week_end"] = pd.to_datetime(trends_weekly["week_end"]).dt.tz_localize(None)
    market["week_end"] = pd.to_datetime(market["week_end"]).dt.tz_localize(None)

    return market.merge(trends_weekly, on=["week_end", "ticker"], how="left")


def save_datasets(daily: pd.DataFrame, weekly: pd.DataFrame, merged: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(PROCESSED_DIR / "trends_daily.csv", index=False)
    weekly.to_csv(PROCESSED_DIR / "trends_weekly.csv", index=False)
    merged.to_csv(PROCESSED_DIR / "market_weekly_with_trends.csv", index=False)
    print(f"Saved trends_daily.csv ({len(daily):,} rows)")
    print(f"Saved trends_weekly.csv ({len(weekly):,} rows)")
    print(f"Saved market_weekly_with_trends.csv ({len(merged):,} rows)")


def main() -> None:
    daily = fetch_all_trends(TICKER_KEYWORDS, start_year=2020, end_year=2026)
    weekly = to_weekly_trends(daily)
    merged = merge_with_market(weekly)
    save_datasets(daily, weekly, merged)

    missing = merged["google_trends"].isna().mean()
    print(f"\nMissing google_trends after merge: {missing:.1%}")

    print("\nGME sample (hype window):")
    sample = merged[
        (merged["ticker"] == "GME")
        & (merged["week_end"] >= "2020-11-01")
        & (merged["week_end"] <= "2021-06-30")
    ][["week_end", "close", "abnormal_volume", "google_trends"]].tail(8)
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
