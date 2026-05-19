"""
Prototype charts from processed market data.

Run data.py and trends.py first, then:  python plots.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
DAILY_PATH = ROOT / "data" / "processed" / "market_daily.csv"
WEEKLY_TRENDS_PATH = ROOT / "data" / "processed" / "market_weekly_with_trends.csv"
FIGURES_DIR = ROOT / "figures"

# GME squeeze window for detailed price/volume chart
GME_ZOOM_START = "2020-11-01"
GME_ZOOM_END = "2021-06-30"

CORR_COLUMNS = [
    "return",
    "rolling_volatility",
    "abnormal_volume",
    "google_trends",
]
CORR_LABELS = [
    "Weekly return",
    "Rolling volatility",
    "Abnormal volume",
    "Google Trends",
]


def load_daily() -> pd.DataFrame:
    if not DAILY_PATH.exists():
        raise FileNotFoundError(
            f"{DAILY_PATH} not found. Run `python data.py` first."
        )
    df = pd.read_csv(DAILY_PATH, parse_dates=["date"])
    return df


def load_weekly_with_trends() -> pd.DataFrame:
    if not WEEKLY_TRENDS_PATH.exists():
        raise FileNotFoundError(
            f"{WEEKLY_TRENDS_PATH} not found. Run `python trends.py` first."
        )
    return pd.read_csv(WEEKLY_TRENDS_PATH, parse_dates=["week_end"])


def plot_gme_price_volume(daily: pd.DataFrame) -> None:
    gme = daily[
        (daily["ticker"] == "GME")
        & (daily["date"] >= GME_ZOOM_START)
        & (daily["date"] <= GME_ZOOM_END)
    ]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(gme["date"], gme["close"], color="#1f77b4", linewidth=1.5, label="Close")
    ax1.set_ylabel("Price (USD)")
    ax1.set_title("GME: Price and Volume (Meme-Stock Hype Window)")

    ax2 = ax1.twinx()
    ax2.bar(gme["date"], gme["volume"], alpha=0.25, color="#ff7f0e", width=1.0, label="Volume")
    ax2.set_ylabel("Volume")

    ax1.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_gme_price_volume.png", dpi=150)
    plt.close(fig)


def plot_rolling_volatility(daily: pd.DataFrame) -> None:
    vol = daily.dropna(subset=["rolling_volatility"])

    fig, ax = plt.subplots(figsize=(12, 5))
    for ticker in sorted(vol["ticker"].unique()):
        sub = vol[vol["ticker"] == ticker]
        ax.plot(sub["date"], sub["rolling_volatility"], label=ticker, linewidth=1.2)

    ax.set_title("20-Day Rolling Volatility (Annualised)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility")
    ax.legend(loc="upper right", ncol=5, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_rolling_volatility.png", dpi=150)
    plt.close(fig)


def plot_drawdown(daily: pd.DataFrame) -> None:
    meme = daily[daily["ticker"].isin(["GME", "AMC"])].dropna(subset=["drawdown"])

    fig, ax = plt.subplots(figsize=(12, 5))
    for ticker in ["GME", "AMC"]:
        sub = meme[meme["ticker"] == ticker]
        ax.plot(sub["date"], sub["drawdown"], label=ticker, linewidth=1.2)

    ax.set_title("Drawdown from Running Peak (GME & AMC)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_drawdown_gme_amc.png", dpi=150)
    plt.close(fig)


def plot_gme_price_vs_trends(weekly: pd.DataFrame) -> None:
    gme = weekly[
        (weekly["ticker"] == "GME")
        & (weekly["week_end"] >= GME_ZOOM_START)
        & (weekly["week_end"] <= GME_ZOOM_END)
    ].dropna(subset=["google_trends"])

    fig, ax1 = plt.subplots(figsize=(12, 5))
    line_price, = ax1.plot(
        gme["week_end"],
        gme["close"],
        color="#1f77b4",
        linewidth=2,
        label="Close price",
    )
    ax1.set_ylabel("Price (USD)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_title("GME: Price vs Google Search Interest (Hype Window)")

    ax2 = ax1.twinx()
    line_trends, = ax2.plot(
        gme["week_end"],
        gme["google_trends"],
        color="#d62728",
        linewidth=2,
        linestyle="--",
        label="Google Trends",
    )
    ax2.set_ylabel("Search interest (0–100)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.set_ylim(0, max(gme["google_trends"].max() * 1.1, 10))

    ax1.set_xlabel("Week ending")
    ax1.legend(handles=[line_price, line_trends], loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_gme_price_vs_trends.png", dpi=150)
    plt.close(fig)


def _correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation for selected weekly features (rows with complete data)."""
    subset = df[CORR_COLUMNS].dropna()
    corr = subset.corr()
    corr.index = CORR_LABELS
    corr.columns = CORR_LABELS
    return corr


def plot_correlation_heatmap(weekly: pd.DataFrame) -> None:
    """Side-by-side heatmaps: all tickers vs meme stocks in the hype window."""
    all_data = weekly.copy()
    meme_hype = weekly[
        weekly["ticker"].isin(["GME", "AMC"])
        & (weekly["week_end"] >= GME_ZOOM_START)
        & (weekly["week_end"] <= GME_ZOOM_END)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    matrices = [
        (_correlation_matrix(all_data), "All tickers (2020–2025)"),
        (_correlation_matrix(meme_hype), "GME & AMC (hype window)"),
    ]

    for ax, (corr, title) in zip(axes, matrices):
        sns.heatmap(
            corr,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title)

    fig.suptitle("Weekly Feature Correlations", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trends_scatter(weekly: pd.DataFrame) -> None:
    """Scatter: Google Trends vs trading activity (volume and volatility)."""
    plot_df = weekly.dropna(
        subset=["google_trends", "abnormal_volume", "rolling_volatility"]
    )
    meme_hype = plot_df[
        plot_df["ticker"].isin(["GME", "AMC"])
        & (plot_df["week_end"] >= GME_ZOOM_START)
        & (plot_df["week_end"] <= GME_ZOOM_END)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    panels = [
        (
            axes[0],
            plot_df,
            "abnormal_volume",
            "Abnormal volume (vs 20-week avg)",
            "All tickers (2020–2025)",
            False,
        ),
        (
            axes[1],
            meme_hype,
            "abnormal_volume",
            "Abnormal volume (vs 20-week avg)",
            "GME & AMC (hype window)",
            True,
        ),
    ]

    palette = {"GME": "#1f77b4", "AMC": "#ff7f0e", "TSLA": "#2ca02c", "NVDA": "#9467bd", "SPY": "#7f7f7f"}

    for ax, data, y_col, y_label, title, show_reg in panels:
        sns.scatterplot(
            data=data,
            x="google_trends",
            y=y_col,
            hue="ticker",
            palette=palette,
            alpha=0.75,
            s=45,
            ax=ax,
            legend=title.startswith("All"),
        )
        if show_reg and len(data) >= 3:
            sns.regplot(
                data=data,
                x="google_trends",
                y=y_col,
                scatter=False,
                color="black",
                line_kws={"linewidth": 1.5, "linestyle": "--"},
                ax=ax,
            )
        ax.set_xlabel("Google Trends (0–100)")
        ax.set_ylabel(y_label)
        ax.set_title(title)

    fig.suptitle("Search Interest vs Abnormal Trading Volume", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_trends_vs_volume_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Second figure: trends vs volatility
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    vol_panels = [
        (axes[0], plot_df, "All tickers (2020–2025)", False),
        (axes[1], meme_hype, "GME & AMC (hype window)", True),
    ]
    for ax, data, title, show_reg in vol_panels:
        sns.scatterplot(
            data=data,
            x="google_trends",
            y="rolling_volatility",
            hue="ticker",
            palette=palette,
            alpha=0.75,
            s=45,
            ax=ax,
            legend=title.startswith("All"),
        )
        if show_reg and len(data) >= 3:
            sns.regplot(
                data=data,
                x="google_trends",
                y="rolling_volatility",
                scatter=False,
                color="black",
                line_kws={"linewidth": 1.5, "linestyle": "--"},
                ax=ax,
            )
        ax.set_xlabel("Google Trends (0–100)")
        ax.set_ylabel("Rolling volatility (annualised)")
        ax.set_title(title)

    fig.suptitle("Search Interest vs Rolling Volatility", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_trends_vs_volatility_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    sns.set_theme(style="whitegrid")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    daily = load_daily()
    print(f"Loaded {len(daily):,} daily rows")

    weekly = load_weekly_with_trends()
    print(f"Loaded {len(weekly):,} weekly rows (with trends)")

    plot_gme_price_volume(daily)
    plot_rolling_volatility(daily)
    plot_drawdown(daily)
    plot_gme_price_vs_trends(weekly)
    plot_correlation_heatmap(weekly)
    plot_trends_scatter(weekly)

    print(f"Saved 7 figures to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
