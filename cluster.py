"""
K-means clustering and PCA on weekly market + Trends features.

Run after trends.py:  python cluster.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed"
WEEKLY_PATH = PROCESSED_DIR / "market_weekly_with_trends.csv"
OUTPUT_PATH = PROCESSED_DIR / "weekly_with_clusters.csv"
FIGURES_DIR = ROOT / "figures"

FEATURE_COLUMNS = [
    "return",
    "rolling_volatility",
    "abnormal_volume",
    "google_trends",
]
N_CLUSTERS = 4
RANDOM_STATE = 42


def load_weekly() -> pd.DataFrame:
    if not WEEKLY_PATH.exists():
        raise FileNotFoundError(f"{WEEKLY_PATH} not found. Run trends.py first.")
    return pd.read_csv(WEEKLY_PATH, parse_dates=["week_end"])


def add_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise features and assign K-means cluster per row."""
    work = df.dropna(subset=FEATURE_COLUMNS).copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(work[FEATURE_COLUMNS])
    work["cluster"] = KMeans(
        n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10
    ).fit_predict(scaled)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pcs = pca.fit_transform(scaled)
    work["pca1"] = pcs[:, 0]
    work["pca2"] = pcs[:, 1]
    return work


def print_cluster_profiles(df: pd.DataFrame) -> None:
    """Print mean features per cluster for report writing."""
    print("\nCluster profiles (mean of each feature):")
    profile = df.groupby("cluster")[FEATURE_COLUMNS].mean().round(3)
    print(profile.to_string())


def plot_regime_timeline_gme(df: pd.DataFrame) -> None:
    """GME close with background coloured by cluster."""
    gme = df[df["ticker"] == "GME"].sort_values("week_end")
    if gme.empty:
        return

    cluster_colors = {0: "#e8e8e8", 1: "#fff3cd", 2: "#f8d7da", 3: "#d4edda"}

    fig, ax = plt.subplots(figsize=(12, 5))
    for i in range(len(gme) - 1):
        row, nxt = gme.iloc[i], gme.iloc[i + 1]
        c = cluster_colors.get(int(row["cluster"]), "#cccccc")
        ax.axvspan(row["week_end"], nxt["week_end"], alpha=0.35, color=c, linewidth=0)

    ax.plot(gme["week_end"], gme["close"], color="#1f77b4", linewidth=2, label="Close")
    ax.set_title("GME Weekly Close with K-Means Hype Regimes (k=4)")
    ax.set_xlabel("Week ending")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "08_regime_timeline_gme.png", dpi=150)
    plt.close(fig)


def plot_pca_clusters(df: pd.DataFrame) -> None:
    """PCA of weekly features, points coloured by cluster."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="pca1",
        y="pca2",
        hue="cluster",
        style="ticker",
        palette="Set2",
        alpha=0.8,
        s=60,
        ax=ax,
    )
    ax.set_title("PCA of Weekly Features (coloured by K-Means cluster)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "09_pca_clusters.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rolling_correlation_gme(df: pd.DataFrame, window: int = 12) -> None:
    """12-week rolling correlation: Google Trends vs abnormal volume (GME)."""
    gme = df[df["ticker"] == "GME"].sort_values("week_end").dropna(
        subset=["google_trends", "abnormal_volume"]
    )
    rolling_r = gme["google_trends"].rolling(window).corr(gme["abnormal_volume"])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(gme["week_end"], rolling_r, color="#2ca02c", linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(f"GME: {window}-Week Rolling Correlation (Google Trends vs Abnormal Volume)")
    ax.set_xlabel("Week ending")
    ax.set_ylabel("Pearson r")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "10_rolling_correlation_gme.png", dpi=150)
    plt.close(fig)


def main() -> None:
    sns.set_theme(style="whitegrid")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    weekly = load_weekly()
    clustered = add_clusters(weekly)
    clustered.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(clustered):,} rows -> {OUTPUT_PATH}")

    print_cluster_profiles(clustered)
    plot_regime_timeline_gme(clustered)
    plot_pca_clusters(clustered)
    plot_rolling_correlation_gme(clustered)
    print(f"Saved figures 08–10 to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
