# Visualising Market Hype: Meme Stocks, Social Sentiment, and Price Volatility

COSC3000 major project — multivariate visualisation of meme-stock price/volume/volatility with Google Trends attention data.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python data.py       # download market data (Yahoo Finance)
python trends.py     # fetch Google Trends and merge with weekly market data
python plots.py      # figures 01–07
python cluster.py    # K-means + PCA + rolling correlation (figures 08–10)
python analysis.py   # print correlation tables for the report
```

## Figures

| # | File | Description |
|---|------|-------------|
| 01 | `01_gme_price_volume.png` | GME price + volume (millions), event annotations |
| 02 | `02_rolling_volatility.png` | Rolling volatility by ticker |
| 03 | `03_drawdown_gme_amc.png` | GME & AMC drawdowns |
| 04 | `04_gme_price_vs_trends.png` | GME price vs Google Trends |
| 05 | `05_correlation_heatmap.png` | Feature correlations (all vs hype window) |
| 06–07 | scatter plots | Trends vs volume / volatility |
| 08 | `08_regime_timeline_gme.png` | GME price with K-means regime bands |
| 09 | `09_pca_clusters.png` | PCA of weekly features |
| 10 | `10_rolling_correlation_gme.png` | Rolling corr: Trends vs abnormal volume |

## Data

- `data/events.csv` — key GME dates for chart annotations  
- `data/processed/` — market, trends, and cluster CSVs  

Market data via [yfinance](https://github.com/ranaroussi/yfinance); search interest via [pytrends](https://github.com/GeneralMills/pytrends).
