# Visualising Market Hype: Meme Stocks, Social Sentiment, and Price Volatility

COSC3000 major project — multivariate visualisation of meme-stock price/volume/volatility with Google Trends attention data.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python data.py    # download market data (Yahoo Finance)
python trends.py  # fetch Google Trends and merge with weekly market data
python plots.py   # generate prototype figures
```

## Data

Processed CSVs are written to `data/processed/`. Market data via [yfinance](https://github.com/ranaroussi/yfinance); search interest via [pytrends](https://github.com/GeneralMills/pytrends).
