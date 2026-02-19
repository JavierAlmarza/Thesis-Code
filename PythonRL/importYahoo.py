import yfinance as yf
import pandas as pd
import numpy as np

sector_tickers = [
    "^SP500-50",  # Communication Services cyan
    "^SP500-25",  # Consumer Discretionary aquamarine
    "^SP500-30",  # Consumer Staples light green
    "^GSPE",  # Energy brown
    "^SP500-40",  # Financials pink
    "^SP500-35",  # Health Care orange
    "^SP500-20",  # Industrials violet
    "^SP500-45",  # Information Technology dark grayish brown
    "^SP500-15",  # Materials dark green
    "^SP500-60",  # Real Estate gray
    "^SP500-55"   # Utilities lavender
]

# Now auto_adjust=True is default, so this returns already adjusted Close
data = yf.download(sector_tickers,  start="2005-01-01", end="2022-01-01", interval="1d")["Close"]

# Compute daily returns
returns = data.pct_change().dropna()
cumulative_returns = (1 + returns).cumprod()

R = cumulative_returns.to_numpy()

print(data.head())
print(returns.head())

# returns is DataFrame: rows=dates, cols=tickers
# transpose so sectors are rows
returns_array = returns.T.values  


# 1) VIX Index
vix = yf.download("^VIX", start="2005-01-01", end="2022-01-01", interval="1d")["Close"]
sp500 = yf.download("^GSPC", start="2005-01-01", end="2022-01-01", interval="1d")["Close"]

# Compute SP500 returns
sp500_returns = sp500.pct_change()

# 20-day and 60-day rolling volatilities (annualized)
vol20 = sp500_returns.rolling(window=20).std() * np.sqrt(252)
vol60 = sp500_returns.rolling(window=60).std() * np.sqrt(252)

# Combine everything into one DataFrame
extra_series = pd.DataFrame({
    "VIX": vix,
    "SP500_vol20": vol20,
    "SP500_vol60": vol60
})

print("Sectors data:")
print(data.head())
print("\nReturns:")
print(returns.head())
print("\nExtra series (VIX & rolling vols):")
print(extra_series.head())