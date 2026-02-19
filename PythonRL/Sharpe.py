import yfinance as yf
import pandas as pd
import numpy as np

# Tickers
tickers = {
    "TR": "^SP500TR",
    "Price": "^GSPC"
}

start = "2008-01-01"
end = "2011-12-31"

# Download and process each ticker
data = {}
for name, ticker in tickers.items():
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker or connection.")
    if "Adj Close" in df.columns:
        series = df["Adj Close"]
    else:
        series = df["Close"]
    series.name = name  # rename series to TR / Price
    data[name] = series

# Combine into a single DataFrame
df = pd.concat(data.values(), axis=1)
df.columns = list(data.keys())
df.dropna(inplace=True)

# Compute daily simple returns
rets = df.pct_change().dropna()

# Annual T-bill yields
t_bill = {2008: 0.0137, 2009: 0.0015, 2010: 0.0014, 2011: 0.0005}

results = []
for year in [2008, 2009, 2010, 2011]:
    year_rets = rets.loc[str(year)]
    for name in ["TR", "Price"]:
        mean_daily = year_rets[name].mean()
        vol_daily = year_rets[name].std()
        
        ann_return = mean_daily * 252
        ann_vol = vol_daily * np.sqrt(252)
        sharpe = (ann_return - t_bill[year]) / ann_vol
        
        results.append((year, name, ann_return, ann_vol, sharpe))

results_df = pd.DataFrame(results, columns=["Year", "Index", "AnnReturn", "AnnVol", "Sharpe"])
print(results_df.round(4))
