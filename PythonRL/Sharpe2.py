import yfinance as yf
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr

# User parameters
initial_year = 2012
N_years = 12

# Tickers
tickers = {
    "TR": "^SP500TR",   # Total return index
    "Price": "^GSPC"    # Price index
}

start = f"{initial_year}-01-01"
end = f"{initial_year + N_years - 1}-12-31"

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
    series.name = name
    data[name] = series

# Combine into a single DataFrame and align dates
df = pd.concat(data.values(), axis=1)
df.columns = list(data.keys())
df.dropna(inplace=True)

# Compute daily simple returns
rets = df.pct_change().dropna()

# Download annual 3-month T-bill yields from FRED
tbill_data = pdr.DataReader('TB3MS', 'fred', start=start, end=end)
tbill_data = tbill_data.resample('Y').mean() / 100  # convert from % to decimal
t_bill = {date.year: val for date, val in tbill_data['TB3MS'].items()}

# Compute annualized returns, vol, and Sharpe ratios
results = []
for year in range(initial_year, initial_year + N_years):
    year_rets = rets.loc[str(year)]
    for name in ["TR", "Price"]:
        mean_daily = year_rets[name].mean()
        vol_daily = year_rets[name].std()
        ann_return = mean_daily * 252
        ann_vol = vol_daily * np.sqrt(252)
        sharpe = (ann_return - t_bill.get(year, 0)) / ann_vol
        results.append((year, name, ann_return, ann_vol, sharpe))

results_df = pd.DataFrame(results, columns=["Year", "Index", "AnnReturn", "AnnVol", "Sharpe"])
print(results_df.round(4))

# Optional: save to CSV
# results_df.to_csv("SP500_Sharpe.csv", index=False)
