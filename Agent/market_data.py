import yfinance as yf
import pandas as pd
from datetime import datetime

def fetch_market_data(ticker: str, months_back: int = 6) -> pd.DataFrame:
    end = pd.Timestamp.today()
    start = end - pd.DateOffset(months=months_back)

    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        progress=False
    )

    if df.empty:
        return df

    df = df[["Close"]].rename(columns={"Close": "Schlusskurs"})
    df.index.name = "Datum"

    return df
