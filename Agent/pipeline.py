import pandas as pd
from news_fetch import collect_all_news
from market_data import fetch_market_data
from utils import ensure_dir

def run_pipeline():
    ensure_dir("data/news")
    ensure_dir("data/market")

    # 1) News sammeln
    df_news = collect_all_news(months_back=6)
    df_news.to_csv("data/news/news_6m.csv", index=False)

    # 2) Kurse sammeln
    tickers = ["NVDA", "TSLA", "AMZN", "META", "ASML"]
    for t in tickers:
        df = fetch_market_data(t, months_back=6)
        df.to_csv(f"data/market/{t}_6m.csv")

    return df_news
