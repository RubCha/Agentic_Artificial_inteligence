import requests
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

import pandas as pd

FINNHUB_API_KEY = "d4fdo09r01qkcvvhpregd4fdo09r01qkcvvhprf0"
NEWS_API_KEY = "e3bcd3c1af35460da72147471ebdc4ce"


# ----------------------------
# ZEIT-HILFSFUNKTIONEN
# ----------------------------

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def cutoff_months_ago(months: int = 6) -> datetime:
    return utc_now() - timedelta(days=months * 30)


# ----------------------------
# KONFIG
# ----------------------------

TARGET_COMPANIES = [
    {"name": "NVIDIA", "ticker": "NVDA"},
    {"name": "Tesla", "ticker": "TSLA"},
    {"name": "ASML Holdings", "ticker": "ASML"},
    {"name": "META", "ticker": "META"},
    {"name": "Amazon", "ticker": "AMZN"},
]

COMPANY_KEYWORDS = {
    "NVDA": ["NVIDIA", "NVDA"],
    "TSLA": ["Tesla", "TSLA"],
    "ASML": ["ASML", "ASML Holdings"],
    "META": ["Meta Platforms", "META", "Facebook"],
    "AMZN": ["Amazon", "AMZN", "Amazon.com"],
}


# ----------------------------
# FINNHUB API
# ----------------------------

def fetch_finnhub_company_news(
    symbol: str,
    months_back: int = 6,
    max_items: int = 200
) -> List[Dict[str, Any]]:

    if not FINNHUB_API_KEY:
        print("WARN: Keine FINNHUB_API_KEY gesetzt")
        return []

    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": symbol,
        "from": cutoff_months_ago(months_back).date().isoformat(),
        "to": utc_now().date().isoformat(),
        "token": FINNHUB_API_KEY,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[Finnhub] Fehler für {symbol}: {e}")
        return []

    results = []
    for item in data[:max_items]:
        ts = item.get("datetime")
        pub_iso = None

        if isinstance(ts, (int, float)):
            pub_iso = datetime.utcfromtimestamp(ts).isoformat() + "Z"

        results.append({
            "provider": "finnhub",
            "symbol": symbol,
            "headline": item.get("headline"),
            "summary": item.get("summary"),
            "url": item.get("url"),
            "image": item.get("image"),
            "source": item.get("source"),
            "category": item.get("category"),
            "published_at_utc": pub_iso,
            "collected_at_utc": utc_now().isoformat()
        })

    return results


# ----------------------------
# NEWSAPI
# ----------------------------

def fetch_newsapi_for_keyword(
    keyword: str,
    months_back: int = 6,
    max_items: int = 30
) -> List[Dict[str, Any]]:

    if not NEWS_API_KEY:
        print("WARN: NEWS_API_KEY fehlt")
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": keyword,
        "language": "en",
        "from": cutoff_months_ago(min(months_back, 1)).date().isoformat(),
        "sortBy": "publishedAt",
        "pageSize": max_items,
        "apiKey": NEWS_API_KEY,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[NewsAPI] Fehler für {keyword}: {e}")
        return []

    results = []
    for item in data.get("articles", []):
        src = item.get("source", {})
        results.append({
            "provider": "newsapi",
            "keyword": keyword,
            "title": item.get("title"),
            "description": item.get("description"),
            "url": item.get("url"),
            "image": item.get("urlToImage"),
            "published_at_utc": item.get("publishedAt"),
            "source": src.get("name"),
            "collected_at_utc": utc_now().isoformat()
        })

    return results


# ----------------------------
# HAUPTFUNKTION
# ----------------------------

def collect_all_news(months_back: int = 6) -> pd.DataFrame:
    all_rows = []

    for comp in TARGET_COMPANIES:
        ticker = comp["ticker"]
        keywords = COMPANY_KEYWORDS[ticker]

        # Finnhub
        for item in fetch_finnhub_company_news(ticker, months_back):
            all_rows.append({
                "ticker": ticker,
                "provider": "finnhub",
                "title": item.get("headline") or item.get("summary"),
                "url": item.get("url"),
                "published_at_utc": item.get("published_at_utc"),
                "source": item.get("source"),
                "keyword": None
            })

        # NewsAPI Keywords
        for kw in keywords:
            for item in fetch_newsapi_for_keyword(kw, months_back):
                all_rows.append({
                    "ticker": ticker,
                    "provider": "newsapi",
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "published_at_utc": item.get("published_at_utc"),
                    "source": item.get("source"),
                    "keyword": kw
                })

    df = pd.DataFrame(all_rows)
    return df
