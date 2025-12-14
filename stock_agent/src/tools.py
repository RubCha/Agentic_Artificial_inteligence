import os
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Any

# --- Konfiguration (wird aus .env geladen) ---
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")

# Zeitrahmen für die News-Suche
DAYS_BACK = 7


class MarketIntelFetcher:
    """
    Klasse, die als 'Sensor' für den Aktienmarkt-Agenten dient.
    Sammelt aktuelle Nachrichten (Finnhub, NewsAPI) und historische
    Aktienkursdaten (yfinance) in einem vereinheitlichten Format.
    """

    def __init__(self, tickers: List[str]):
        self.tickers = tickers
        if not FINNHUB_API_KEY or not NEWS_API_KEY:
            print("Warnung: API-Schlüssel für Finnhub/NewsAPI nicht geladen. Der Agent ist eingeschränkt.")
        print(f"MarketIntelFetcher initialisiert für Ticker: {self.tickers}")

    # ----------------------------------------------------
    # 1. News-Beschaffung (Standard, verify=True)
    # ----------------------------------------------------

    def _fetch_finnhub_news(self, ticker: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        Holt Company-spezifische News via Finnhub.
        """
        if not FINNHUB_API_KEY:
            return []

        url = "https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": ticker,
            "from": start_date,
            "to": end_date,
            "token": FINNHUB_API_KEY
        }
        try:
            # Standardaufruf (verify=True). Erwartet nun eine funktionierende Zertifikatskette.
            response = requests.get(url, params=params)
            response.raise_for_status()

            # Normalisierung der Finnhub-Daten
            news = response.json()
            for item in news:
                item['source_api'] = 'finnhub'
                item['ticker'] = ticker
                item['publishedAt'] = datetime.fromtimestamp(item.pop('datetime', 0)).isoformat()

            return news
        except requests.exceptions.RequestException as e:
            print(f"Fehler beim Finnhub-Abruf für {ticker}: {e}")
            return []

    def _fetch_newsapi_keyword(self, keyword: str) -> List[Dict[str, Any]]:
        """
        Holt allgemeine, makro-relevante News via NewsAPI.
        """
        if not NEWS_API_KEY:
            return []

        url = "https://newsapi.org/v2/everything"
        from_date = (datetime.now() - timedelta(days=DAYS_BACK)).strftime('%Y-%m-%d')

        params = {
            "q": keyword,
            "from": from_date,
            "language": "en",
            "sortBy": "relevancy",
            "apiKey": NEWS_API_KEY,
            "pageSize": 50
        }
        try:
            # Standardaufruf (verify=True). Erwartet nun eine funktionierende Zertifikatskette.
            response = requests.get(url, params=params)
            response.raise_for_status()

            # Normalisierung der NewsAPI-Daten
            articles = response.json().get('articles', [])
            for item in articles:
                item['source_api'] = 'newsapi'
                item['summary'] = item.pop('description', 'No summary.')
                item['title'] = item.pop('title', 'No title.')
                item['url'] = item.pop('url', '#')
                item['ticker'] = 'MACRO'

            return articles
        except requests.exceptions.RequestException as e:
            print(f"Fehler beim NewsAPI-Abruf für {keyword}: {e}")
            return []

    def fetch_all_market_news(self) -> pd.DataFrame:
        """
        Kombiniert Abrufe von allen Quellen und gibt ein bereinigtes DataFrame zurück.
        """
        start_date = (datetime.now() - timedelta(days=DAYS_BACK)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')

        all_news = []

        # 1. Company-spezifische News (Finnhub)
        for ticker in self.tickers:
            print(f"-> Sammle Finnhub News für {ticker}...")
            news_data = self._fetch_finnhub_news(ticker, start_date, end_date)
            all_news.extend(news_data)

        # 2. Makro/Politische News (NewsAPI)
        macro_keywords = ["Federal Reserve", "US-Inflation", "Geopolitik", "Handelsabkommen"]
        for keyword in macro_keywords:
            print(f"-> Sammle NewsAPI News für Keyword '{keyword}'...")
            news_data = self._fetch_newsapi_keyword(keyword)
            all_news.extend(news_data)

        if not all_news:
            print("Keine News gefunden. DataFrame ist leer.")
            return pd.DataFrame()

        df = pd.DataFrame(all_news)

        # Bereinigung und Deduplizierung
        if 'url' in df.columns:
            df = df.drop_duplicates(subset=['url', 'title', 'publishedAt'], keep='first')

        final_cols = ['ticker', 'title', 'summary', 'publishedAt', 'url', 'source_api']
        for col in final_cols:
            if col not in df.columns:
                df[col] = None

        df = df[final_cols].rename(columns={'summary': 'text'})

        print(f"Insgesamt {len(df)} einzigartige Nachrichten gesammelt.")
        return df

    # ----------------------------------------------------
    # 2. Kursdaten-Beschaffung (yfinance Standard)
    # ----------------------------------------------------

    # In src/tools.py

    def fetch_historical_prices(self, period="6mo") -> pd.DataFrame:
        """
        Holt historische Kursdaten für alle konfigurierten Ticker mittels yfinance.
        """
        print("-> Starte Abruf historischer Kurse (yfinance Standard)...")
        try:
            # WICHTIG: Verwende auto_adjust=True, um sicherzustellen,
            # dass wir nur die bereinigten Adjusted Close Preise bekommen.
            # OHNE ['Adj Close'] Auswahl, da yfinance bei auto_adjust nur eine Spalte zurückgibt.
            df_prices = yf.download(
                self.tickers,
                period=period,
                interval="1d",
                auto_adjust=True,  # NEU/WICHTIG: Erzwingt Adjusted Close und vereinfacht die Struktur
                actions=False  # Nur Preise, keine Splits/Dividenden
            )['Close']  # Wähle die Spalte 'Close', die jetzt der Adjusted Close ist

            if df_prices.empty:
                print("Keine historischen Kursdaten von yfinance abgerufen.")
                return pd.DataFrame()

            # Sicherstellen, dass es ein DataFrame bleibt, auch bei nur einem Ticker
            if isinstance(df_prices, pd.Series):
                df_prices = df_prices.to_frame(name=self.tickers[0])

            print(f"Historische Daten für {len(df_prices)} Tage abgerufen.")
            return df_prices

        except Exception as e:
            print(f"Fehler beim Abruf historischer Daten: {e}")
            return pd.DataFrame()