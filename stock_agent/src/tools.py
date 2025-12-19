# src/tools.py
"""
Utilities: RotatingKeyPool, MarketIntelFetcher (Finnhub + NewsAPI + yfinance),
HistoricalTools (indicators), DBManager (SQLite), ReportWriter (CSV).
Keys are read from .env via RotatingKeyPool.from_env.
"""
import os
import time
import itertools
import logging
import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

import requests
import pandas as pd
import numpy as np
import yfinance as yf
import time
import random


def _safe_get(self, url, params, pool):
    for attempt in range(5):
        key = pool.next()
        # ... (dein Key-Handling) ...

        r = requests.get(url, params=params)

        if r.status_code == 429:
            # Der wichtigste Teil: Warte bei Fehler kurz (Exponential Backoff)
            wait = (2 ** attempt) + random.random()
            logger.warning(f"IP gedrosselt! Warte {wait:.1f}s...")
            time.sleep(wait)
            pool.mark_bad(key, seconds=60)  # Sperre diesen Key für 1 Minute
            continue


logger = logging.getLogger("aladdin.tools")
THROTTLE_DEFAULT = float(os.getenv("THROTTLE_SECONDS", "0.4"))

def parse_env_list(env_var: str) -> List[str]:
    raw = os.getenv(env_var, "") or ""
    return [k.strip() for k in raw.split(",") if k.strip()]


class RotatingKeyPool:
    def __init__(self, keys: List[str], throttle: float = THROTTLE_DEFAULT):
        self.keys = keys
        self.throttle = throttle
        self.bad_until = {}  # key -> timestamp until which it's marked as bad
        self._iter = None
        self.last_call_at = 0
        self.reset_iterator()

    @classmethod
    def from_env(cls, env_var: str, throttle: float = THROTTLE_DEFAULT) -> 'RotatingKeyPool':
        keys = parse_env_list(env_var)
        if not keys:
            raise ValueError(f"No keys found in environment variable {env_var}")
        return cls(keys, throttle)

    def reset_iterator(self) -> None:
        """Reset the key iterator to start from the beginning."""
        self._iter = itertools.cycle(self.keys)

    def mark_bad(self, key: str, seconds: int = 60) -> None:
        """Markiert einen Key als ungültig für X Sekunden."""
        self.bad_until[key] = time.time() + seconds

    def __iter__(self):
        return self

    def next(self) -> str:
        if not self._iter:
            raise RuntimeError("KeyPool empty")

        for _ in range(len(self.keys)):
            key = next(self._iter)
            until = self.bad_until.get(key, 0)

            if time.time() > until:
                now = time.time()
                # ERHÖHTE SICHERHEIT: Mindestens 1.2 Sek. Pause zwischen JEDEM Call
                # Das verteilt deine 36 Keys auf ca. 45 Sekunden pro Runde
                safe_throttle = max(self.throttle, 1.2)

                elapsed = now - self.last_call_at
                if elapsed < safe_throttle:
                    time.sleep(safe_throttle - elapsed)

                self.last_call_at = time.time()
                return key

        logger.warning("ALLE Keys auf Cooldown! Warte 10s...")
        time.sleep(10)
        return next(self._iter)

# ----------------------------
# Market Data Fetcher mit Auto-Key-Switch & Throttle
# ----------------------------
class MarketIntelFetcher:
    def __init__(self, finnhub_pool: RotatingKeyPool, news_pool: RotatingKeyPool, user_agent: str = None):
        self.finnhub_pool = finnhub_pool
        self.news_pool = news_pool
        self.user_agent = user_agent or os.getenv("HTTP_USER_AGENT", "Mozilla/5.0")
        self.headers = {"User-Agent": self.user_agent}
        self.days_back = int(os.getenv("NEWS_DAYS_BACK", "7"))

    def _safe_get(self, url: str, params: dict = None, headers: dict = None, pool: RotatingKeyPool = None,
                  timeout: int = 15):
        headers = headers or self.headers.copy()
        params = params or {}

        for attempt in range(5):
            key = pool.next() if pool else None
            if key:
                if "finnhub.io" in url:
                    params["token"] = key
                else:
                    params["apiKey"] = key

            try:
                r = requests.get(url, params=params, headers=headers, timeout=timeout)

                if r.status_code == 429:  # Rate Limit geschlagen
                    # Warte länger bei jedem Versuch: 2s, 4s, 8s...
                    wait = (2 ** attempt) + random.random()
                    logger.warning(f"Rate Limit (429)! IP/Key Pause: {wait:.1f}s")
                    if pool: pool.mark_bad(key, seconds=60)
                    time.sleep(wait)
                    continue

                r.raise_for_status()
                return r
            except Exception as e:
                logger.warning(f"Versuch {attempt + 1} Fehler: {e}")
                time.sleep(2)
        raise RuntimeError("API nach 5 Versuchen nicht erreichbar.")

    def _fetch_finnhub_company(self, symbol: str) -> List[Dict[str, Any]]:
        if not self.finnhub_pool:
            return []
        url = "https://finnhub.io/api/v1/company-news"
        to_date = datetime.utcnow().date()
        from_date = to_date - timedelta(days=self.days_back)
        params = {"symbol": symbol, "from": from_date.isoformat(), "to": to_date.isoformat()}
        try:
            r = self._safe_get(url, params=params, pool=self.finnhub_pool)
            data = r.json() or []
            result = []
            for item in data:
                result.append({
                    "ticker": symbol.upper(),
                    "title": item.get("headline") or "",
                    "text": item.get("summary") or item.get("headline") or "",
                    "publishedAt": datetime.utcfromtimestamp(item.get("datetime", 0)).isoformat(),
                    "url": item.get("url"),
                    "source_api": "finnhub"
                })
            return result
        except Exception as e:
            logger.exception("Finnhub fetch error for %s: %s", symbol, e)
            return []

    def _fetch_newsapi_keyword(self, keyword: str) -> List[Dict[str, Any]]:
        if not self.news_pool:
            return []
        url = "https://newsapi.org/v2/everything"
        from_date = (datetime.utcnow() - timedelta(days=self.days_back)).strftime("%Y-%m-%d")
        params = {
            "q": keyword,
            "from": from_date,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": 50
        }
        try:
            r = self._safe_get(url, params=params, pool=self.news_pool)
            items = r.json().get("articles", []) or []
            normalized = []
            for a in items:
                normalized.append({
                    "ticker": "MACRO",
                    "title": a.get("title") or "",
                    "text": a.get("description") or a.get("content") or "",
                    "publishedAt": a.get("publishedAt"),
                    "url": a.get("url"),
                    "source_api": "newsapi"
                })
            return normalized
        except Exception as e:
            logger.exception("NewsAPI fetch error for %s: %s", keyword, e)
            return []

    def fetch_all_market_news(self, tickers: List[str]) -> pd.DataFrame:
        all_items = []
        for t in tickers:
            try:
                items = self._fetch_finnhub_company(t)
                all_items.extend(items)
            except Exception as e:
                logger.exception("Error fetching Finnhub for %s: %s", t, e)
            time.sleep(self.finnhub_pool.throttle if self.finnhub_pool else THROTTLE_DEFAULT)
        macro_keywords = ["Federal Reserve", "US inflation", "geopolitics", "trade deals", "interest rate"]
        for kw in macro_keywords:
            try:
                items = self._fetch_newsapi_keyword(kw)
                all_items.extend(items)
            except Exception as e:
                logger.exception("Error fetching NewsAPI for %s: %s", kw, e)
            time.sleep(self.news_pool.throttle if self.news_pool else THROTTLE_DEFAULT)
        if not all_items:
            return pd.DataFrame()
        df = pd.DataFrame(all_items)
        cols = ['ticker', 'title', 'text', 'publishedAt', 'url', 'source_api']
        for c in cols:
            if c not in df.columns:
                df[c] = None
        if 'url' in df.columns:
            df = df.drop_duplicates(subset=['url', 'title'], keep='first')
        return df[cols]

    def fetch_historical_prices(self, tickers: List[str], period: str = "6mo") -> pd.DataFrame:
        try:
            df = yf.download(tickers, period=period, interval="1d", auto_adjust=True, progress=False)
            if df.empty:
                return pd.DataFrame()
            if isinstance(df, pd.DataFrame) and 'Close' in df.columns:
                df_prices = df['Close']
            else:
                df_prices = df
            if isinstance(df_prices, pd.Series):
                df_prices = df_prices.to_frame(name=tickers[0].upper())
            df_prices.columns = [str(c).upper() for c in df_prices.columns]
            return df_prices
        except Exception as e:
            logger.exception("yfinance error: %s", e)
            return pd.DataFrame()

    def fetch_fundamentals(self, ticker: str) -> Dict[str, Any]:
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            return {
                "peRatio": info.get("trailingPE"),
                "forwardPE": info.get("forwardPE"),
                "debtToEquity": info.get("debtToEquity"),
                "dividendYield": info.get("dividendYield"),
                "marketCap": info.get("marketCap")
            }
        except Exception as e:
            logger.exception("Error fetching fundamentals for %s: %s", ticker, e)
            return {}

# ----------------------------
# Technical indicators / helpers
# ----------------------------
class HistoricalTools:
    @staticmethod
    def sma(series: pd.Series, window: int):
        return series.rolling(window=window).mean()

    @staticmethod
    def ema(series: pd.Series, window: int):
        return series.ewm(span=window, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, window: int = 14):
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = up.ewm(alpha=1/window, adjust=False).mean()
        ma_down = down.ewm(alpha=1/window, adjust=False).mean()
        rs = ma_up / ma_down
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def macd(series: pd.Series, fast=12, slow=26, signal=9):
        ema_fast = HistoricalTools.ema(series, fast)
        ema_slow = HistoricalTools.ema(series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - signal_line
        return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})

    @staticmethod
    def atr(series: pd.Series, window: int = 14):
        tr = series.pct_change().abs()
        atr = tr.rolling(window=window).mean() * series
        return atr

    @staticmethod
    def compute_indicators(series: pd.Series) -> Dict[str, Any]:
        out = {}
        series = series.dropna()
        out["sma_20"] = float(HistoricalTools.sma(series, 20).iloc[-1]) if len(series) >= 20 else None
        out["sma_50"] = float(HistoricalTools.sma(series, 50).iloc[-1]) if len(series) >= 50 else None
        out["ema_20"] = float(HistoricalTools.ema(series, 20).iloc[-1]) if len(series) >= 20 else None
        rsi_series = HistoricalTools.rsi(series)
        out["rsi_14"] = float(rsi_series.iloc[-1]) if not rsi_series.empty else None
        macd_df = HistoricalTools.macd(series)
        out["macd"] = float(macd_df["macd"].iloc[-1]) if not macd_df.empty else None
        out["atr_14"] = float(HistoricalTools.atr(series).iloc[-1]) if len(series) >= 14 else None
        return out

    @staticmethod
    def compute_correlation_with_spy(df_prices: pd.DataFrame, ticker: str, period: str = "6mo"):
        try:
            spy = yf.download("SPY", period=period, interval="1d", auto_adjust=True, progress=False)['Close']
            ticker_series = df_prices[ticker]
            combined = pd.concat([ticker_series.pct_change(), spy.pct_change()], axis=1).dropna()
            combined.columns = ["ticker", "spy"]
            corr = combined.corr().loc["ticker", "spy"]
            return float(corr)
        except Exception:
            return 0.0

    @staticmethod
    def atr_stop_loss(series: pd.Series, multiplier: int = 3):
        atr_series = HistoricalTools.atr(series)
        if atr_series.empty:
            return None
        last_price = float(series.iloc[-1])
        last_atr = float(atr_series.iloc[-1])
        stop = max(0.0, last_price - multiplier * last_atr)
        return round(stop, 4)

# ----------------------------
# DB Manager (SQLite)
# ----------------------------
class DBManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)

    def _init_db(self):
        with self._conn() as c:
            c.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_history (
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                publishedAt TEXT,
                sentiment REAL,
                relevance INTEGER,
                title TEXT,
                url TEXT
            )""")
            c.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                ticker TEXT,
                prediction TEXT,
                pred_price REAL,
                stop_loss REAL,
                extra TEXT
            )""")
            c.commit()

    def store_sentiments(self, df_analyzed: pd.DataFrame):
        if df_analyzed is None or df_analyzed.empty:
            return
        with self._conn() as conn:
            for _, r in df_analyzed.iterrows():
                conn.execute("""
                    INSERT INTO sentiment_history (ticker, publishedAt, sentiment, relevance, title, url)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    r.get('ticker'),
                    r.get('publishedAt'),
                    float(r.get('sentiment_score', 0.0)) if r.get('sentiment_score') is not None else 0.0,
                    int(r.get('relevance_score', 0)) if r.get('relevance_score') is not None else 0,
                    r.get('title'),
                    r.get('url')
                ))
            conn.commit()

    def store_prediction(self, pred: Dict[str, Any]):
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO predictions (timestamp, ticker, prediction, pred_price, stop_loss, extra)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                pred.get("timestamp"),
                pred.get("ticker"),
                pred.get("prediction"),
                pred.get("pred_price"),
                pred.get("stop_loss"),
                json.dumps(pred.get("extra", {}))
            ))
            conn.commit()

    def run_backtest(self, days_forward: int = 7) -> List[Dict[str, Any]]:
        rows = []
        with self._conn() as conn:
            cur = conn.execute("SELECT id, timestamp, ticker, prediction, pred_price, stop_loss FROM predictions ORDER BY id")
            for r in cur.fetchall():
                pid, ts, ticker, prediction, pred_price, stop_loss = r
                try:
                    ts_dt = datetime.fromisoformat(ts)
                except Exception:
                    continue
                check_date = ts_dt + timedelta(days=days_forward)
                try:
                    data = yf.download(ticker, start=ts_dt.date().isoformat(), end=(check_date + timedelta(days=1)).date().isoformat(), progress=False)
                    if data.empty:
                        continue
                    price_after = float(data['Close'].iloc[-1])
                    ret = (price_after - pred_price) / pred_price if pred_price else None
                    rows.append({
                        "id": pid,
                        "ticker": ticker,
                        "timestamp": ts,
                        "prediction": prediction,
                        "pred_price": pred_price,
                        "price_after": price_after,
                        "return": ret,
                        "stop_loss": stop_loss
                    })
                except Exception as e:
                    logger.debug("Backtest fetch error for %s: %s", ticker, e)
                    continue
        return rows

# ----------------------------
# Report Writer (CSV)
# ----------------------------
class ReportWriter:
    @staticmethod
    def write_csv(path: str, results: List[Dict[str, Any]]):
        if not results:
            logger.info("Keine Ergebnisse zum Schreiben.")
            return
        rows = []
        for r in results:
            rows.append({
                "ticker": r.get("ticker"),
                "last_price": r.get("last_price"),
                "rating": r.get("recommendation", {}).get("rating"),
                "confidence": r.get("recommendation", {}).get("confidence"),
                "rationale": r.get("recommendation", {}).get("rationale"),
                "stop_loss": r.get("stop_loss"),
                "correlation": r.get("correlation")
            })
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        logger.info("ReportWriter: CSV geschrieben: %s", path)
