#!/usr/bin/env python3
"""
main.py - Orchestrator für Aladdin Lite (Terminal-only)

Flow:
 - Lädt .env
 - Initialisiert Key-Pools, Fetcher, Analyst, DB
 - Interaktive CLI: Prompt -> Extrahiere Ticker -> Sammle Daten -> Analysiere -> Speichere -> Report (CSV)
 - Optional: automatischer Testmodus (--test)
"""
import os
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(__file__)
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("aladdin")

# Lokale Module
from src.tools import RotatingKeyPool, MarketIntelFetcher, HistoricalTools, DBManager, ReportWriter
from src.brain import MarketAnalyst

# Konfiguration
HISTORICAL_PERIOD = os.getenv("HISTORICAL_PERIOD", "6mo")
THROTTLE_SECONDS = float(os.getenv("THROTTLE_SECONDS", "0.4"))

# Ensure folders
os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)

def run_agent(analyst: MarketAnalyst, fetcher: MarketIntelFetcher, db: DBManager, prompt: str, test_tickers: list = None):
    logger.info("Prompt erhalten: %s", prompt)
    tickers = test_tickers or analyst.extract_relevant_tickers(prompt)
    if not tickers:
        logger.warning("Keine Ticker extrahiert. Abbruch.")
        return

    logger.info("Extrahierte Ticker: %s", ", ".join(tickers))

    # 1) News & Prices
    logger.info("Sammle News...")
    df_news = fetcher.fetch_all_market_news(tickers)
    time.sleep(THROTTLE_SECONDS)

    logger.info("Sammle historische Preise...")
    df_prices = fetcher.fetch_historical_prices(tickers, period=HISTORICAL_PERIOD)
    time.sleep(THROTTLE_SECONDS)

    if df_prices.empty:
        logger.error("Keine Kursdaten verfügbar. Abbruch.")
        return

    # 2) Sentiment-Analyse
    logger.info("Analysiere News (Sentiment)...")
    df_analyzed = analyst.analyze_news_batch(df_news)
    if not df_analyzed.empty:
        db.store_sentiments(df_analyzed)
        analyzed_csv = os.path.join(ROOT, "data", "analyzed_news_latest.csv")
        df_analyzed.to_csv(analyzed_csv, index=False)
        logger.info("Analysierte News gespeichert: %s (%d rows)", analyzed_csv, len(df_analyzed))
    else:
        logger.info("Keine analysierten News (leer).")

    # 3) Pro Ticker: Indikatoren, Fundamentals, Empfehlung, Speicherung
    results = []
    for ticker in tickers:
        t_upper = ticker.upper()
        if t_upper not in df_prices.columns:
            logger.warning("Keine Preisdaten für %s – übersprungen.", t_upper)
            continue

        prices = df_prices[[t_upper]].dropna()
        indicators = HistoricalTools.compute_indicators(prices[t_upper])
        fundamentals = fetcher.fetch_fundamentals(t_upper)
        correlation = HistoricalTools.compute_correlation_with_spy(df_prices, t_upper, period=HISTORICAL_PERIOD)

        news_for_ticker = df_analyzed[df_analyzed['ticker'] == t_upper] if not df_analyzed.empty else df_analyzed

        rec = analyst.make_final_recommendation(
            ticker=t_upper,
            prices=prices,
            indicators=indicators,
            fundamentals=fundamentals,
            news_df=news_for_ticker,
            correlation=correlation
        )

        stop_loss = HistoricalTools.atr_stop_loss(prices[t_upper], multiplier=3)

        db.store_prediction({
            "timestamp": datetime.utcnow().isoformat(),
            "ticker": t_upper,
            "prediction": rec.get("rating", "UNKNOWN"),
            "pred_price": float(prices[t_upper].iloc[-1]),
            "stop_loss": stop_loss,
            "extra": rec
        })

        results.append({
            "ticker": t_upper,
            "last_price": float(prices[t_upper].iloc[-1]),
            "recommendation": rec,
            "stop_loss": stop_loss,
            "correlation": correlation,
            "fundamentals": fundamentals,
            "indicators": indicators
        })

        time.sleep(THROTTLE_SECONDS)

    # 4) Report speichern
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(ROOT, "reports", f"aladdin_report_{timestamp}.csv")
    ReportWriter.write_csv(report_path, results)
    logger.info("Report gespeichert: %s", report_path)

    # 5) Optional: Backtest (7 Tage)
    logger.info("Starte Backtest (7 Tage) – kurze Übersicht:")
    bt = db.run_backtest(days_forward=7)
    if bt:
        logger.info("Backtest-Ergebnisse (letzte 10):")
        for row in bt[-10:]:
            logger.info("%s", row)
    else:
        logger.info("Keine Backtest-Daten gefunden.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Aladdin Lite Terminal")
    parser.add_argument("--test", action="store_true", help="Starte automatischen Testlauf mit Beispiel-Tickern")
    args = parser.parse_args()

    logger.info("Starte Aladdin Lite (Terminal)")

    # Init
    db = DBManager(os.path.join(ROOT, "data", "aladdin.db"))
    gem_pool = RotatingKeyPool.from_env("GEMINI_API_KEYS", throttle=THROTTLE_SECONDS)
    finhub_pool = RotatingKeyPool.from_env("FINNHUB_API_KEYS", throttle=THROTTLE_SECONDS)
    news_pool = RotatingKeyPool.from_env("NEWS_API_KEYS", throttle=THROTTLE_SECONDS)

    fetcher = MarketIntelFetcher(finnhub_pool=finhub_pool, news_pool=news_pool)
    analyst = MarketAnalyst(gemini_pool=gem_pool)

    if args.test:
        logger.info("Starte Testmodus mit Beispiel-Tickern...")
        test_prompt = "Check Apple, Tesla and Microsoft news and fundamentals"
        test_tickers = ["AAPL", "TSLA", "MSFT"]
        run_agent(analyst, fetcher, db, test_prompt, test_tickers=test_tickers)
    else:
        try:
            while True:
                prompt = input("\n[Eingabe] Analyse-Wunsch (oder 'exit'): ").strip()
                if prompt.lower() in ("exit", "quit"):
                    logger.info("Beende Aladdin Lite.")
                    break
                if not prompt:
                    continue
                run_agent(analyst, fetcher, db, prompt)
        except KeyboardInterrupt:
            logger.info("Benutzerabbruch. Ende.")
