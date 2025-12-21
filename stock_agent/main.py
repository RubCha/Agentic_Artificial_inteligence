import sys
import os

# Fügt den 'src'-Ordner explizit dem Suchpfad hinzu
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import yfinance as yf
import ssl
from datetime import datetime, timedelta
from typing import List

# DIESE ZWEI ZEILEN SIND KRITISCH, UM DIE .env-DATEI ZU LADEN
from dotenv import load_dotenv
load_dotenv()
# -----------------------------------------------------------

# Importiere die Module
from src.tools import MarketIntelFetcher
from src.brain import MarketAnalyst

# --- GLOBALER SSL-WORKAROUND (Bleibt für Standard-Libs und yfinance) ---
try:
    if hasattr(ssl, '_create_unverified_context'):
        ssl._create_default_https_context = ssl._create_unverified_context
    print("ACHTUNG: Globale SSL-Verifizierung für yfinance/Standard-Libs aktiviert.")
except Exception as e:
    pass
# -----------------------------------------------------------------------

# --- Konfiguration ---
HISTORICAL_PERIOD = "6mo"

# Optional: Stelle sicher, dass der Datenordner existiert
if not os.path.exists("data"):
    os.makedirs("data")


# --- HILFSFUNKTIONEN ---

def _analyze_historical_correlation(ticker: str, df_prices: pd.DataFrame, period: str) -> float:
    print(f"-> Analysiere historische Korrelation für {ticker}...")
    try:
        # auto_adjust=True erzwingen und 'Close' statt 'Adj Close' nutzen
        data = yf.download("SPY", period=period, interval="1d", auto_adjust=True)
        if data.empty:
            return 0.0

        market_index = data['Close']
        ticker_prices = df_prices[ticker]

        ticker_returns = ticker_prices.pct_change().dropna()
        market_returns = market_index.pct_change().dropna()

        combined = pd.concat([ticker_returns, market_returns], axis=1).dropna()
        combined.columns = [ticker, 'SPY']

        correlation = combined.corr().loc[ticker, 'SPY']
        print(f"   Aktuelle {period} Korrelation {ticker}/SPY: {correlation:.2f}")
        return float(correlation)

    except Exception as e:
        print(f"   Fehler bei der Korrelationsanalyse für {ticker}: {e}")
        return 0.0

def _aggregate_current_news(ticker: str, df_news: pd.DataFrame) -> str:
    """
    Fasst die relevanten News für den Ticker und das Makroumfeld zusammen.
    """
    company_news = df_news[df_news['ticker'] == ticker].sort_values(by='relevance_score', ascending=False).head(5)
    macro_news = df_news[df_news['ticker'] == 'MACRO'].sort_values(by='relevance_score', ascending=False).head(3)

    summary = f"Wichtigste News für {ticker} (basierend auf {len(company_news)} Artikeln):\n"
    for _, row in company_news.iterrows():
        impact = row.get('impact_explanation', 'Keine Begründung verfügbar.')
        summary += f"- COMPANY: Score {row['relevance_score']}, Sentiment {row['sentiment']}: {row['title']} ({impact})\n"

    summary += f"\nWichtigstes Makro-Umfeld (basierend auf {len(macro_news)} Artikeln):\n"
    for _, row in macro_news.iterrows():
        impact = row.get('impact_explanation', 'Keine Begründung verfügbar.')
        summary += f"- MACRO: Score {row['relevance_score']}, Sentiment {row['sentiment']}: {row['title']} ({impact})\n"

    return summary


def _make_final_prediction(analyst: MarketAnalyst, ticker: str, news_summary: str, corr_val: float):
    corr_info = f"Die historische Korrelation von {ticker} mit dem S&P 500 beträgt {corr_val:.2f}."

    final_prompt = (
        f"Du bist ein Senior-Portfolio-Manager. Basierend auf diesen Daten, gib eine Vorhersage für {ticker} (1-5 Tage).\n"
        f"Daten:\n{news_summary}\n{corr_info}\n"
        "Antworte mit einer klaren Vorhersage (Bullish/Bearish/Neutral) und Begründung."
    )

    try:
        # WICHTIG: Nutze die Rotation auch hier!
        client = analyst._get_random_client()

        # Hinweis: 'gemini-2.5-flash' ist stabiler als die Pro-Bezeichnung
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=final_prompt
        )
        print(f"\n\n=== FINALE VORHERSAGE FÜR {ticker} ===")
        print(response.text)
        print("=======================================")

    except Exception as e:
        print(f"Fehler bei der finalen Vorhersage für {ticker}: {e}")


# --- Haupt-Agenten-Logik (Interaktiv) ---

def run_agent(analyst: MarketAnalyst, user_prompt: str):
    """
    Führt den Agenten-Analysezyklus dynamisch aus.
    """

    # ------------------------------------------------------------------------
    # 1. STRATEGIE: Dynamische Ticker-Auswahl (KEIN FALLBACK)
    # ------------------------------------------------------------------------

    # a) Versuch, Ticker aus dem Prompt zu extrahieren (Gemini-Client)
    print(f"\n[Strategist] Analysiere Prompt: '{user_prompt}'")
    target_tickers = analyst.extract_relevant_tickers(user_prompt)

    if not target_tickers:
        print("\n[Strategist] Konnte keine relevanten Ticker finden. Analyse abgebrochen.")
        return

    # 2. DATENBESCHAFFUNG
    fetcher = MarketIntelFetcher(tickers=target_tickers)
    print("\n--- 1. BESCHAFFUNG VON ROHDATEN (MarketIntelFetcher) ---")

    df_news = fetcher.fetch_all_market_news()
    df_prices = fetcher.fetch_historical_prices(period=HISTORICAL_PERIOD)

    # Jetzt prüfen wir die tatsächlichen Ticker, die Daten hatten
    available_tickers = [t for t in target_tickers if t in df_prices.columns]

    if df_news.empty or not available_tickers:
        print("\n[Fehler] Datenbeschaffung fehlgeschlagen (entweder News oder Kursdaten fehlen).")
        return

    # 3. ANALYSE & FILTERUNG
    print("\n--- 2. ANALYSE UND SENTIMENT-BESTIMMUNG (MarketAnalyst) ---")
    df_analyzed_news = analyst.analyze_news_batch(df_news)

    if df_analyzed_news.empty:
        print("\n[Ergebnis] Keine relevanten News nach Sentiment-Analyse übrig.")
        return

    df_analyzed_news.to_csv("data/analyzed_news_latest.csv", index=False)
    print("Analysierte News zur Kontrolle in data/analyzed_news_latest.csv gespeichert.")

    # 4. SYNTHESE & ENDAUSGABE
    print("\n--- 3. SYNTHESE UND VORHERSAGE (Gemini-2.5-Pro) ---")

    # Durchlaufe jeden Ticker für die finale Analyse
    for ticker in available_tickers:
        # 4a. Korrelation berechnen
        corr_val = _analyze_historical_correlation(ticker, df_prices, HISTORICAL_PERIOD)

        # 4b. News zusammenfassen
        news_summary = _aggregate_current_news(ticker, df_analyzed_news)

        # 4c. Finale Vorhersage durch Gemini-Pro
        _make_final_prediction(analyst, ticker, news_summary, corr_val)


# --- AGENT START (Interaktive Schleife) ---
if __name__ == "__main__":

    print("=========================================================")
    print("       Aladdin Lite Agent - Interaktiver Marktanalyst     ")
    print("=========================================================")
    print("  Beispiel-Prompts: 'Analysiere den KI-Sektor' oder      ")
    print("  'Finde Unternehmen, die von fallenden Zinsen profitieren'")
    print("  Gib 'exit' oder 'quit' ein, um das Programm zu beenden.")
    print("=========================================================")

    # Initialisiere MarketAnalyst NUR EINMAL
    try:
        analyst = MarketAnalyst()
    except ValueError as e:
        print(f"\n--- FEHLER IM AGENTEN-START ---")
        print(f"BITTE PRÜFEN SIE DIE .ENV-DATEI UND DEN API-SCHLÜSSEL: {e}")
        sys.exit(1)

    while True:
        try:
            user_prompt = input("\n[Eingabe] Dein Analyse-Wunsch (Prompt): ")

            if user_prompt.lower() in ['exit', 'quit']:
                print("\nAgent beendet. Auf Wiedersehen!")
                break

            if not user_prompt:
                continue

            # Starte den Agenten-Ablauf mit der Analyst-Instanz
            run_agent(analyst, user_prompt)

        except KeyboardInterrupt:
            print("\nAgent beendet durch Benutzer (Ctrl+C). Auf Wiedersehen!")
            break
        except Exception as e:
            # Ein Fehler in der Schleife soll nicht zum Absturz führen
            print(f"\nEin kritischer Fehler ist im Hauptprozess aufgetreten: {e}")
            import traceback

            traceback.print_exc()
            print("Setze den Agenten für eine neue Eingabe zurück...")