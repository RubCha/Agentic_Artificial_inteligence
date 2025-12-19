# src/brain.py
"""
MarketAnalyst: LLM-Wrapper (gemini) optional + rule-based fallback.
- extract_relevant_tickers(prompt)
- analyze_news_batch(df_news) -> adds relevance/sentiment
- make_final_recommendation(...) -> returns {rating, confidence, rationale, stop_loss}
"""
import os
import time
import json
import logging
from typing import List, Dict, Any

import pandas as pd

logger = logging.getLogger("aladdin.brain")
LLM_THROTTLE = float(os.getenv("LLM_THROTTLE", "0.4"))

# Try genai import for Gemini (optional)
GENAI_AVAILABLE = False
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
    logger.info("google-genai verfügbar. Gemini-Integration aktiviert.")
except Exception:
    logger.info("google-genai nicht verfügbar. Fallback-Modus aktiv.")

from tools import RotatingKeyPool, HistoricalTools

# Simple naive sentiment (fallback)
POS_WORDS = {"good", "positive", "beat", "beats", "upgraded", "growth", "rise", "profit", "record", "outperform"}
NEG_WORDS = {"bad", "negative", "miss", "missed", "downgrade", "down", "loss", "fall", "decline", "lawsuit", "recall"}

def naive_sentiment_score(text: str) -> float:
    t = (text or "").lower()
    pos = sum(t.count(w) for w in POS_WORDS)
    neg = sum(t.count(w) for w in NEG_WORDS)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)

class GeminiClientWrapper:
    def __init__(self, key_pool: RotatingKeyPool, model_default: str = "gemini-2.5-flash"):
        self.pool = key_pool
        self.model_default = model_default
        self.clients = {}  # lazy clients

    def _get_client(self, key: str):
        if not GENAI_AVAILABLE:
            raise RuntimeError("genai nicht installiert")
        if key not in self.clients:
            self.clients[key] = genai.Client(api_key=key)
        return self.clients[key]

    def generate_json(self, prompt: str, json_schema: Dict[str, Any], model: str = None, max_retries: int = 5) -> Dict[
        str, Any]:
        if not GENAI_AVAILABLE:
            raise RuntimeError("genai nicht installiert")

        model = model or self.model_default
        attempts = 0

        while attempts < max_retries:
            key = None
            try:
                key = self.pool.next()  # Holt den nächsten freien Key inkl. 0.4s Pause
                client = self._get_client(key)

                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction="Du bist ein präziser Finanzanalyst. Antworte ausschließlich im JSON-Format.",
                        response_mime_type="application/json",
                        response_schema=json_schema
                    )
                )

                if not response.text:
                    raise ValueError("Leere Antwort von Gemini")

                return json.loads(response.text)

            except Exception as e:
                err_str = str(e).lower()
                # Spezifischer Check auf Rate-Limits (429)
                if "429" in err_str or "resource" in err_str or "exhausted" in err_str:
                    logger.error(f"RATE LIMIT bei Key {key[:8]}... - Markiere Key als BAD.")
                    if key:
                        self.pool.mark_bad(key, seconds=60)
                else:
                    logger.warning(f"Anderer Gemini Fehler mit Key {key[:8] if key else 'None'}: {e}")

                attempts += 1
                # Kurze exponentielle Pause vor dem nächsten Versuch mit neuem Key
                time.sleep(0.5 * attempts)
                continue

        raise RuntimeError(f"Gemini-Aufrufe nach {max_retries} Versuchen über verschiedene Keys fehlgeschlagen.")

class MarketAnalyst:
    def __init__(self, gemini_pool: RotatingKeyPool = None, model_name: str = "gemini-2.5-flash"):
        self.pool = gemini_pool or RotatingKeyPool([])
        self.model = model_name
        if GENAI_AVAILABLE and self.pool and self.pool.keys:
            self.client = GeminiClientWrapper(self.pool, model_default=self.model)
        else:
            self.client = None
            logger.info("LLM nicht verfügbar oder keine Keys - Verwende Fallback-Logik.")

    def extract_relevant_tickers(self, prompt: str) -> List[str]:
        found_tickers = []

        # 1. Versuch: Gemini (Die elegante Lösung)
        if self.client:
            schema = {
                "type": "object",
                "properties": {
                    "tickers": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["tickers"]
            }
            try:
                res = self.client.generate_json(prompt, schema, model=self.model)
                found_tickers = [t.strip().upper() for t in res.get("tickers", [])]
            except Exception:
                # Falls 429 Error, gehen wir direkt zum Fallback
                pass

        # 2. Allgemeiner Fallback (Wenn Gemini blockiert ist)
        if not found_tickers:
            # Alles was wir ignorieren wollen (erweiterbar)
            stop_words = {"UND", "DIE", "DER", "DAS", "WIE", "IST", "SOLL", "WENN", "ABER", "VON", "ZUM", "MIT", "DEN"}

            # Wir teilen den Satz und säubern die Wörter
            words = [w.strip(".,()\"'!?") for w in prompt.split()]

            for w in words:
                w_upper = w.upper()

                # Ein Ticker ist meistens 1-5 Zeichen lang, nur Buchstaben
                # UND er darf kein deutsches Füllwort sein
                if 1 <= len(w) <= 5 and w.isalpha() and w_upper not in stop_words:
                    found_tickers.append(w_upper)

                # Spezialfall: Wenn jemand "Apple" schreibt, aber Gemini tot ist:
                # Hier könnte man eine kleine Library wie 'yfinance' Ticker-Suche nutzen,
                # aber für den Moment reicht die Erkennung von 1-5 Zeichen.

        # 3. Validierung (Optional: Nur was Yahoo wirklich kennt)
        # Das verhindert, dass "KAUFEN" als Ticker durchgeht
        unique_tickers = list(dict.fromkeys(found_tickers))
        return unique_tickers[:7]

    def analyze_news_batch(self, df_news: pd.DataFrame) -> pd.DataFrame:
        """
        Analysiert News in 10er-Batches pro Ticker.
        Spart ca. 90% der API-Calls und schont die Key-Pools.
        """
        if df_news is None or df_news.empty:
            return pd.DataFrame()

        all_analyzed_rows = []

        # Wir gruppieren nach Ticker, um den Kontext zu wahren
        for ticker, group in df_news.groupby('ticker'):
            ticker_upper = str(ticker).upper()

            # Teile die News des Tickers in 10er Blöcke
            for i in range(0, len(group), 10):
                chunk = group.iloc[i:i + 10]

                if self.client:
                    # Definiere das Schema für die Batch-Antwort
                    schema = {
                        "type": "object",
                        "properties": {
                            "analyses": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "url": {"type": "string"},
                                        "relevance_score": {"type": "integer"},
                                        "sentiment": {"type": "string"},
                                        "sentiment_score": {"type": "number"},
                                        "impact_explanation": {"type": "string"}
                                    },
                                    "required": ["url", "relevance_score", "sentiment", "sentiment_score",
                                                 "impact_explanation"]
                                }
                            }
                        },
                        "required": ["analyses"]
                    }

                    # Erstelle eine kompakte Liste für den Prompt
                    batch_data = []
                    for _, r in chunk.iterrows():
                        batch_data.append({
                            "url": r.get("url"),
                            "title": r.get("title", "")[:200],
                            "text": r.get("text", "")[:300]
                        })

                    prompt = f"Analysiere diese {len(batch_data)} Nachrichten für {ticker_upper}. " \
                             f"Antworte im JSON-Format mit einer Liste von Analysen pro URL."

                    try:
                        # Ein einziger Call für bis zu 10 News
                        res = self.client.generate_json(prompt, schema, model=self.model)
                        analysis_results = {a['url']: a for a in res.get("analyses", [])}

                        for _, r in chunk.iterrows():
                            ana = analysis_results.get(r['url'])
                            if ana:
                                all_analyzed_rows.append({
                                    **r.to_dict(),
                                    "relevance_score": int(ana.get("relevance_score", 0)),
                                    "sentiment": ana.get("sentiment"),
                                    "sentiment_score": float(ana.get("sentiment_score", 0.0)),
                                    "impact_explanation": ana.get("impact_explanation", "")
                                })
                            else:
                                # Einzel-Fallback falls eine URL im JSON-Batch fehlt
                                score = naive_sentiment_score((r.get("title") or "") + " " + (r.get("text") or ""))
                                all_analyzed_rows.append(
                                    self._create_fallback_row(r, ticker_upper, score, "Batch-Missing"))
                        continue  # Erfolgreich verarbeitet

                    except Exception as e:
                        logger.warning(f"Batch-Analyse fehlgeschlagen für {ticker_upper}: {e}")

                # --- Fallback für den gesamten Chunk (API-Error oder kein Client) ---
                for _, r in chunk.iterrows():
                    text = (r.get("title") or "") + " " + (r.get("text") or "")
                    score = naive_sentiment_score(text)
                    all_analyzed_rows.append(self._create_fallback_row(r, ticker_upper, score, "LLM Fallback"))

                time.sleep(LLM_THROTTLE)

        df_final = pd.DataFrame(all_analyzed_rows)
        # Filter nach Relevanz, falls Daten vorhanden
        return df_final[df_final["relevance_score"] > 0] if not df_final.empty else df_final

    def _create_fallback_row(self, r, ticker, score, reason):
        """Hilfsmethode zur Erzeugung konsistenter Fallback-Daten."""
        return {
            **r.to_dict(),
            "ticker": ticker,
            "relevance_score": 50,
            "sentiment": "Positive" if score > 0 else ("Negative" if score < 0 else "Neutral"),
            "sentiment_score": float(score),
            "impact_explanation": reason
        }

    def make_final_recommendation(self, ticker: str, prices: pd.DataFrame, indicators: Dict[str, Any],
                                  fundamentals: Dict[str, Any], news_df: pd.DataFrame, correlation: float) -> Dict[str, Any]:
        ctx = {
            "ticker": ticker,
            "last_price": float(prices.iloc[-1, 0]) if not prices.empty else None,
            "indicators": indicators,
            "fundamentals": fundamentals,
            "correlation": correlation,
            "recent_sentiment_mean": float(news_df["sentiment_score"].mean()) if (news_df is not None and not news_df.empty) else 0.0,
            "news_count": int(len(news_df)) if news_df is not None else 0
        }

        if self.client:
            schema = {
                "type": "object",
                "properties": {
                    "rating": {"type": "string"},
                    "confidence": {"type": "number"},
                    "rationale": {"type": "string"},
                    "stop_loss": {"type": "number"}
                },
                "required": ["rating", "confidence", "rationale"]
            }
            prompt = f"Du bist Senior Portfolio Manager. Hier strukturierte Daten:\n{json.dumps(ctx)}\nGib JSON mit rating (BUY/HOLD/SELL), confidence(0..1), rationale und optional stop_loss."
            try:
                res = self.client.generate_json(prompt, schema, model=self.model)
                return {
                    "rating": res.get("rating"),
                    "confidence": float(res.get("confidence", 0.5)),
                    "rationale": res.get("rationale"),
                    "stop_loss": float(res.get("stop_loss")) if res.get("stop_loss") else None
                }
            except Exception as e:
                logger.warning("Gemini final synthesis failed: %s - fallback to rule", e)

        # Rule-based fallback
        score = 0.0
        score += ctx["recent_sentiment_mean"] * 1.5
        macd = indicators.get("macd")
        if macd is not None:
            score += 0.01 * macd
        rsi = indicators.get("rsi_14")
        if rsi:
            if rsi > 70:
                score -= 0.5
            elif rsi < 30:
                score += 0.5
        pe = fundamentals.get("peRatio")
        if pe and pe < 30:
            score += 0.2
        score = score * (1 - 0.2 * abs(correlation))

        if score > 0.4:
            rating = "BUY"
        elif score < -0.4:
            rating = "SELL"
        else:
            rating = "HOLD"
        conf = min(0.99, max(0.05, 0.5 + score / 2))
        rationale = f"Rule-based: sentiment={ctx['recent_sentiment_mean']:.2f}, macd={macd}, rsi={rsi}, pe={pe}"
        return {"rating": rating, "confidence": conf, "rationale": rationale, "stop_loss": None}
