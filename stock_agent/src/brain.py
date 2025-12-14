import os
import pandas as pd
import json
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

# NEU: ZWINGT DEN GEMINI-CLIENT (und damit den gesamten gRPC/SSL-Stack der Google-Libs),
# die SSL-Prüfung zu ignorieren. Dies ist die einzige Lösung, wenn der globale Python-Fix versagt.
# Dies bleibt als einziger Workaround, da Gemini/gRPC sehr empfindlich sind.
# ------------------------------------------------------------------------------------------------
import grpc
from google.auth.transport import requests
import ssl
try:
    if hasattr(ssl, '_create_unverified_context'):
        # Patche requests, das von google.auth verwendet wird, um SSL zu ignorieren
        requests.AuthorizedSession.request = lambda self, method, url, **kwargs: requests.requests.request(
            method, url, verify=False, **kwargs
        )
        # Patche grpc, das von google-genai verwendet wird
        _default_ssl_channel_credentials = grpc.ssl_channel_credentials
        grpc.ssl_channel_credentials = lambda *args, **kwargs: _default_ssl_channel_credentials(
            *args, **kwargs, root_certificates=ssl.get_default_verify_paths().default_cadata.encode('utf-8') if not args else None
        )
        print("ACHTUNG (Brain.py): Gemini/gRPC SSL-Patch angewendet.")
except Exception as e:
    print(f"Fehler beim Gemini SSL-Patch: {e}")
# ------------------------------------------------------------------------------------------------

# Notwendig für die Kommunikation mit Gemini
try:
    from google import genai
    from google.genai import types
except ImportError:
    pass

# --- Konfiguration (wird aus .env in main.py geladen) ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


class MarketAnalyst:
    """
    Diese Klasse nutzt die Gemini API, um Roh-Nachrichten zu bewerten,
    zu kategorisieren und ein Sentiment zuzuordnen (Filterung und Priorisierung).
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Initialisiert den Analysten und die Gemini-Verbindung.
        """
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY ist nicht gesetzt. Bitte Schlüssel in der .env-Datei hinterlegen.")

        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = model_name
        self.system_instruction = (
            "Du bist ein erfahrener Wall Street Analytiker und bewertest die Auswirkungen von Nachrichten "
            "auf den US-Aktienmarkt. Antworte IMMER im folgenden JSON-Format. "
            "Antworte nur mit dem JSON-Objekt, keine zusätzlichen Erklärungen oder Text."
        )
        print(f"MarketAnalyst initialisiert mit Modell: {self.model}")

    def extract_relevant_tickers(self, user_prompt: str) -> List[str]:
        """
        Nutzt Gemini, um aus einem User-Prompt eine Liste von Ticker-Symbolen zu extrahieren.
        """
        # DEFINITION DES ERWÜNSCHTEN DATENFORMATS (JSON-SCHEMA)
        json_schema = {
            "type": "object",
            "properties": {
                "tickers": {"type": "array",
                            "description": "Eine Liste von relevanten US-Aktien-Tickern, die zum Thema passen.",
                            "items": {"type": "string"}},
                "reasoning": {"type": "string", "description": "Kurze Begründung, warum diese Ticker relevant sind."}
            },
            "required": ["tickers", "reasoning"]
        }

        prompt = (
            f"Du bist ein Börsenstratege. Analysiere den folgenden Nutzerwunsch und identifiziere die 5-7 wichtigsten, "
            f"großen US-Aktien-Ticker, die für diesen Sektor oder dieses Thema am relevantesten sind. "
            f"Behandle 'KI', 'E-Mobilität' und 'Cloud' als Sektoren. "
            f"Nutzerwunsch: '{user_prompt}'"
        )

        try:
            # API-Aufruf mit striktem JSON-Schema
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    response_mime_type="application/json",
                    response_schema=json_schema
                )
            )

            result = json.loads(response.text)
            tickers = result.get('tickers', [])
            reasoning = result.get('reasoning', 'Keine Begründung verfügbar.')

            print(f"\n[Strategist] Aus dem Prompt extrahierte Ticker: {tickers}")
            print(f"[Strategist] Begründung: {reasoning}")

            return tickers

        except Exception as e:
            print(f"\nFehler bei der Ticker-Extraktion: {e}")
            return []

    def _get_analysis_prompt(self, news_text: str, ticker: str) -> Tuple[str, dict]:
        # ... (Funktion bleibt gleich)
        json_schema = {
            "type": "object",
            "properties": {
                "relevance_score": {"type": "integer",
                                    "description": "Bewertung der Relevanz für den Aktienkurs auf einer Skala von 0 (unwichtig) bis 10 (extrem wichtig)."},
                "category": {"type": "string",
                             "description": "Die relevanteste Kategorie: 'Company-Specific', 'Macroeconomics', 'Geopolitics', 'Social/ESG', 'Other'."},
                "sentiment": {"type": "string",
                              "description": "Der kurzfristige Sentiment-Impact: 'Bullish', 'Bearish' oder 'Neutral'."},
                "impact_explanation": {"type": "string",
                                       "description": "Eine kurze, prägnante Begründung für die Bewertung (max. 3 Sätze)."}
            },
            "required": ["relevance_score", "category", "sentiment", "impact_explanation"]
        }

        prompt = (
            f"Analysiere die folgende Nachricht. Der relevante Ticker ist '{ticker}' (falls MACRO, gilt es für den Gesamtmarkt). "
            "Ignoriere unwichtige, nicht-finanzielle Nachrichten (setze relevance_score auf 0). "
            f"Nachricht: \"{news_text}\""
        )

        return prompt, json_schema

    def analyze_news_batch(self, df_news: pd.DataFrame) -> pd.DataFrame:
        # ... (Funktion bleibt gleich)
        analyzed_results = []

        for index, row in tqdm(df_news.iterrows(), total=len(df_news), desc="Analysiere News mit Gemini"):
            news_text = row.get('text')
            ticker = row.get('ticker')

            if not news_text or len(news_text) < 20:
                continue

            prompt, json_schema = self._get_analysis_prompt(news_text, ticker)

            try:
                # API-Aufruf mit striktem JSON-Schema
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_instruction,
                        response_mime_type="application/json",
                        response_schema=json_schema
                    )
                )

                analysis = json.loads(response.text)
                result_row = row.to_dict()
                result_row.update(analysis)
                analyzed_results.append(result_row)

            except Exception as e:
                print(f"\nFehler bei Index {index} ({ticker}): {e}")

        if analyzed_results:
            df_analyzed = pd.DataFrame(analyzed_results)
            # Filterung: Entferne Artikel mit Relevanz 0
            df_analyzed = df_analyzed[df_analyzed.get('relevance_score', 0) > 0]

            print(f"\n{len(df_analyzed)} relevante Nachrichten nach Filterung übrig.")
            return df_analyzed
        else:
            print("Analyse abgeschlossen, aber keine Ergebnisse gefunden.")
            return pd.DataFrame()