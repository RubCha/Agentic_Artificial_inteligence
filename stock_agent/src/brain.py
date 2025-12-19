import os
import pandas as pd
import json
import time
import random
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

# --- SSL-PATCH (Beibehalten für Stabilität) ---
import grpc
from google.auth.transport import requests
import ssl

try:
    if hasattr(ssl, '_create_unverified_context'):
        requests.AuthorizedSession.request = lambda self, method, url, **kwargs: requests.requests.request(
            method, url, verify=False, **kwargs
        )
        _default_ssl_channel_credentials = grpc.ssl_channel_credentials
        grpc.ssl_channel_credentials = lambda *args, **kwargs: _default_ssl_channel_credentials(
            *args, **kwargs,
            root_certificates=ssl.get_default_verify_paths().default_cadata.encode('utf-8') if not args else None
        )
        print("ACHTUNG (Brain.py): Gemini/gRPC SSL-Patch angewendet.")
except Exception as e:
    print(f"Fehler beim Gemini SSL-Patch: {e}")
# ------------------------------------------------

# Notwendig für die Kommunikation mit Gemini
try:
    from google import genai
    from google.genai import types
except ImportError:
    pass

# --- KEY-ROTATION SETUP ---
# Wir laden die Liste der Keys und entfernen Leerzeichen/leere Einträge
GEMINI_API_KEYS = [key.strip() for key in os.environ.get("GEMINI_API_KEYS", "").split(',') if key.strip()]
# Fallback auf den Einzel-Key, falls die Liste aus irgendeinem Grund nicht lädt
SINGLE_KEY = os.environ.get("GEMINI_API_KEY")


class MarketAnalyst:
    """
    Diese Klasse nutzt die Gemini API mit KEY-ROTATION, um Ratenlimits zu umgehen.
    """

    def extract_relevant_tickers(self, user_prompt: str) -> List[str]:
        """
        Nutzt Gemini, um Ticker zu extrahieren.
        Versucht es bei Server-Überlastung (503) bis zu 3-mal mit verschiedenen Keys.
        """
        json_schema = {
            "type": "object",
            "properties": {
                "tickers": {"type": "array", "items": {"type": "string"}},
                "reasoning": {"type": "string"}
            },
            "required": ["tickers", "reasoning"]
        }

        prompt = (
            f"Du bist ein Börsenstratege. Analysiere den folgenden Nutzerwunsch und identifiziere die 5-7 wichtigsten, "
            f"großen US-Aktien-Ticker, die für diesen Sektor am relevantesten sind.\n"
            f"Nutzerwunsch: '{user_prompt}'"
        )

        # RETRY-LOGIK (Versuche es 3-mal bei Fehlern wie 503)
        for attempt in range(3):
            client = self._get_random_client()
            try:
                response = client.models.generate_content(
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
                reasoning = result.get('reasoning', 'Keine Begründung.')

                print(f"\n[Strategist] Versuch {attempt + 1} erfolgreich.")
                print(f"[Strategist] Ticker: {tickers}")
                print(f"[Strategist] Begründung: {reasoning}")

                return tickers

            except Exception as e:
                print(f"\n[Strategist] Versuch {attempt + 1} fehlgeschlagen (Fehler: {e}).")
                if attempt < 2:
                    print("Probiere es mit einem anderen Key erneut...")
                    time.sleep(2)  # Kurz warten, bevor der nächste Key probiert wird
                else:
                    print("Alle Versuche für die Ticker-Extraktion sind fehlgeschlagen.")

        return []

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        # Nutze die Liste oder den Einzel-Key
        self.keys = GEMINI_API_KEYS if GEMINI_API_KEYS else ([SINGLE_KEY] if SINGLE_KEY else [])

        if not self.keys:
            raise ValueError("Weder GEMINI_API_KEYS noch GEMINI_API_KEY gefunden!")

        # Wir initialisieren für JEDEN Key einen eigenen Client im Voraus
        self.clients = [genai.Client(api_key=k) for k in self.keys]
        self.model = model_name
        self.system_instruction = (
            "Du bist ein erfahrener Wall Street Analytiker und bewertest die Auswirkungen von Nachrichten "
            "auf den US-Aktienmarkt. Antworte IMMER im folgenden JSON-Format. "
            "Antworte nur mit dem JSON-Objekt, keine zusätzlichen Erklärungen oder Text."
        )
        print(f"MarketAnalyst initialisiert mit {len(self.clients)} Gemini-Clients (Keys).")

    def _get_random_client(self):
        """Wählt für jede Anfrage einen zufälligen Client aus dem Pool."""
        return random.choice(self.clients)

    def extract_relevant_tickers(self, user_prompt: str) -> List[str]:
        """Extrahiert Ticker-Symbole mit einem rotierenden Client."""
        json_schema = {
            "type": "object",
            "properties": {
                "tickers": {"type": "array", "items": {"type": "string"}},
                "reasoning": {"type": "string"}
            },
            "required": ["tickers", "reasoning"]
        }

        prompt = (f"Identifiziere 5-7 US-Aktien-Ticker für: '{user_prompt}'")

        client = self._get_random_client()
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    response_mime_type="application/json",
                    response_schema=json_schema
                )
            )
            result = json.loads(response.text)
            print(f"\n[Strategist] Ticker: {result.get('tickers')}")
            return result.get('tickers', [])
        except Exception as e:
            print(f"Fehler Ticker-Extraktion: {e}")
            return []

    def _get_analysis_prompt(self, news_text: str, ticker: str):
        json_schema = {
            "type": "object",
            "properties": {
                "relevance_score": {"type": "integer"},
                "category": {"type": "string"},
                "sentiment": {"type": "string"},
                "impact_explanation": {"type": "string"}
            },
            "required": ["relevance_score", "category", "sentiment", "impact_explanation"]
        }
        prompt = (f"Analysiere News für '{ticker}': \"{news_text}\"")
        return prompt, json_schema

    def analyze_news_batch(self, df_news: pd.DataFrame) -> pd.DataFrame:
        analyzed_results = []

        for index, row in tqdm(df_news.iterrows(), total=len(df_news), desc="Analysiere News mit Gemini"):
            news_text = row.get('text')
            ticker = row.get('ticker')

            if not news_text or len(news_text) < 20:
                continue

            prompt, json_schema = self._get_analysis_prompt(news_text, ticker)

            # ROTATION: Wähle für jede einzelne Nachricht einen neuen Key
            client = self._get_random_client()

            try:
                response = client.models.generate_content(
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

                # KLEINER SCHUTZ: 0.2s Pause, damit Google dich nicht per IP blockt
                time.sleep(0.2)

            except Exception as e:
                if "429" in str(e):
                    # Falls ein Key doch voll ist, kurz warten und weitermachen (Rotation regelt)
                    time.sleep(1)
                else:
                    print(f"\nFehler bei Index {index} ({ticker}): {e}")

        if analyzed_results:
            df_analyzed = pd.DataFrame(analyzed_results)
            df_analyzed = df_analyzed[df_analyzed.get('relevance_score', 0) > 0]
            print(f"\n{len(df_analyzed)} relevante Nachrichten nach Filterung übrig.")
            return df_analyzed
        return pd.DataFrame()