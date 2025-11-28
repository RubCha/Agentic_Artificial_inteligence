import os
try:
    from google import genai
except ImportError:
    raise ImportError("The 'google-genai' library is required. Install it via 'pip install google-genai'.")

import pandas as pd

def get_gemini_client():
    """
    Lazy initialization of Google Gemini API client.
    Tries to load API key from 'GEMINI_API_KEY' environment variable or config module.
    Returns a genai.Client instance (singleton).
    """
    if getattr(get_gemini_client, "client", None) is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            try:
                import config
                api_key = config.GEMINI_API_KEY
            except (ImportError, AttributeError):
                raise EnvironmentError("Gemini API key not found. Set GEMINI_API_KEY env var or define config.GEMINI_API_KEY.")
        get_gemini_client.client = genai.Client(api_key=api_key)
    return get_gemini_client.client

def neutralize_text(text, model=None, system_instruction=None):
    """
    Neutralize a single news article text by removing emotional or biased language,
    using Google Gemini API (via google-genai library). Returns the neutralized text.
    :param text: str, original news article content.
    :param model: str, optional, Gemini model to use (default from GEMINI_MODEL env or 'gemini-2.5-flash').
    :param system_instruction: str, optional, custom system instruction to guide neutralization.
    :return: str, neutralized version of the input text.
    """
    if not text or not isinstance(text, str):
        return text
    if model is None:
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    if system_instruction is None:
        system_instruction = ("You are a news editor that rewrites articles in a neutral, factual tone. "
                              "Remove any emotional language, personal opinions, or bias, leaving only objective facts.")
    try:
        client = get_gemini_client()
        response = client.models.generate_content(
            model=model,
            contents=text,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0,
                max_output_tokens=1024
            )
        )
        return response.text
    except Exception as e:
        print(f"Error neutralizing text: {e}")
        return text

def neutralize_dataframe(df, text_column='text', output_column='neutral_text', model=None, system_instruction=None):
    """
    Neutralize all news article texts in a pandas DataFrame column.
    :param df: pandas.DataFrame containing the news articles.
    :param text_column: str, name of the column with original article text.
    :param output_column: str, name of the column to add with neutralized text.
    :param model: str, optional, Gemini model to use.
    :param system_instruction: str, optional, custom system instruction for all items.
    :return: pandas.DataFrame with a new column for neutralized text.
    """
    if text_column not in df.columns:
        raise KeyError(f"Column '{text_column}' not found in DataFrame.")
    df_copy = df.copy()
    neutral_texts = []
    for idx, text in df_copy[text_column].iteritems():
        neutral = neutralize_text(text, model=model, system_instruction=system_instruction)
        neutral_texts.append(neutral)
    df_copy[output_column] = neutral_texts
    return df_copy

if _name_ == "_main_":
    # Example usage of the neutralization functions.
    example_text = ("This amazing product has delighted customers around the world! "
                    "However, some users absolutely hate the new feature.")
    print("Original Text:")
    print(example_text)
    print("\nNeutralized Text:")
    print(neutralize_text(example_text))
    # Example with DataFrame
    sample_data = {
        "title": ["Example News"],
        "text": [example_text]
    }
    df_example = pd.DataFrame(sample_data)
    df_neutral = neutralize_dataframe(df_example, text_column="text", output_column="neutral_text")
    print("\nDataFrame with Neutralized Text:")
    print(df_neutral)
    # Optionally, save the DataFrame to CSV
    # df_neutral.to_csv("neutralized_articles.csv", index=False)