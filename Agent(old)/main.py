from pipeline import run_pipeline

if __name__ == "__main__":
    print("Starte News- und Marktdaten-Pipeline…")
    df = run_pipeline()
    print("Fertig! Anzahl gesammelter News:", len(df))
