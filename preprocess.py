"""
preprocess.py
Cleans raw text data for sentiment analysis.
Run: python preprocess.py
"""
import re
import pandas as pd

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)      # remove URLs
    text = re.sub(r"@\w+", "", text)                 # remove mentions
    text = re.sub(r"#(\w+)", r"\1", text)             # keep hashtag word, drop #
    text = re.sub(r"[^a-z\s]", " ", text)             # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()          # collapse whitespace
    return text

def load_and_clean(path, sample_size=50000, seed=42):
    df = pd.read_csv(path, encoding="latin-1")
    df = df[["Sentiment", "SentimentText"]].dropna()
    df = df.sample(n=min(sample_size, len(df)), random_state=seed)  # subsample for speed
    df["clean_text"] = df["SentimentText"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0]
    return df

if __name__ == "__main__":
    df = load_and_clean("data/raw.csv")
    df.to_csv("data/clean.csv", index=False)
    print(f"Saved {len(df)} cleaned rows to data/clean.csv")
    print(df["Sentiment"].value_counts())
