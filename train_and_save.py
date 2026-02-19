"""Train a simple TF-IDF + classifier on the provided CSV and save
vectorizer.pkl and model.pkl in the project root.

This script tries common CSV column layouts (e.g. v1/v2 or label/text).
Run locally or as part of your Render build step.
"""
import pickle
import sys
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Ensure NLTK resources
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()
STOPWORDS = set()
try:
    STOPWORDS = set(stopwords.words('english'))
except Exception:
    print("Warning: NLTK stopwords not available; continuing without stopword filtering")


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum()]
    if STOPWORDS:
        tokens = [t for t in tokens if t not in STOPWORDS]
    stemmed = [ps.stem(t) for t in tokens]
    return " ".join(stemmed)


def load_dataset():
    # Try common filenames
    candidates = ["spam2.csv", "spam2_utf8.csv"]
    df = None
    for c in candidates:
        p = Path(c)
        if p.exists():
            try:
                df = pd.read_csv(p, encoding='utf-8', low_memory=False)
                print(f"Loaded {c} with shape {df.shape}")
                break
            except Exception:
                try:
                    df = pd.read_csv(p, encoding='latin-1', low_memory=False)
                    print(f"Loaded {c} with latin-1 encoding, shape {df.shape}")
                    break
                except Exception as e:
                    print(f"Failed to read {c}: {e}")

    if df is None:
        raise SystemExit("No dataset CSV found. Place spam2.csv or spam2_utf8.csv in the project root and retry.")

    # Attempt to infer label/text columns
    if set(['v1', 'v2']).issubset(df.columns):
        labels = df['v1'].astype(str)
        texts = df['v2'].astype(str)
    elif set(['label', 'text']).issubset(df.columns):
        labels = df['label'].astype(str)
        texts = df['text'].astype(str)
    else:
        # fallback: take first two columns
        labels = df.iloc[:, 0].astype(str)
        texts = df.iloc[:, 1].astype(str)

    # normalize labels to 0/1 where spam==1
    labels = labels.str.lower().map(lambda s: 1 if 'spam' in s else 0)
    return texts.tolist(), labels.tolist()


def main():
    texts, labels = load_dataset()
    X = [preprocess_text(t) for t in texts]

    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_valid, y_train, y_valid = train_test_split(X_vec, labels, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_valid)
    acc = accuracy_score(y_valid, preds)
    print(f"Validation accuracy: {acc:.4f}")

    # Save artifacts
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('model.pkl', 'wb') as f:
        pickle.dump(clf, f)

    print("Saved vectorizer.pkl and model.pkl to project root")


if __name__ == '__main__':
    main()
