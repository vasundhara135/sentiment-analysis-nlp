"""
train.py
Trains a TF-IDF + Logistic Regression sentiment classifier.
Run: python train.py
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score

def main():
    df = pd.read_csv("data/clean.csv")
    X = df["clean_text"]
    y = df["Sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    joblib.dump(model, "model.joblib")
    joblib.dump(vectorizer, "vectorizer.joblib")
    print("Saved model.joblib and vectorizer.joblib")

if __name__ == "__main__":
    main()
