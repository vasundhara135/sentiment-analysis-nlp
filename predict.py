"""
predict.py
Loads the saved model and predicts sentiment for new text.
Run: python predict.py "This product is amazing"
"""
import sys
import joblib
from preprocess import clean_text

def predict(text):
    model = joblib.load("model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    label = "Positive" if pred == 1 else "Negative"
    confidence = prob[pred]
    return label, confidence

if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "I love this product, it works great!"
    label, confidence = predict(text)
    print(f"Text: {text}")
    print(f"Sentiment: {label} (confidence: {confidence:.2%})")
