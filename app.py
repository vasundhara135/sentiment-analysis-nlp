
import streamlit as st
import joblib
from preprocess import clean_text

st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="💬",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    return model, vectorizer

try:
    model, vectorizer = load_artifacts()
except Exception as error:
    st.error(f"Could not load the model files: {error}")
    st.info("Ensure model.joblib and vectorizer.joblib are in the project folder.")
    st.stop()

st.title("💬 Sentiment Analysis Dashboard")
st.write(
    "Analyze product reviews and social media posts using "
    "a machine learning text classifier."
)

st.caption("Model: TF-IDF + Logistic Regression")

text = st.text_area(
    "Enter a review or social media post",
    placeholder="Example: This product is excellent and easy to use!",
    height=150
)

if st.button("Analyze Sentiment", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        cleaned = clean_text(text)

        if not cleaned:
            st.warning("Please enter text containing words that can be analyzed.")
        else:
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]
            probabilities = model.predict_proba(vec)[0]
            confidence = float(probabilities[list(model.classes_).index(pred)])

            st.subheader("Analysis Result")

            if pred == 1:
                st.success("Positive Sentiment 😊")
            else:
                st.error("Negative Sentiment 😞")

            st.metric("Model confidence", f"{confidence:.1%}")
            st.progress(confidence)

            with st.expander("View processed text"):
                st.write(cleaned)

            st.caption(
                "This is a machine learning prediction and may be incorrect, "
                "especially for sarcasm, mixed opinions, or unfamiliar wording."
            )
