
# Sentiment Analysis on Social Media and Product Reviews

A machine learning project that classifies text as positive or negative using Natural Language Processing (NLP), TF-IDF feature extraction, and Logistic Regression.

## Features

- Predicts positive or negative sentiment.
- Provides a confidence score for predictions.
- Includes an interactive Streamlit web application.
- Cleans text by removing URLs, mentions, and unnecessary characters.
- Supports model training and command-line prediction.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Streamlit
- Joblib

## Project Structure

```text
sentiment-analysis-nlp/
├── app.py
├── predict.py
├── preprocess.py
├── train.py
├── model.joblib
├── vectorizer.joblib
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup and Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/YOUR_USERNAME/sentiment-analysis-nlp.git
   cd sentiment-analysis-nlp
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

## Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The app opens in your browser. Enter a product review or social media post and click **Analyze Sentiment**.

## Make a Prediction from the Command Line

```bash
python predict.py "This product is amazing!"
```

## Training the Model

The training pipeline uses a labeled sentiment dataset.

1. Place the permitted dataset at `data/raw.csv`.
2. Run preprocessing:

   ```bash
   python preprocess.py
   ```

3. Train the model:

   ```bash
   python train.py
   ```

The training process generates the model and vectorizer files used by the application. The dataset is not included in this repository; obtain it from an authorized source before retraining.

## Limitations

- The classifier predicts only positive or negative sentiment.
- Sarcasm, mixed opinions, slang, and context-dependent language may affect predictions.
- Confidence scores are model estimates and are not guaranteed to represent actual accuracy.

## Disclaimer

This project is intended for educational and demonstration purposes. Predictions may be incorrect and should not be treated as definitive judgments of a person's opinion.
