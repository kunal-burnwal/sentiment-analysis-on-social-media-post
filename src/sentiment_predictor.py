import pickle
import re
import os
import nltk
import pandas as pd
from nltk.corpus import stopwords

# Download stopwords if not present
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
vectorizer_path = "E:/SystemAnalysisProject/models/tfidf_vectorizer.pkl"
model_path = "E:/SystemAnalysisProject/models/sentiment_model.pkl"

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Text cleaning function
def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Prediction loop
while True:
    user_input = input("\n💬 Enter a message (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("👋 Exiting sentiment predictor.")
        break

    cleaned = clean_text(user_input)
    vector = vectorizer.transform([cleaned]).toarray()
    prediction = model.predict(vector)[0]

    print(f"🧠 Predicted Sentiment: **{prediction}**")
