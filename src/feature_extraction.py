import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

def main():
    input_file = "E:\SystemAnalysisProject\data\cleaned_sentiment_data.csv"
    vectorizer_file = os.path.join("..", "models", "tfidf_vectorizer.pkl")

    print("Loading cleaned data...")
    df = pd.read_csv(input_file)

    # ✅ Remove bad entries
    df = df.dropna(subset=['cleaned_text'])
    df = df[df['cleaned_text'].str.strip() != '']

    # ✅ Limit to 20,000 tweets to avoid memory crash
    df = df.head(20000)

    print("Transforming text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['cleaned_text']).toarray()

    with open(vectorizer_file, 'wb') as f:
        pickle.dump(vectorizer, f)

    print(f"✅ TF-IDF transformation complete.")
    print(f"Shape of TF-IDF matrix: {X.shape}")

if __name__ == "__main__":
    main()
