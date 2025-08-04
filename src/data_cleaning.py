import pandas as pd
import re
import nltk
import os
from nltk.corpus import stopwords

# Download stopwords if running for the first time
nltk.download('stopwords')

# Load stop words
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)       # Remove URLs
    text = re.sub(r"@\w+", "", text)                 # Remove @mentions
    text = re.sub(r"#\w+", "", text)                 # Remove hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)          # Remove punctuation & numbers
    text = text.lower()                              # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

def main():
    # Define input and output paths
    input_file = r"e:\SystemAnalysisProject\data\raw_sentiment_data.csv"
    output_file = r"e:\SystemAnalysisProject\data\cleaned_sentiment_data.csv"

    print("Looking for file at:", input_file)  # Debugging line

    # ✅ Check if file exists before loading
    if not os.path.exists(input_file):
        print("❌ ERROR: File not found at", input_file)
        return

    print("✅ File found, loading raw data...")
    df = pd.read_csv(input_file, encoding='latin-1', header=None)
    df = df[[0, 5]]  # Keep sentiment and text columns
    df.columns = ['sentiment', 'text']

    # Map numerical sentiment to labels
    df['sentiment'] = df['sentiment'].replace({0: 'Negative', 2: 'Neutral', 4: 'Positive'})

    # Apply text cleaning
    print("Cleaning text...")
    df['cleaned_text'] = df['text'].apply(clean_text)

    # Save the cleaned data
    df.to_csv(output_file, index=False)
    print(f"✅ Cleaned data saved to: {output_file}")

    # Show sample output
    print(df[['text', 'cleaned_text']].head())

if __name__ == "__main__":
    main()
