import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Paths
    data_path = "E:/SystemAnalysisProject/data/cleaned_sentiment_data.csv"
    vectorizer_path = "E:/SystemAnalysisProject/models/tfidf_vectorizer.pkl"
    model_path = "E:/SystemAnalysisProject/models/sentiment_model.pkl"

    # Load cleaned data
    print("Loading cleaned data...")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['cleaned_text'])
    df = df[df['cleaned_text'].str.strip() != '']

    # Reduce size to avoid memory errors
    df = df.sample(20000, random_state=42)


    # Load saved vectorizer
    print("Loading TF-IDF vectorizer...")
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    # Convert text to vectors
    X = vectorizer.transform(df['cleaned_text']).toarray()
    y = df['sentiment']

    # Train/test split
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Logistic Regression model
    print("Training model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    print("✅ Model trained and saved successfully.")

if __name__ == "__main__":
    main()
