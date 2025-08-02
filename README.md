Sentiment Analysis on Social Media Posts 

This project analyzes the sentiment of tweets and posts using Natural Language Processing (NLP) and Machine Learning techniques.

Features
- Cleans and preprocesses raw text data.
- Extracts features using Bag-of-Words or TF-IDF.
- Trains ML models (Naive Bayes, Logistic Regression, etc.).
- Classifies sentiment into Positive, Negative, or Neutral.
- Visualizes sentiment distribution and word frequencies.

Tech Stack
- Python
- NLTK
- Scikit-learn
- Seaborn / Matplotlib
- Pandas, NumPy

Dataset
- Taken from kaggle :-
(https://www.kaggle.com/datasets/kazanova/sentiment140) # took dataset from kaggle


How It Works
1. Preprocesses the raw text (remove stopwords, stemming, etc. --- '@', '#', '!') 
2. Vectorizes text using TF-IDF or CountVectorizer
3. Trains an ML model on labeled data
4. Predicts sentiment of new/unseen posts
5. Plots analytics: pie charts, etc.

Output
- Sentiment class per post
- Model accuracy and confusion matrix
- Visualizations

Future Enhancements
- Deploy as REST API using FastAPI
- Use advanced models like BERT or LSTM
- Integrate with real-time Twitter API

Author
**Kunal Burnwal**  
[LinkedIn](https://www.linkedin.com/in/kunal-burnwal-addict-to-code) | [GitHub](https://github.com/Kunal-addict-to-code)


