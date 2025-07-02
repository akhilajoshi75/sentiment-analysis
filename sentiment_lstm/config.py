import os

# Base directory of the sentiment_lstm service
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Subreddit and keyword settings
SUBREDDIT = "politics"
KEYWORD = "India"
FETCH_LIMIT = 100

# Model and tokenizer paths (absolute paths)
MODEL_PATH = os.path.join(BASE_DIR, "lstm_sentiment_model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "lstm_tokenizer.pkl")

# Alert threshold for negative sentiment (not used in all components)
ALERT_THRESHOLD = -0.5  # rolling avg sentiment score
