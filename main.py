# main.py

from reddit_fetcher import fetch_reddit_data
from trend_analysis.analyzer import compute_trend, check_spike
import requests

def get_sentiment_from_api(text):
    try:
        response = requests.post("http://localhost:5000/predict", json={"text": text})
        if response.status_code == 200:
            return response.json().get("sentiment", "neutral")
        else:
            return "neutral"
    except Exception as e:
        print("API error:", e)
        return "neutral"

def process_reddit_posts():
    df = fetch_reddit_data()
    df['text_combined'] = df['title'] + " " + df['text']
    df['sentiment'] = df['text_combined'].apply(get_sentiment_from_api)
    return df

def run_pipeline():
    df = process_reddit_posts()
    trend_df = compute_trend(df)
    alert = check_spike(trend_df)
    return trend_df, alert
