import pandas as pd
from config import ALERT_THRESHOLD

def compute_trend(sentiment_df):
    sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['created_utc'])
    sentiment_df['sentiment_score'] = sentiment_df['sentiment'].apply(lambda x: 1 if x == 'positive' else -1)
    trend = sentiment_df.resample('1H', on='timestamp').mean(numeric_only=True)
    trend['rolling_avg'] = trend['sentiment_score'].rolling(window=3).mean()
    return trend

def check_spike(trend_df):
    if len(trend_df) < 3:
        return "Insufficient data"
    recent_avg = trend_df['rolling_avg'].iloc[-1]
    if recent_avg < ALERT_THRESHOLD:
        return "⚠️ ALERT: Negative Sentiment Spike Detected!"
    return "✅ Sentiment Stable"


