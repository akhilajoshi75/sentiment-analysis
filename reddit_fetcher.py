# reddit_fetcher.py

import praw
import datetime
import pandas as pd
import requests

# Initialize Reddit API
reddit = praw.Reddit(
    client_id="Q5FwOJvNDEPdY_aZGEgMgA",
    client_secret="Fni4MKF6kyUFRz94emXoV8udW_Jg-w",
    user_agent="RealTimeSentimentSentimentDashboard"
)

def get_sentiment_from_api(text):
    try:
        response = requests.post("https://sentiment-analysis-production-9ce2.up.railway.app/", json={"text": text})
        if response.status_code == 200:
            return response.json().get("sentiment", "neutral")
        else:
            return "neutral"
    except Exception as e:
        print("API error:", e)
        return "neutral"

def fetch_reddit_posts(keyword, subreddit_name="all", limit=50):
    posts = []
    subreddit = reddit.subreddit(subreddit_name)

    for post in subreddit.search(keyword, sort='new', limit=limit):
        title = post.title
        body = post.selftext
        content = f"{title} {body}".strip()

        sentiment = get_sentiment_from_api(content)

        posts.append({
            'title': title,
            'text': body,
            'created_utc': datetime.datetime.fromtimestamp(post.created_utc),
            'sentiment': sentiment
        })

    return pd.DataFrame(posts)
