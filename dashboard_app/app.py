import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
from reddit_fetcher import fetch_reddit_posts
import datetime
import os

# Initialize app
app = dash.Dash(__name__)
app.title = "Sentiment Dashboard"

app.layout = html.Div([
    html.H1("Reddit Sentiment Analysis", style={"textAlign": "center"}),

    html.Div([
        dcc.Input(id='keyword-input', type='text', placeholder='Enter keyword...', style={'width': '60%'}),
        html.Button('Analyze', id='analyze-button', n_clicks=0),
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),

    html.Div(id='alert-box', style={'textAlign': 'center', 'marginBottom': '20px', 'fontWeight': 'bold'}),
    dcc.Graph(id='sentiment-graph'),

    html.Div(id='post-count', style={"textAlign": "center", "marginTop": "20px"})
])

@app.callback(
    Output('sentiment-graph', 'figure'),
    Output('alert-box', 'children'),
    Output('post-count', 'children'),
    Input('analyze-button', 'n_clicks'),
    State('keyword-input', 'value')
)
def update_sentiment_graph(n_clicks, keyword):
    if not keyword:
        return dash.no_update, "Enter a keyword to analyze sentiment.", ""

    posts_df = fetch_reddit_posts(keyword, limit=50)
    if posts_df.empty:
        return dash.no_update, "No relevant posts found.", ""

    sentiment_counts = posts_df['sentiment'].value_counts().reindex(['positive', 'negative'], fill_value=0)

    fig = px.bar(
        sentiment_counts,
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        labels={'x': 'Sentiment', 'y': 'Number of Posts'},
        title=f"Sentiment Analysis for '{keyword}' on Reddit",
        color=sentiment_counts.index,
        color_discrete_map={'positive': 'green', 'negative': 'red'}
    )

    # Check for alert
    negative_ratio = sentiment_counts['negative'] / sentiment_counts.sum()
    alert_msg = (
        "⚠️ High negative sentiment detected!" if negative_ratio > 0.6
        else "✅ Sentiment looks normal."
    )

    return fig, alert_msg, f"Analyzed {len(posts_df)} recent posts."

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host='0.0.0.0', port=port)


