import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import dash
from dash import Input, Output, dcc, html, dash_table
import dash_bootstrap_components as dbc
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import plotly.express as px
from textblob import TextBlob

DATA_PATH = Path(__file__).resolve().parent / "Dataset" / "twitter_dataset.csv"
SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]
BRAND_OPTIONS = ["All", "Apple", "Samsung", "Both", "Other"]
TOKEN_PATTERN = re.compile(r"[A-Za-z]{2,}")


def ensure_nltk_data():
    resources = [
        ("sentiment/vader_lexicon.zip", "vader_lexicon"),
        ("tokenizers/punkt.zip", "punkt"),
        ("corpora/stopwords", "stopwords"),
    ]
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name)


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"RT[\s]+", "", text)
    text = re.sub(r"https?://\S+", "", text)
    return text.strip()


def get_final_sentiment(combined_score):
    if combined_score >= 0.05:
        return "Positive"
    if combined_score <= -0.05:
        return "Negative"
    return "Neutral"


def detect_brand(text):
    if not isinstance(text, str):
        return "Other"
    lowered = text.lower()
    has_apple = any(token in lowered for token in ["apple", "iphone", "ipad", "ipod", "ios"])
    has_samsung = "samsung" in lowered or "galaxy" in lowered
    if has_apple and has_samsung:
        return "Both"
    if has_apple:
        return "Apple"
    if has_samsung:
        return "Samsung"
    return "Other"


def analyze_dataframe(df):
    tweet_column = None
    for col in ["tweet", "text", "Tweet", "Text", "full_text"]:
        if col in df.columns:
            tweet_column = col
            break

    if tweet_column is None:
        raise ValueError("Could not find a tweet text column. Expected one of: tweet, text, full_text")

    df = df.copy()
    df["raw_tweet"] = df[tweet_column].astype(str)
    df["cleaned_tweet"] = df["raw_tweet"].apply(clean_text)
    df["textblob_polarity"] = df["cleaned_tweet"].apply(lambda x: TextBlob(x).sentiment.polarity)
    df["textblob_subjectivity"] = df["cleaned_tweet"].apply(lambda x: TextBlob(x).sentiment.subjectivity)

    sia = SentimentIntensityAnalyzer()
    df["vader_compound"] = df["cleaned_tweet"].apply(lambda x: sia.polarity_scores(x)["compound"])

    df["combined_sentiment_score"] = (df["textblob_polarity"] + df["vader_compound"]) / 2
    df["final_sentiment"] = df["combined_sentiment_score"].apply(get_final_sentiment)
    df["brand"] = df["raw_tweet"].apply(detect_brand)
    df["tweet_length"] = df["cleaned_tweet"].apply(len)
    return df


def load_data():
    if not DATA_PATH.exists():
        return pd.DataFrame(), f"Dataset not found at {DATA_PATH}."

    df = pd.read_csv(DATA_PATH)
    if df.empty:
        return df, "Dataset loaded but it is empty."

    try:
        df = analyze_dataframe(df)
    except ValueError as exc:
        return pd.DataFrame(), str(exc)

    return df, None


def filter_dataframe(df, keyword, sentiments, brand, polarity_range, subjectivity_range):
    if df.empty:
        return df
    filtered = df.copy()

    if keyword:
        pattern = re.escape(keyword.strip())
        filtered = filtered[filtered["raw_tweet"].str.contains(pattern, case=False, na=False)]

    if sentiments:
        filtered = filtered[filtered["final_sentiment"].isin(sentiments)]

    if brand and brand != "All":
        filtered = filtered[filtered["brand"] == brand]

    if polarity_range:
        filtered = filtered[
            (filtered["textblob_polarity"] >= polarity_range[0])
            & (filtered["textblob_polarity"] <= polarity_range[1])
        ]

    if subjectivity_range:
        filtered = filtered[
            (filtered["textblob_subjectivity"] >= subjectivity_range[0])
            & (filtered["textblob_subjectivity"] <= subjectivity_range[1])
        ]

    return filtered


def build_metrics(df):
    if df.empty:
        return 0, 0.0, 0.0, 0.0, 0.0
    total = len(df)
    avg_polarity = df["textblob_polarity"].mean()
    avg_subjectivity = df["textblob_subjectivity"].mean()
    positive_pct = (df["final_sentiment"] == "Positive").mean() * 100
    negative_pct = (df["final_sentiment"] == "Negative").mean() * 100
    return total, avg_polarity, avg_subjectivity, positive_pct, negative_pct


def build_sentiment_chart(df):
    if df.empty or "final_sentiment" not in df.columns:
        return px.bar(title="No sentiment data available")

    sentiment_counts = (
        df["final_sentiment"]
        .value_counts()
        .reindex(SENTIMENT_ORDER)
        .fillna(0)
        .astype(int)
    )
    chart_df = sentiment_counts.reset_index()
    chart_df.columns = ["sentiment", "count"]

    fig = px.bar(
        chart_df,
        x="sentiment",
        y="count",
        color="sentiment",
        color_discrete_map={"Positive": "#2E8B57", "Neutral": "#6C757D", "Negative": "#C0392B"},
        title="Sentiment Distribution",
        text="count",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), xaxis_title=None, yaxis_title="Tweets")
    return fig


def build_polarity_histogram(df):
    if df.empty:
        return px.histogram(title="No polarity data available")
    fig = px.histogram(
        df,
        x="combined_sentiment_score",
        nbins=30,
        title="Combined Sentiment Score Distribution",
        color_discrete_sequence=["#1F77B4"],
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), xaxis_title="Score", yaxis_title="Tweets")
    return fig


def build_scatter(df):
    if df.empty:
        return px.scatter(title="No sentiment data available")
    fig = px.scatter(
        df,
        x="textblob_polarity",
        y="textblob_subjectivity",
        color="final_sentiment",
        color_discrete_map={"Positive": "#2E8B57", "Neutral": "#6C757D", "Negative": "#C0392B"},
        hover_data=["cleaned_tweet"],
        title="Polarity vs Subjectivity",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), xaxis_title="Polarity", yaxis_title="Subjectivity")
    return fig


def build_top_terms(df, top_n, stop_words):
    if df.empty:
        return px.bar(title="No terms available")
    counter = Counter()
    for text in df["cleaned_tweet"].dropna().astype(str):
        tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
        counter.update([token for token in tokens if token not in stop_words])
    most_common = counter.most_common(top_n)
    if not most_common:
        return px.bar(title="No terms available")
    terms_df = pd.DataFrame(most_common, columns=["term", "count"])
    fig = px.bar(
        terms_df,
        x="count",
        y="term",
        orientation="h",
        title="Top Terms",
        color_discrete_sequence=["#FF8C42"],
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), xaxis_title="Count", yaxis_title=None)
    return fig


def build_brand_comparison(df):
    if df.empty or "brand" not in df.columns:
        return px.bar(title="No brand comparison data available")
    compare_df = df[df["brand"].isin(["Apple", "Samsung"])]
    if compare_df.empty:
        return px.bar(title="No Apple/Samsung tweets found")
    sentiment_counts = (
        compare_df.groupby(["brand", "final_sentiment"]).size().reset_index(name="count")
    )
    fig = px.bar(
        sentiment_counts,
        x="brand",
        y="count",
        color="final_sentiment",
        barmode="group",
        color_discrete_map={"Positive": "#2E8B57", "Neutral": "#6C757D", "Negative": "#C0392B"},
        title="Apple vs Samsung Sentiment Comparison",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), xaxis_title=None, yaxis_title="Tweets")
    return fig


def build_brand_summary(df):
    if df.empty:
        return pd.DataFrame(columns=["brand", "tweets", "avg_score", "positive_pct", "negative_pct"])
    compare_df = df[df["brand"].isin(["Apple", "Samsung"])]
    if compare_df.empty:
        return pd.DataFrame(columns=["brand", "tweets", "avg_score", "positive_pct", "negative_pct"])
    summary = (
        compare_df
        .groupby("brand")
        .apply(
            lambda group: pd.Series({
                "tweets": len(group),
                "avg_score": group["combined_sentiment_score"].mean(),
                "positive_pct": (group["final_sentiment"] == "Positive").mean() * 100,
                "negative_pct": (group["final_sentiment"] == "Negative").mean() * 100,
            })
        )
        .reset_index()
    )
    return summary


def make_card(title, value, suffix=""):
    display_value = f"{value:.2f}{suffix}" if isinstance(value, float) else f"{value}{suffix}"
    return dbc.Card(
        dbc.CardBody([
            html.H6(title, className="card-title text-muted"),
            html.H2(display_value, className="card-text"),
        ]),
        className="shadow-sm kpi-card",
    )


def make_table(df, max_rows=10):
    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": col.replace("_", " ").title(), "id": col} for col in df.columns],
        page_size=max_rows,
        style_cell={"textAlign": "left", "padding": "8px", "whiteSpace": "normal", "height": "auto"},
        style_header={"backgroundColor": "#F0F3F7", "fontWeight": "bold"},
        style_table={"overflowX": "auto"},
    )


ensure_nltk_data()
stop_words = set(stopwords.words("english"))
stop_words.update(["amp", "https", "http", "rt", "co"])

df, load_error = load_data()
last_updated = datetime.now().strftime("%b %d, %Y %I:%M %p")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Sentiment Dashboard</title>
        {%favicon%}
        {%css%}
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --ink: #111827;
                --muted: #6B7280;
                --card: #FFFFFF;
                --bg: #F4F6FB;
                --accent: #FF8C42;
                --accent-dark: #1F2A44;
            }
            body {
                font-family: "Space Grotesk", sans-serif;
                background: linear-gradient(180deg, #EEF1F6 0%, #F9FAFB 100%);
            }
            .app-shell {
                background: transparent;
                padding: 24px 8px 40px;
            }
            .sidebar {
                background: var(--card);
                border-radius: 18px;
                padding: 20px;
                box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
            }
            .main-panel {
                padding-left: 12px;
            }
            .kpi-card {
                border: none;
                border-radius: 16px;
            }
            .card-title {
                letter-spacing: 0.02em;
            }
            .graph-card {
                background: var(--card);
                border-radius: 18px;
                padding: 16px;
                box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

if load_error:
    content = dbc.Alert(load_error, color="danger")
else:
    content = html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div("Filters", className="text-uppercase fw-bold text-muted mb-3"),
                    dbc.Label("Keyword"),
                    dbc.Input(id="keyword-input", placeholder="Search text", type="text"),
                    dbc.Label("Sentiment", className="mt-3"),
                    dbc.Checklist(
                        id="sentiment-filter",
                        options=[{"label": label, "value": label} for label in SENTIMENT_ORDER],
                        value=SENTIMENT_ORDER,
                        inline=True,
                    ),
                    dbc.Label("Brand", className="mt-3"),
                    dbc.Select(
                        id="brand-filter",
                        options=[{"label": option, "value": option} for option in BRAND_OPTIONS],
                        value="All",
                    ),
                    dbc.Label("Polarity Range", className="mt-3"),
                    dcc.RangeSlider(
                        id="polarity-range",
                        min=-1,
                        max=1,
                        step=0.05,
                        value=[-1, 1],
                        marks={-1: "-1", -0.5: "-0.5", 0: "0", 0.5: "0.5", 1: "1"},
                    ),
                    dbc.Label("Subjectivity Range", className="mt-3"),
                    dcc.RangeSlider(
                        id="subjectivity-range",
                        min=0,
                        max=1,
                        step=0.05,
                        value=[0, 1],
                        marks={0: "0", 0.5: "0.5", 1: "1"},
                    ),
                    dbc.Label("Top Terms", className="mt-3"),
                    dcc.Slider(
                        id="top-terms-count",
                        min=5,
                        max=30,
                        step=1,
                        value=12,
                        marks={5: "5", 15: "15", 30: "30"},
                    ),
                    html.Div(
                        f"Last updated: {last_updated}",
                        className="text-muted mt-3",
                    ),
                ], className="sidebar"),
            ], md=3),
            dbc.Col([
                html.Div([
                    html.H1("Twitter Sentiment Analysis Dashboard", className="fw-bold"),
                    html.P("Interactive sentiment insights inspired by the notebook results.", className="text-muted"),
                ]),
                dbc.Row([
                    dbc.Col(html.Div(id="kpi-total"), md=4),
                    dbc.Col(html.Div(id="kpi-polarity"), md=4),
                    dbc.Col(html.Div(id="kpi-subjectivity"), md=4),
                    dbc.Col(html.Div(id="kpi-positive"), md=4),
                    dbc.Col(html.Div(id="kpi-negative"), md=4),
                ], className="g-3"),
                dbc.Tabs([
                    dbc.Tab([
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="sentiment-chart"), md=6),
                            dbc.Col(dcc.Graph(id="polarity-hist"), md=6),
                        ], className="mt-4"),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="scatter-chart"), md=6),
                            dbc.Col(dcc.Graph(id="top-terms-chart"), md=6),
                        ], className="mt-4"),
                    ], label="Overview"),
                    dbc.Tab([
                        dbc.Row([
                            dbc.Col([
                                html.H5("Apple Snapshot"),
                                html.Div(id="kpi-apple-total"),
                                html.Div(id="kpi-apple-score"),
                                html.Div(id="kpi-apple-positive"),
                                html.Div(id="kpi-apple-negative"),
                            ], md=6),
                            dbc.Col([
                                html.H5("Samsung Snapshot"),
                                html.Div(id="kpi-samsung-total"),
                                html.Div(id="kpi-samsung-score"),
                                html.Div(id="kpi-samsung-positive"),
                                html.Div(id="kpi-samsung-negative"),
                            ], md=6),
                        ], className="mt-4"),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="brand-compare-chart"), md=12),
                        ], className="mt-4"),
                        dbc.Row([
                            dbc.Col([
                                html.H5("Apple vs Samsung Summary"),
                                html.Div(id="brand-summary-table"),
                            ], md=12),
                        ], className="mt-4"),
                    ], label="Apple vs Samsung"),
                    dbc.Tab([
                        dbc.Row([
                            dbc.Col([
                                html.H5("Top Positive Tweets"),
                                html.Div(id="table-positive"),
                            ], md=6),
                            dbc.Col([
                                html.H5("Top Negative Tweets"),
                                html.Div(id="table-negative"),
                            ], md=6),
                        ], className="mt-4"),
                        dbc.Row([
                            dbc.Col([
                                html.H5("Filtered Tweets"),
                                html.Div(id="table-all"),
                            ], md=12),
                        ], className="mt-4"),
                    ], label="Tweets"),
                ], className="mt-4"),
            ], md=9, className="main-panel"),
        ], className="g-3"),
    ], className="app-shell")

app.layout = dbc.Container(content, fluid=True)


@app.callback(
    [
        Output("kpi-total", "children"),
        Output("kpi-polarity", "children"),
        Output("kpi-subjectivity", "children"),
        Output("kpi-positive", "children"),
        Output("kpi-negative", "children"),
        Output("kpi-apple-total", "children"),
        Output("kpi-apple-score", "children"),
        Output("kpi-apple-positive", "children"),
        Output("kpi-apple-negative", "children"),
        Output("kpi-samsung-total", "children"),
        Output("kpi-samsung-score", "children"),
        Output("kpi-samsung-positive", "children"),
        Output("kpi-samsung-negative", "children"),
        Output("sentiment-chart", "figure"),
        Output("polarity-hist", "figure"),
        Output("scatter-chart", "figure"),
        Output("top-terms-chart", "figure"),
        Output("brand-compare-chart", "figure"),
        Output("brand-summary-table", "children"),
        Output("table-positive", "children"),
        Output("table-negative", "children"),
        Output("table-all", "children"),
    ],
    [
        Input("keyword-input", "value"),
        Input("sentiment-filter", "value"),
        Input("brand-filter", "value"),
        Input("polarity-range", "value"),
        Input("subjectivity-range", "value"),
        Input("top-terms-count", "value"),
    ],
)
def update_dashboard(keyword, sentiments, brand, polarity_range, subjectivity_range, top_terms_count):
    filtered = filter_dataframe(df, keyword, sentiments, brand, polarity_range, subjectivity_range)
    total, avg_polarity, avg_subjectivity, positive_pct, negative_pct = build_metrics(filtered)

    apple_df = filtered[filtered["brand"] == "Apple"] if not filtered.empty else filtered
    samsung_df = filtered[filtered["brand"] == "Samsung"] if not filtered.empty else filtered
    apple_total, apple_polarity, _, apple_positive, apple_negative = build_metrics(apple_df)
    samsung_total, samsung_polarity, _, samsung_positive, samsung_negative = build_metrics(samsung_df)

    sentiment_fig = build_sentiment_chart(filtered)
    polarity_fig = build_polarity_histogram(filtered)
    scatter_fig = build_scatter(filtered)
    top_terms_fig = build_top_terms(filtered, int(top_terms_count or 10), stop_words)
    brand_compare_fig = build_brand_comparison(filtered)
    brand_summary = build_brand_summary(filtered)
    brand_summary_table = make_table(brand_summary, max_rows=6)

    if filtered.empty:
        empty_df = pd.DataFrame(columns=["cleaned_tweet", "combined_sentiment_score", "final_sentiment"])
        pos_table = make_table(empty_df)
        neg_table = make_table(empty_df)
        all_table = make_table(empty_df, max_rows=12)
    else:
        pos_df = filtered.sort_values("combined_sentiment_score", ascending=False).head(10)
        neg_df = filtered.sort_values("combined_sentiment_score", ascending=True).head(10)
        pos_table = make_table(pos_df[["cleaned_tweet", "combined_sentiment_score"]])
        neg_table = make_table(neg_df[["cleaned_tweet", "combined_sentiment_score"]])
        all_table = make_table(
            filtered[["cleaned_tweet", "final_sentiment", "combined_sentiment_score", "brand"]].head(200),
            max_rows=12,
        )

    return (
        make_card("Total Tweets", total),
        make_card("Average Polarity", avg_polarity),
        make_card("Average Subjectivity", avg_subjectivity),
        make_card("Positive Share", positive_pct, "%"),
        make_card("Negative Share", negative_pct, "%"),
        make_card("Apple Tweets", apple_total),
        make_card("Apple Avg Score", apple_polarity),
        make_card("Apple Positive", apple_positive, "%"),
        make_card("Apple Negative", apple_negative, "%"),
        make_card("Samsung Tweets", samsung_total),
        make_card("Samsung Avg Score", samsung_polarity),
        make_card("Samsung Positive", samsung_positive, "%"),
        make_card("Samsung Negative", samsung_negative, "%"),
        sentiment_fig,
        polarity_fig,
        scatter_fig,
        top_terms_fig,
        brand_compare_fig,
        brand_summary_table,
        pos_table,
        neg_table,
        all_table,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
