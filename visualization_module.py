"""
visualization_module.py
=========================================================
Vizualizacije za DiplomskiApp (OPTIMIZED)

Podržano:
1. Pie chart emocija
2. Bar chart emocija
3. Emotional timeline po vremenu i govorniku
4. Continuous score histogram
5. Speaker sentiment comparison

Radi za:
- GoEmotions CSV
- BERT CSV
- bilo koji CSV koji sadrži:
    predicted_emotion

Ako CSV već sadrži:
    continuous_score
    sentiment_group

onda koristi postojeće vrijednosti.

Ako ne sadrži:
    automatski računa fallback score iz labela.

=========================================================
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# =====================================================
# FALLBACK VALENCE MAP
# =====================================================

EMOTION_VALENCE = {
    # negative
    "anger": -0.80,
    "annoyance": -0.65,
    "disapproval": -0.55,
    "disgust": -0.90,
    "fear": -0.70,
    "nervousness": -0.55,
    "sadness": -0.75,
    "disappointment": -0.65,
    "grief": -0.95,
    "remorse": -0.60,
    "embarrassment": -0.45,

    # neutral / mixed
    "confusion": -0.15,
    "realization": 0.05,
    "surprise": 0.00,
    "neutral": 0.00,

    # positive
    "curiosity": 0.25,
    "approval": 0.45,
    "caring": 0.50,
    "admiration": 0.70,
    "gratitude": 0.85,
    "joy": 0.90,
    "amusement": 0.75,
    "love": 1.00,
    "optimism": 0.70,
    "pride": 0.65,
    "relief": 0.55,
    "excitement": 0.85,
}


# =====================================================
# LOAD CSV
# =====================================================

def load_result_csv(csv_path):
    df = pd.read_csv(csv_path)

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    return df


# =====================================================
# SENTIMENT GROUP
# =====================================================

def score_group(score):
    if score <= -0.35:
        return "negative"
    elif score >= 0.35:
        return "positive"
    return "neutral"


# =====================================================
# ENSURE CONTINUOUS SCORE
# =====================================================

def ensure_scores(df):
    """
    Ako CSV već ima continuous_score koristi njega.
    Inače radi fallback preko predicted_emotion.
    """

    df = df.copy()

    if "continuous_score" not in df.columns:
        df["continuous_score"] = (
            df["predicted_emotion"]
            .map(EMOTION_VALENCE)
            .fillna(0.0)
        )

    if "sentiment_group" not in df.columns:
        df["sentiment_group"] = df["continuous_score"].apply(score_group)

    return df


# =====================================================
# PIE CHART
# =====================================================

def create_pie_chart(df):
    counts = (
        df["predicted_emotion"]
        .value_counts()
        .reset_index()
    )

    counts.columns = ["emotion", "count"]

    fig = px.pie(
        counts,
        names="emotion",
        values="count",
        hole=0.38,
        title="Distribucija emocija"
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label"
    )

    return fig


# =====================================================
# BAR CHART
# =====================================================

def create_bar_chart(df):
    counts = (
        df["predicted_emotion"]
        .value_counts()
        .reset_index()
    )

    counts.columns = ["emotion", "count"]

    fig = px.bar(
        counts,
        x="emotion",
        y="count",
        text="count",
        color="count",
        title="Broj poruka po emociji"
    )

    fig.update_layout(
        xaxis_title="Emocija",
        yaxis_title="Broj poruka",
        showlegend=False
    )

    return fig


# =====================================================
# HISTOGRAM SCORE
# =====================================================

def create_score_histogram(df):
    df = ensure_scores(df)

    fig = px.histogram(
        df,
        x="continuous_score",
        nbins=30,
        title="Distribucija kontinuiranog skora"
    )

    fig.update_layout(
        xaxis_title="Continuous score",
        yaxis_title="Broj poruka"
    )

    return fig


# =====================================================
# SPEAKER COMPARISON
# =====================================================

def create_speaker_comparison(df):
    df = ensure_scores(df)

    if "speaker" not in df.columns:
        return go.Figure()

    stats = (
        df.groupby("speaker")["continuous_score"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        stats,
        x="speaker",
        y="continuous_score",
        text="continuous_score",
        title="Prosječni emocionalni skor po govorniku"
    )

    fig.update_traces(texttemplate="%{text:.2f}")

    return fig


# =====================================================
# MAIN TIMELINE
# =====================================================

def create_timeline_chart(df):
    """
    Koristi pravi continuous_score iz modela.
    """

    df = ensure_scores(df)

    if "speaker" not in df.columns:
        df["speaker"] = "Speaker"

    marker_colors = {
        "negative": "#d99a9a",
        "neutral": "#93a4b8",
        "positive": "#8fc79d"
    }

    speakers = list(df["speaker"].dropna().unique())

    palette = [
        "#3b82f6",
        "#f59e0b",
        "#10b981",
        "#8b5cf6",
        "#ef4444",
        "#14b8a6"
    ]

    line_colors = {
        sp: palette[i % len(palette)]
        for i, sp in enumerate(speakers)
    }

    fig = go.Figure()

    # background zones
    fig.add_hrect(y0=-1.0, y1=-0.60, fillcolor="rgba(255,215,215,0.35)", line_width=0)
    fig.add_hrect(y0=-0.60, y1=-0.20, fillcolor="rgba(255,235,220,0.28)", line_width=0)
    fig.add_hrect(y0=-0.20, y1=0.20, fillcolor="rgba(235,238,242,0.30)", line_width=0)
    fig.add_hrect(y0=0.20, y1=0.60, fillcolor="rgba(224,244,228,0.28)", line_width=0)
    fig.add_hrect(y0=0.60, y1=1.00, fillcolor="rgba(210,240,216,0.35)", line_width=0)

    for speaker in speakers:

        sdf = df[df["speaker"] == speaker].copy()

        hover_cols = ["predicted_emotion"]

        if "emotion_score" in sdf.columns:
            hover_cols.append("emotion_score")

        fig.add_trace(
            go.Scatter(
                x=sdf["datetime"],
                y=sdf["continuous_score"],
                mode="lines+markers",
                name=speaker,

                line=dict(
                    color=line_colors[speaker],
                    width=4,
                    shape="spline",
                    smoothing=0.7
                ),

                marker=dict(
                    size=10,
                    color=[
                        marker_colors[g]
                        for g in sdf["sentiment_group"]
                    ],
                    line=dict(color="white", width=1.3)
                ),

                text=sdf["text"],

                customdata=sdf[hover_cols],

                hovertemplate=(
                    "<b>Govornik:</b> " + speaker + "<br>"
                    "<b>Vrijeme:</b> %{x}<br>"
                    "<b>Continuous:</b> %{y:.3f}<br>"
                    "<b>Emocija:</b> %{customdata[0]}<br>"
                    "<b>Poruka:</b> %{text}<extra></extra>"
                )
            )
        )

    fig.update_layout(
        title="Emocionalni tok razgovora",
        xaxis_title="Vrijeme",
        yaxis_title="Continuous score",
        template="plotly_white",
        hovermode="closest",
        legend_title="Sudionik",
        height=650
    )

    fig.update_yaxes(
        range=[-1.05, 1.05],
        tickvals=[-1, -0.5, 0, 0.5, 1],
        ticktext=[
            "jako negativno",
            "negativno",
            "neutralno",
            "pozitivno",
            "jako pozitivno"
        ]
    )

    return fig