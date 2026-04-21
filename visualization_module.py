"""
visualization_module.py
=========================================================
Vizualizacije za DiplomskiApp

Podržano:
1. Pie chart emocija
2. Bar chart emocija
3. Kontinuirani emotional timeline po vremenu i govorniku

Radi za:
- GoEmotions CSV
- BERT CSV
- bilo koji CSV koji sadrži predicted_emotion

Koristi:
- Plotly
- Streamlit

=========================================================
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# =====================================================
# VALENCIJA EMOCIJA
# =====================================================

EMOTION_VALENCE = {
    # negativno
    "anger": -1.00,
    "annoyance": -0.85,
    "disapproval": -0.75,
    "disgust": -1.00,
    "fear": -0.90,
    "nervousness": -0.65,
    "sadness": -0.90,
    "disappointment": -0.80,
    "grief": -1.00,
    "remorse": -0.70,
    "embarrassment": -0.60,

    # sredina
    "confusion": -0.35,
    "realization": -0.10,
    "surprise": 0.00,
    "neutral": 0.00,

    # pozitivno
    "curiosity": 0.30,
    "approval": 0.50,
    "caring": 0.55,
    "admiration": 0.75,
    "gratitude": 0.90,
    "joy": 0.95,
    "amusement": 0.80,
    "love": 1.00,
    "optimism": 0.75,
    "pride": 0.70,
    "relief": 0.65,
    "excitement": 0.90,

    # ekman/bart fallback
    "joy": 0.95,
    "neutral": 0.00,
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
# PIE CHART
# =====================================================

def create_pie_chart(df):
    counts = df["predicted_emotion"].value_counts().reset_index()
    counts.columns = ["emotion", "count"]

    fig = px.pie(
        counts,
        names="emotion",
        values="count",
        title="Distribucija emocija",
        hole=0.35
    )

    fig.update_traces(textposition="inside", textinfo="percent+label")

    return fig


# =====================================================
# BAR CHART
# =====================================================

def create_bar_chart(df):
    counts = df["predicted_emotion"].value_counts().reset_index()
    counts.columns = ["emotion", "count"]

    fig = px.bar(
        counts,
        x="emotion",
        y="count",
        title="Broj poruka po emociji",
        text="count"
    )

    fig.update_layout(
        xaxis_title="Emocija",
        yaxis_title="Broj poruka"
    )

    return fig


# =====================================================
# SCORE CALCULATION
# =====================================================

def add_continuous_score(df):
    """
    Dodaj continuous score po labeli.
    """
    scores = []

    for label in df["predicted_emotion"]:
        score = EMOTION_VALENCE.get(label, 0.0)
        scores.append(score)

    df["continuous_score"] = scores

    def group(score):
        if score <= -0.35:
            return "negativno"
        elif score >= 0.35:
            return "pozitivno"
        return "neutralno"

    df["score_group"] = df["continuous_score"].apply(group)

    return df


# =====================================================
# TIMELINE GRAPH
# =====================================================

def create_timeline_chart(df):
    """
    Emotional flow po vremenu i sudionicima.
    """

    df = add_continuous_score(df)

    marker_colors = {
        "negativno": "#d99a9a",
        "neutralno": "#93a4b8",
        "pozitivno": "#8fc79d"
    }

    speakers = list(df["speaker"].unique())

    palette = [
        "#3b82f6",
        "#f59e0b",
        "#10b981",
        "#8b5cf6",
        "#ef4444",
        "#14b8a6"
    ]

    line_colors = {}

    for i, speaker in enumerate(speakers):
        line_colors[speaker] = palette[i % len(palette)]

    fig = go.Figure()

    # zones
    fig.add_hrect(y0=-1.05, y1=-0.60, fillcolor="rgba(255,215,215,0.40)", line_width=0)
    fig.add_hrect(y0=-0.60, y1=-0.20, fillcolor="rgba(255,235,220,0.30)", line_width=0)
    fig.add_hrect(y0=-0.20, y1=0.20, fillcolor="rgba(235,238,242,0.35)", line_width=0)
    fig.add_hrect(y0=0.20, y1=0.60, fillcolor="rgba(224,244,228,0.30)", line_width=0)
    fig.add_hrect(y0=0.60, y1=1.05, fillcolor="rgba(210,240,216,0.40)", line_width=0)

    for speaker in speakers:

        sdf = df[df["speaker"] == speaker].copy()

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
                    color=[marker_colors[g] for g in sdf["score_group"]],
                    line=dict(color="white", width=1.4)
                ),

                text=sdf["text"],

                customdata=sdf[["predicted_emotion"]],

                hovertemplate=(
                    "<b>Govornik:</b> " + speaker + "<br>"
                    "<b>Vrijeme:</b> %{x}<br>"
                    "<b>Skor:</b> %{y:.2f}<br>"
                    "<b>Emocija:</b> %{customdata[0]}<br>"
                    "<b>Poruka:</b> %{text}<extra></extra>"
                )
            )
        )

    fig.update_layout(
        title="Emocionalni tok razgovora",
        xaxis_title="Vrijeme",
        yaxis_title="Emocionalni skor",
        template="plotly_white",
        hovermode="closest",
        legend_title="Sudionik"
    )

    fig.update_yaxes(
        tickvals=[-1, -0.5, 0, 0.5, 1],
        ticktext=[
            "jako negativno",
            "blago negativno",
            "neutralno",
            "blago pozitivno",
            "jako pozitivno"
        ],
        range=[-1.05, 1.05]
    )

    return fig