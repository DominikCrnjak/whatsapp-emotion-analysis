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
6. 3D VAD emotional space

Radi za:
- GoEmotions CSV
- BERT CSV
- bilo koji CSV koji sadrži:
    predicted_emotion

Ako CSV već sadrži:
    continuous_score
    sentiment_group
    vad_valence
    vad_arousal
    vad_dominance

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
# ENSURE SCORES
# =====================================================

def ensure_scores(df):
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
    counts = df["predicted_emotion"].value_counts().reset_index()
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
    counts = df["predicted_emotion"].value_counts().reset_index()
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
# TIMELINE
# =====================================================

def create_timeline_chart(df):
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

    fig.add_hrect(y0=-1.0, y1=-0.60, fillcolor="rgba(255,215,215,0.35)", line_width=0)
    fig.add_hrect(y0=-0.60, y1=-0.20, fillcolor="rgba(255,235,220,0.28)", line_width=0)
    fig.add_hrect(y0=-0.20, y1=0.20, fillcolor="rgba(235,238,242,0.30)", line_width=0)
    fig.add_hrect(y0=0.20, y1=0.60, fillcolor="rgba(224,244,228,0.28)", line_width=0)
    fig.add_hrect(y0=0.60, y1=1.00, fillcolor="rgba(210,240,216,0.35)", line_width=0)

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
                    color=[marker_colors[g] for g in sdf["sentiment_group"]],
                    line=dict(color="white", width=1.3)
                ),

                text=sdf["text"],

                customdata=sdf[["predicted_emotion"]],

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


# =====================================================
# 3D VAD GRAPH
# =====================================================

def create_vad_3d_chart(df):
    """
    3D Emotional Space:
    X = Valence
    Y = Arousal
    Z = Dominance
    """

    required = [
        "vad_valence",
        "vad_arousal",
        "vad_dominance"
    ]

    for col in required:
        if col not in df.columns:
            return go.Figure()

    df = ensure_scores(df)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=df["vad_valence"],
            y=df["vad_arousal"],
            z=df["vad_dominance"],

            mode="markers+lines",

            marker=dict(
                size=6,
                color=df["continuous_score"],
                colorscale="RdYlGn",
                cmin=-1,
                cmax=1,
                colorbar=dict(title="Emotion")
            ),

            line=dict(
                width=2,
                color="rgba(120,120,120,0.35)"
            ),

            text=[
                f"{row.text}<br>"
                f"<b>Emotion:</b> {row.predicted_emotion}<br>"
                f"<b>Score:</b> {row.continuous_score:.2f}<br>"
                f"<b>V:</b> {row.vad_valence:.2f}<br>"
                f"<b>A:</b> {row.vad_arousal:.2f}<br>"
                f"<b>D:</b> {row.vad_dominance:.2f}"
                for _, row in df.iterrows()
            ],

            hovertemplate="%{text}<extra></extra>"
        )
    )

    fig.update_layout(
        title="3D Emotional Space (VAD)",
        height=800,

        scene=dict(
            xaxis_title="Valence",
            yaxis_title="Arousal",
            zaxis_title="Dominance",

            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            zaxis=dict(range=[0, 1]),
        )
    )

    return fig

# =====================================================
# SANKEY EMOTION FLOW
# Dodaj u visualization_module.py
# =====================================================

from collections import Counter
import plotly.graph_objects as go


def create_emotion_sankey(df):
    """
    Sankey dijagram prijelaza emocija između poruka.

    Koristi:
        predicted_emotion

    Opcionalno:
        continuous_score
        speaker
        datetime

    Ideja:
    prethodna emocija -> sljedeća emocija

    Vizualno:
    pozitivne emocije gore
    negativne dolje
    """

    if "predicted_emotion" not in df.columns:
        return go.Figure()

    data = df.copy()

    # =================================================
    # CLEAN
    # =================================================

    data = data.dropna(subset=["predicted_emotion"])

    if len(data) < 2:
        return go.Figure()

    emotion_sequence = data["predicted_emotion"].astype(str).tolist()

    # =================================================
    # SORT PO VALENCIJI
    # =================================================

    emotions_present = sorted(
        list(set(emotion_sequence)),
        key=lambda x: EMOTION_VALENCE.get(x, 0),
        reverse=True
    )

    # =================================================
    # COLOR FUNCTION
    # =================================================

    def emotion_color(emotion):

        val = EMOTION_VALENCE.get(emotion, 0)

        # negative = red
        if val < 0:
            strength = min(abs(val), 1)
            r = 220
            g = int(220 * (1 - strength))
            b = int(220 * (1 - strength))
            return f"rgba({r},{g},{b},0.85)"

        # positive = green
        elif val > 0:
            strength = min(val, 1)
            r = int(220 * (1 - strength))
            g = 180 + int(75 * strength)
            b = int(220 * (1 - strength))
            return f"rgba({r},{g},{b},0.85)"

        # neutral
        else:
            return "rgba(240,200,50,0.85)"

    # =================================================
    # NODES
    # =================================================

    n = len(emotions_present)

    source_map = {
        emotion: i
        for i, emotion in enumerate(emotions_present)
    }

    target_map = {
        emotion: i + n
        for i, emotion in enumerate(emotions_present)
    }

    labels = emotions_present + emotions_present
    node_colors = (
        [emotion_color(e) for e in emotions_present] +
        [emotion_color(e) for e in emotions_present]
    )

    # =================================================
    # TRANSITIONS
    # =================================================

    transition_counts = Counter()

    for i in range(len(emotion_sequence) - 1):

        src = emotion_sequence[i]
        tgt = emotion_sequence[i + 1]

        transition_counts[(src, tgt)] += 1

    sources = []
    targets = []
    values = []
    link_colors = []

    for (src, tgt), count in transition_counts.items():

        sources.append(source_map[src])
        targets.append(target_map[tgt])
        values.append(count)

        base = emotion_color(src)
        link = (
            base
            .replace("0.85", "0.28")
            .replace("0.9", "0.28")
        )

        link_colors.append(link)

    # =================================================
    # FIGURE
    # =================================================

    fig = go.Figure(
        go.Sankey(

            arrangement="snap",

            node=dict(
                pad=25,
                thickness=26,
                line=dict(
                    color="rgba(0,0,0,0.20)",
                    width=0.5
                ),
                label=labels,
                color=node_colors
            ),

            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors
            )
        )
    )

    # =================================================
    # LAYOUT
    # =================================================

    fig.update_layout(

    title={
        "text": "Tok emocija između uzastopnih poruka",
        "x": 0.5,
        "xanchor": "center",
        "font": dict(size=22, color="white")
    },

    font=dict(
        size=13,
        color="white"
    ),

    paper_bgcolor="#0f172a",   # vanjska pozadina
    plot_bgcolor="#0f172a",    # unutarnja pozadina

    margin=dict(l=20, r=20, t=70, b=20),

    annotations=[
        dict(
            x=0.01,
            y=1.05,
            text="Prethodna emocija",
            showarrow=False,
            font=dict(size=14, color="#cbd5e1")
        ),
        dict(
            x=0.99,
            y=1.05,
            text="Sljedeća emocija",
            showarrow=False,
            font=dict(size=14, color="#cbd5e1")
        )
    ]
)

    return fig

# =====================================================
# INFLUENCE TIMELINE
# =====================================================

def create_influence_timeline(df):
    """
    Emotional influence između sudionika.

    Mjeri kako poruka jednog govornika utječe na sljedeći odgovor drugoga.

    Delta =
        score(odgovor druge osobe)
        -
        prethodni poznati score te osobe

    Pozitivno = popravio raspoloženje
    Negativno = pogoršao raspoloženje
    """

    df = ensure_scores(df).copy()

    if "speaker" not in df.columns:
        return go.Figure()

    if "datetime" in df.columns:
        df = df.sort_values("datetime").reset_index(drop=True)

    speakers = list(df["speaker"].dropna().unique())

    if len(speakers) < 2:
        return go.Figure()

    # samo prva dva sudionika
    s1 = speakers[0]
    s2 = speakers[1]

    directions = [
        (s1, s2),
        (s2, s1)
    ]

    fig = go.Figure()

    traces_per_direction = {}

    trace_index = 0

    for sender, receiver in directions:

        rows = []
        prev_receiver_score = None

        for i in range(len(df) - 1):

            row = df.iloc[i]
            nxt = df.iloc[i + 1]

            # sender -> receiver prijelaz
            if row["speaker"] == sender and nxt["speaker"] == receiver:

                current_score = nxt["continuous_score"]

                if prev_receiver_score is None:
                    delta = 0.0
                else:
                    delta = current_score - prev_receiver_score

                rows.append({
                    "time": nxt["datetime"],
                    "delta": delta,
                    "trigger_text": row["text"],
                    "response_text": nxt["text"],
                    "sender_score": row["continuous_score"],
                    "receiver_score": current_score
                })

                prev_receiver_score = current_score

        rdf = pd.DataFrame(rows)

        if len(rdf) == 0:
            continue

        color_list = [
            "#22c55e" if x > 0.05 else
            "#ef4444" if x < -0.05 else
            "#94a3b8"
            for x in rdf["delta"]
        ]

        fig.add_trace(
            go.Bar(
                x=rdf["time"],
                y=rdf["delta"],
                visible=(trace_index == 0),
                name=f"{sender} → {receiver}",

                marker=dict(
                    color=color_list
                ),

                customdata=list(zip(
                    rdf["trigger_text"],
                    rdf["response_text"],
                    rdf["sender_score"],
                    rdf["receiver_score"]
                )),

                hovertemplate=
                    "<b>Smjer:</b> " + sender + " → " + receiver + "<br>" +
                    "<b>Vrijeme:</b> %{x}<br>" +
                    "<b>Promjena:</b> %{y:.3f}<br><br>" +
                    "<b>Trigger poruka:</b><br>%{customdata[0]}<br><br>" +
                    "<b>Odgovor:</b><br>%{customdata[1]}<br><br>" +
                    "<b>Score sender:</b> %{customdata[2]:.3f}<br>" +
                    "<b>Score receiver:</b> %{customdata[3]:.3f}<extra></extra>"
            )
        )

        traces_per_direction[f"{sender} → {receiver}"] = trace_index
        trace_index += 1

    if trace_index == 0:
        return go.Figure()

    buttons = []

    for label, idx in traces_per_direction.items():

        visible = [False] * trace_index
        visible[idx] = True

        buttons.append(
            dict(
                label=label,
                method="update",
                args=[
                    {"visible": visible},
                    {"title": f"Utjecaj poruka: {label}"}
                ]
            )
        )

    # all
    buttons.insert(
        0,
        dict(
            label="Prikaži sve",
            method="update",
            args=[
                {"visible": [True] * trace_index},
                {"title": "Utjecaj poruka između sudionika"}
            ]
        )
    )

    fig.update_layout(

        title="Utjecaj poruka između sudionika",

        template="plotly_dark",

        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",

        height=620,

        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                x=1.02,
                y=1.15,
                showactive=True,
                buttons=buttons
            )
        ],

        xaxis_title="Vrijeme",
        yaxis_title="Promjena emotional score-a",

        hovermode="closest"
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="rgba(255,255,255,0.35)"
    )

    fig.update_yaxes(
        zeroline=False
    )

    return fig