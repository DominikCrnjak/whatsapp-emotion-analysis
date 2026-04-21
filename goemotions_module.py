"""
goemotions_module.py
=========================================================
Optimizirani modul za analizu emocija koristeći:

Model:
    SamLowe/roberta-base-go_emotions

Podržana 2 moda rada:

1) FULL MODE
   -> koristi originalne GoEmotions klase

2) EKMAN MODE
   -> mapira GoEmotions emocije u:
      anger, joy, sadness, fear, surprise, neutral

Namjena:
- Streamlit app
- Diplomski rad
- Batch obrada CSV WhatsApp razgovora

=========================================================
"""

import pandas as pd
from transformers import pipeline
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional

# =====================================================
# GLOBAL SETTINGS
# =====================================================

MODEL_NAME = "SamLowe/roberta-base-go_emotions"

MIN_SCORE_THRESHOLD = 0.15
MAX_EMOTIONS_PER_MESSAGE = 6
DEFAULT_BATCH_SIZE = 32

# =====================================================
# GOEMOTIONS -> EKMAN MAP
# =====================================================

EKMAN_MAP = {
    # anger
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",
    "disgust": "anger",

    # joy
    "joy": "joy",
    "amusement": "joy",
    "approval": "joy",
    "admiration": "joy",
    "gratitude": "joy",
    "love": "joy",
    "optimism": "joy",
    "pride": "joy",
    "relief": "joy",
    "excitement": "joy",
    "caring": "joy",
    "desire": "joy",

    # sadness
    "sadness": "sadness",
    "grief": "sadness",
    "disappointment": "sadness",
    "remorse": "sadness",
    "embarrassment": "sadness",

    # fear
    "fear": "fear",
    "nervousness": "fear",

    # surprise
    "surprise": "surprise",
    "realization": "surprise",
    "confusion": "surprise",
    "curiosity": "surprise",

    # neutral
    "neutral": "neutral",
}

VALID_EKMAN = [
    "anger",
    "joy",
    "sadness",
    "fear",
    "surprise",
    "neutral",
]

# =====================================================
# MODEL CACHE
# =====================================================

_classifier = None


def load_model():
    """
    Učitaj model samo jednom (cache).
    """
    global _classifier

    if _classifier is None:
        _classifier = pipeline(
            "text-classification",
            model=MODEL_NAME,
            top_k=None,
            truncation=True
        )

    return _classifier


# =====================================================
# HELPERS
# =====================================================

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Očisti ulazni CSV.
    """
    df = df.copy()

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]

    df = df.sort_values("datetime").reset_index(drop=True)

    return df


def select_strong_emotions(
    preds: List[Dict],
    threshold: float = MIN_SCORE_THRESHOLD,
    max_emotions: int = MAX_EMOTIONS_PER_MESSAGE
):
    """
    Uzmi samo dovoljno jake emocije.
    Ako nijedna ne prođe threshold -> top1.
    """
    preds_sorted = sorted(preds, key=lambda x: x["score"], reverse=True)

    selected = [p for p in preds_sorted if p["score"] >= threshold]

    if not selected:
        selected = [preds_sorted[0]]

    return selected[:max_emotions]


def map_to_ekman(preds: List[Dict]) -> Dict[str, float]:
    """
    Zbroji scoreove nakon mapiranja u Ekman klase.
    """
    scores = defaultdict(float)

    for p in preds:
        label = p["label"]
        score = p["score"]

        ek = EKMAN_MAP.get(label, "neutral")
        scores[ek] += score

    return dict(scores)


# =====================================================
# SINGLE TEXT PREDICTION
# =====================================================

def predict_full(preds: List[Dict]):
    """
    Vrati top1 original GoEmotions label.
    """
    preds_sorted = sorted(preds, key=lambda x: x["score"], reverse=True)

    top = preds_sorted[0]

    used = select_strong_emotions(preds_sorted)

    used_text = "; ".join(
        f'{p["label"]} ({p["score"]:.3f})'
        for p in used
    )

    return {
        "predicted_emotion": top["label"],
        "emotion_score": top["score"],
        "used_emotions": used_text
    }


def predict_ekman(preds: List[Dict]):
    """
    Vrati top1 Ekman label.
    """
    strong = select_strong_emotions(preds)

    mapped = map_to_ekman(strong)

    best_label = max(mapped, key=mapped.get)
    best_score = mapped[best_label]

    used_text = "; ".join(
        f"{k} ({v:.3f})"
        for k, v in sorted(
            mapped.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )

    return {
        "predicted_emotion": best_label,
        "emotion_score": best_score,
        "used_emotions": used_text
    }


# =====================================================
# MAIN ANALYSIS
# =====================================================

def run_goemotions_analysis(
    input_csv: str,
    output_csv: Optional[str] = None,
    mode: str = "full",
    batch_size: int = DEFAULT_BATCH_SIZE
):
    """
    Analiza cijelog CSV-a.

    Parameters
    ----------
    input_csv : str
        Ulazni CSV

    output_csv : str
        Gdje spremiti rezultat

    mode : str
        "full" ili "ekman"

    batch_size : int
        Batch inference

    Returns
    -------
    DataFrame
    """

    if mode not in ["full", "ekman"]:
        raise ValueError("mode mora biti 'full' ili 'ekman'")

    df = pd.read_csv(input_csv)
    df = clean_dataframe(df)

    classifier = load_model()

    texts = df["text"].tolist()

    predictions = classifier(
        texts,
        batch_size=batch_size
    )

    final_labels = []
    final_scores = []
    used_emotions = []

    for preds in predictions:

        if mode == "full":
            result = predict_full(preds)

        else:
            result = predict_ekman(preds)

        final_labels.append(result["predicted_emotion"])
        final_scores.append(result["emotion_score"])
        used_emotions.append(result["used_emotions"])

    df["predicted_emotion"] = final_labels
    df["emotion_score"] = final_scores
    df["used_emotions"] = used_emotions
    df["model_mode"] = mode

    if output_csv:
        Path(output_csv).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        df.to_csv(
            output_csv,
            index=False,
            encoding="utf-8-sig"
        )

    return df


# =====================================================
# SUMMARY STATS
# =====================================================

def emotion_summary(df: pd.DataFrame):
    """
    Broj emocija.
    """
    return (
        df["predicted_emotion"]
        .value_counts()
        .reset_index()
        .rename(
            columns={
                "index": "emotion",
                "predicted_emotion": "count"
            }
        )
    )


# =====================================================
# TEST RUN
# =====================================================

if __name__ == "__main__":

    result = run_goemotions_analysis(
        input_csv="outputs/chat.csv",
        output_csv="outputs/chat_goemotions.csv",
        mode="ekman"
    )

    print(result.head())