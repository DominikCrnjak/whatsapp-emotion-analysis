"""
goemotions_module.py
=========================================================
Upgradeani GoEmotions modul za analizu emocija.

Model:
    SamLowe/roberta-base-go_emotions

Podržana 2 moda rada:

1) FULL MODE
   -> originalne GoEmotions klase

2) EKMAN MODE
   -> anger, joy, sadness, fear, surprise, neutral

Izlazni CSV (usklađen s bert_module.py):

    predicted_emotion
    emotion_score
    used_emotions
    continuous_score
    sentiment_group
    model_name
    model_mode

Namjena:
- Streamlit app
- Diplomski rad
- WhatsApp CSV analiza
- Direktna usporedba modela

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
# VALENCE MAP
# =====================================================

VALENCE_MAP = {
    # negative
    "anger": -1.00,
    "annoyance": -0.85,
    "disapproval": -0.75,
    "disgust": -1.00,
    "fear": -0.90,
    "nervousness": -0.70,
    "sadness": -0.90,
    "grief": -1.00,
    "disappointment": -0.80,
    "remorse": -0.70,
    "embarrassment": -0.55,

    # slight mixed
    "confusion": -0.20,
    "realization": 0.05,
    "curiosity": 0.20,
    "surprise": 0.15,

    # neutral
    "neutral": 0.00,

    # positive
    "approval": 0.45,
    "admiration": 0.70,
    "gratitude": 0.95,
    "joy": 1.00,
    "amusement": 0.80,
    "love": 1.00,
    "optimism": 0.75,
    "pride": 0.70,
    "relief": 0.65,
    "excitement": 0.95,
    "caring": 0.60,
    "desire": 0.45,
}

DEFAULT_VALENCE = 0.0

# =====================================================
# GOEMOTIONS -> EKMAN MAP
# =====================================================

EKMAN_MAP = {
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",
    "disgust": "anger",

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

    "sadness": "sadness",
    "grief": "sadness",
    "disappointment": "sadness",
    "remorse": "sadness",
    "embarrassment": "sadness",

    "fear": "fear",
    "nervousness": "fear",

    "surprise": "surprise",
    "realization": "surprise",
    "confusion": "surprise",
    "curiosity": "surprise",

    "neutral": "neutral",
}

# =====================================================
# VAD MAP (0-1 scale)
# Theoretical base:
# Russell Circumplex + PAD model + Warriner lexicon inspired
# =====================================================

VAD_MAP = {
    "anger": (0.15, 0.82, 0.72),
    "annoyance": (0.28, 0.62, 0.55),
    "disapproval": (0.25, 0.50, 0.45),
    "disgust": (0.10, 0.72, 0.48),

    "fear": (0.10, 0.90, 0.18),
    "nervousness": (0.22, 0.82, 0.25),

    "sadness": (0.18, 0.35, 0.18),
    "grief": (0.05, 0.28, 0.08),
    "disappointment": (0.25, 0.38, 0.28),
    "remorse": (0.22, 0.42, 0.22),
    "embarrassment": (0.35, 0.58, 0.30),

    "confusion": (0.42, 0.52, 0.32),
    "realization": (0.52, 0.42, 0.48),
    "surprise": (0.55, 0.76, 0.50),

    "neutral": (0.50, 0.10, 0.50),

    "curiosity": (0.68, 0.52, 0.45),
    "desire": (0.72, 0.62, 0.62),
    "approval": (0.72, 0.35, 0.55),
    "admiration": (0.82, 0.58, 0.66),
    "gratitude": (0.88, 0.48, 0.62),
    "joy": (0.92, 0.72, 0.66),
    "amusement": (0.86, 0.78, 0.62),
    "love": (0.96, 0.62, 0.72),
    "optimism": (0.84, 0.55, 0.62),
    "pride": (0.86, 0.64, 0.82),
    "relief": (0.78, 0.32, 0.56),
    "excitement": (0.92, 0.92, 0.76),
    "caring": (0.80, 0.42, 0.52)
}


# =====================================================
# MODEL CACHE
# =====================================================

_classifier = None


def load_model():
    """
    Učitaj model samo jednom.
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
# DATA CLEANING
# =====================================================

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Očisti CSV.
    """
    df = df.copy()

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    df = df.dropna(subset=["text"])

    df["text"] = (
        df["text"]
        .astype(str)
        .str.strip()
    )

    df = df[df["text"] != ""]

    df = df.sort_values("datetime").reset_index(drop=True)

    return df


# =====================================================
# HELPERS
# =====================================================

def select_strong_emotions(
    preds: List[Dict],
    threshold: float = MIN_SCORE_THRESHOLD,
    max_emotions: int = MAX_EMOTIONS_PER_MESSAGE
):
    """
    Uzmi dovoljno jake emocije.
    Ako nijedna ne prođe threshold -> top1
    """
    preds_sorted = sorted(
        preds,
        key=lambda x: x["score"],
        reverse=True
    )

    selected = [
        p for p in preds_sorted
        if p["score"] >= threshold
    ]

    if not selected:
        selected = [preds_sorted[0]]

    return selected[:max_emotions]


def score_group(score: float):
    """
    Pretvori continuous score u grupu.
    """
    if score <= -0.35:
        return "negative"
    elif score >= 0.35:
        return "positive"
    else:
        return "neutral"


def compute_continuous_score(preds: List[Dict]):
    """
    Ponderirani prosjek valencije.
    """
    weighted_sum = 0.0
    total = 0.0

    for p in preds:
        label = p["label"]
        score = p["score"]

        val = VALENCE_MAP.get(label, DEFAULT_VALENCE)

        weighted_sum += val * score
        total += score

    if total == 0:
        return 0.0

    return weighted_sum / total


def map_to_ekman(preds: List[Dict]):
    """
    Zbroji scoreove po Ekman klasama.
    """
    scores = defaultdict(float)

    for p in preds:
        label = p["label"]
        score = p["score"]

        ek = EKMAN_MAP.get(label, "neutral")
        scores[ek] += score

    return dict(scores)


def compute_vad_scores(preds):
    v_sum = 0
    a_sum = 0
    d_sum = 0
    total = 0

    for p in preds:
        label = p["label"]
        weight = p["score"]

        v, a, d = VAD_MAP.get(label, (0.5, 0.5, 0.5))

        v_sum += v * weight
        a_sum += a * weight
        d_sum += d * weight
        total += weight

    if total == 0:
        return (0.5, 0.5, 0.5)

    return (
        v_sum / total,
        a_sum / total,
        d_sum / total
    )


# =====================================================
# PREDICTION MODES
# =====================================================

def predict_full(preds: List[Dict]):
    """
    Full GoEmotions mode.
    """
    preds_sorted = sorted(
        preds,
        key=lambda x: x["score"],
        reverse=True
    )

    top = preds_sorted[0]

    used = select_strong_emotions(preds_sorted)

    used_text = "; ".join(
        f'{p["label"]} ({p["score"]:.3f})'
        for p in used
    )

    cont = compute_continuous_score(used)

    v, a, d = compute_vad_scores(used)

    return {
        "predicted_emotion": top["label"],
        "emotion_score": top["score"],
        "used_emotions": used_text,
        "continuous_score": cont,
        "sentiment_group": score_group(cont),
        "vad_valence": v,
        "vad_arousal": a,
        "vad_dominance": d
    }


def predict_ekman(preds: List[Dict]):
    """
    Ekman mode.
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

    # Ekman valence
    ekman_val = {
        "anger": -0.65,
        "sadness": -0.55,
        "fear": -0.50,
        "joy": 0.75,
        "neutral": 0.0
    }

    total = sum(mapped.values())

    cont = 0.0
    if total > 0:
        cont = sum(
            ekman_val[k] * v
            for k, v in mapped.items()
        ) / total

    pseudo_preds = [{"label":k, "score":v} for k,v in mapped.items()]
    v, a, d = compute_vad_scores(pseudo_preds)

    return {
        "predicted_emotion": best_label,
        "emotion_score": best_score,
        "used_emotions": used_text,
        "continuous_score": cont,
        "sentiment_group": score_group(cont),
        "vad_valence": v,
        "vad_arousal": a,
        "vad_dominance": d
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
    Pokreni analizu cijelog CSV-a.
    """

    if mode not in ["full", "ekman"]:
        raise ValueError("mode mora biti full ili ekman")

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
    final_used = []
    final_cont = []
    final_groups = []
    final_v = []
    final_a = []
    final_d = []

    for preds in predictions:

        if mode == "full":
            result = predict_full(preds)
        else:
            result = predict_ekman(preds)

        final_labels.append(result["predicted_emotion"])
        final_scores.append(result["emotion_score"])
        final_used.append(result["used_emotions"])
        final_cont.append(result["continuous_score"])
        final_groups.append(result["sentiment_group"])
        final_v.append(result["vad_valence"])
        final_a.append(result["vad_arousal"])
        final_d.append(result["vad_dominance"])

    df["predicted_emotion"] = final_labels
    df["emotion_score"] = final_scores
    df["used_emotions"] = final_used
    df["continuous_score"] = final_cont
    df["sentiment_group"] = final_groups
    df["model_name"] = "goemotions"
    df["model_mode"] = mode
    df["vad_valence"] = final_v
    df["vad_arousal"] = final_a
    df["vad_dominance"] = final_d

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
# SUMMARY
# =====================================================

def emotion_summary(df: pd.DataFrame):
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
# TEST
# =====================================================

if __name__ == "__main__":

    df = run_goemotions_analysis(
        input_csv="outputs/chat.csv",
        output_csv="outputs/chat_go.csv",
        mode="ekman"
    )

    print(df.head())