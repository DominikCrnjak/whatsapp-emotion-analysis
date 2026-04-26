"""
bert_module.py
=========================================================
Upgradeani modul za predikciju emocija pomoću treniranog
BERT modela.

Namjena:
- Streamlit web app
- Diplomski rad
- WhatsApp CSV analiza

Ulazni CSV očekuje stupce:
    datetime
    text

Izlaz:
    predicted_emotion
    emotion_score
    used_emotions
    continuous_score
    sentiment_group
    model_name

Podržava:
- batch inference
- automatsko učitavanje modela jednom
- top N emocija
- kontinuirani emocionalni score
- CSV export

=========================================================
"""

import json
import pandas as pd
import torch

from pathlib import Path
from typing import Optional, List

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# =====================================================
# POSTAVKE
# =====================================================

MODEL_PATH = r"F:/Model/emotion_model/checkpoint-798"
LABEL_MAPPING_PATH = "./label_mapping.json"

MAX_LENGTH = 128
DEFAULT_BATCH_SIZE = 32
TOP_N_EMOTIONS = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# VALENCE MAPA
# =====================================================

VALENCE_MAP = {
    "anger": -0.65,
    "sadness": -0.55,
    "fear": -0.50,
    "joy": 0.75,
    "surprise": 0.10,
    "neutral": 0.0
}

DEFAULT_VALENCE = 0.0

# =====================================================
# VAD MAPA (teorijska aproksimacija Russell / Warriner stil)
# skala 0-1
# =====================================================

VAD_MAP = {
    "anger":   (0.15, 0.82, 0.72),
    "sadness": (0.18, 0.32, 0.22),
    "fear":    (0.12, 0.88, 0.18),
    "joy":     (0.90, 0.72, 0.68),
    "surprise": (0.55, 0.86, 0.52),
    "neutral": (0.50, 0.10, 0.50)
}

DEFAULT_VAD = (0.50, 0.50, 0.50)
# =====================================================
# CACHE
# =====================================================

_tokenizer = None
_model = None
_id2label = None


# =====================================================
# LOAD MODEL
# =====================================================

def load_bert_model():
    """
    Učitaj model samo jednom.
    """
    global _tokenizer, _model, _id2label

    if _tokenizer is None:

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        _model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH
        )

        _model.to(DEVICE)
        _model.eval()

        with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        _id2label = {
            int(k): v
            for k, v in mapping["id2label"].items()
        }

    return _tokenizer, _model, _id2label


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

def compute_continuous_score(top_items):
    """
    Ponderirani prosjek valencije.
    """
    weighted_sum = 0.0
    total_weight = 0.0

    for label, score in top_items:
        val = VALENCE_MAP.get(label, DEFAULT_VALENCE)
        weighted_sum += val * score
        total_weight += score

    if total_weight == 0:
        return 0.0

    return weighted_sum / total_weight

def compute_vad_scores(top_items):
    """
    Ponderirani prosjek VAD vrijednosti.
    """

    v_sum = 0.0
    a_sum = 0.0
    d_sum = 0.0
    total = 0.0

    for label, score in top_items:

        v, a, d = VAD_MAP.get(label, DEFAULT_VAD)

        v_sum += v * score
        a_sum += a * score
        d_sum += d * score
        total += score

    if total == 0:
        return DEFAULT_VAD

    return (
        v_sum / total,
        a_sum / total,
        d_sum / total
    )


def score_group(score):
    """
    Tekstualna grupa sentimenta.
    """
    if score <= -0.35:
        return "negative"
    elif score >= 0.35:
        return "positive"
    else:
        return "neutral"


# =====================================================
# BATCH PREDICTION
# =====================================================

def predict_batch(texts: List[str]):
    """
    Batch predikcija.

    Vraća:
    labels
    scores
    used_emotions
    continuous_scores
    sentiment_groups
    """
    tokenizer, model, id2label = load_bert_model()

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    inputs = {
        k: v.to(DEVICE)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)

    final_labels = []
    final_scores = []
    final_used = []
    final_cont_scores = []
    final_groups = []
    final_v = []
    final_a = []
    final_d = []

    for row in probs:

        row = row.cpu()

        indexed = list(enumerate(row.tolist()))

        indexed_sorted = sorted(
            indexed,
            key=lambda x: x[1],
            reverse=True
        )

        # top1
        best_id = indexed_sorted[0][0]
        best_score = indexed_sorted[0][1]

        best_label = id2label[best_id]

        # top N
        top_items = []

        for idx, score in indexed_sorted[:TOP_N_EMOTIONS]:
            label = id2label[idx]
            top_items.append((label, score))

        used_text = "; ".join(
            f"{label} ({score:.3f})"
            for label, score in top_items
        )

        cont_score = compute_continuous_score(top_items)
        group = score_group(cont_score)

        v, a, d = compute_vad_scores(top_items)

        final_labels.append(best_label)
        final_scores.append(float(best_score))
        final_used.append(used_text)
        final_cont_scores.append(float(cont_score))
        final_groups.append(group)
        final_v.append(v)
        final_a.append(a)
        final_d.append(d)

    return (
        final_labels,
        final_scores,
        final_used,
        final_cont_scores,
        final_groups,
        final_v,
        final_a,
        final_d
    )


# =====================================================
# MAIN ANALYSIS
# =====================================================

def run_bert_analysis(
    input_csv: str,
    output_csv: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE
):
    """
    Pokreni analizu cijelog CSV-a.
    """

    df = pd.read_csv(input_csv)
    df = clean_dataframe(df)

    texts = df["text"].tolist()

    final_labels = []
    final_scores = []
    final_used = []
    final_cont = []
    final_groups = []
    final_v = []
    final_a = []
    final_d = []

    for i in range(0, len(texts), batch_size):

        batch = texts[i:i + batch_size]

        (
            labels,
            scores,
            used,
            cont,
            groups,
            v_vals,
            a_vals,
            d_vals
        ) = predict_batch(batch)

        final_labels.extend(labels)
        final_scores.extend(scores)
        final_used.extend(used)
        final_cont.extend(cont)
        final_groups.extend(groups)
        final_v.extend(v_vals)
        final_a.extend(a_vals)
        final_d.extend(d_vals)

    # upis rezultata
    df["predicted_emotion"] = final_labels
    df["emotion_score"] = final_scores
    df["used_emotions"] = final_used
    df["continuous_score"] = final_cont
    df["sentiment_group"] = final_groups
    df["model_name"] = "custom_bert"
    df["vad_valence"] = final_v
    df["vad_arousal"] = final_a
    df["vad_dominance"] = final_d

    # spremanje
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
# TEST
# =====================================================

if __name__ == "__main__":

    df = run_bert_analysis(
        input_csv="outputs/chat.csv",
        output_csv="outputs/chat_bert.csv"
    )

    print(df.head())