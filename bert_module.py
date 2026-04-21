"""
bert_module.py
=========================================================
Modul za predikciju emocija pomoću treniranog BERT modela.

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
    model_name

Podržava:
- batch inference
- automatsko učitavanje modela jednom
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
# SINGLE BATCH PREDICTION
# =====================================================

def predict_batch(texts: List[str]):
    """
    Batch predikcija.
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

    scores, ids = torch.max(probs, dim=1)

    labels = [
        id2label[i.item()]
        for i in ids
    ]

    scores = [
        float(s.item())
        for s in scores
    ]

    return labels, scores


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

    for i in range(0, len(texts), batch_size):

        batch = texts[i:i + batch_size]

        labels, scores = predict_batch(batch)

        final_labels.extend(labels)
        final_scores.extend(scores)

    df["predicted_emotion"] = final_labels
    df["emotion_score"] = final_scores
    df["model_name"] = "custom_bert"

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