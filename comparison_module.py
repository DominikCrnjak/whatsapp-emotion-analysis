"""
comparison_module.py
=========================================================
Usporedba modela emocija:

1. GoEmotions (mapiran u Ekmann model)
2. BERT model

Rezultat:
- CSV/tablica usporedbe
- agreement %
- confusion matrix
- mismatch primjeri


=========================================================
"""

import pandas as pd
from sklearn.metrics import confusion_matrix

from goemotions_module import run_goemotions_analysis
from bert_module import run_bert_analysis


# =====================================================
# LABEL ORDER
# =====================================================

LABELS = [
    "anger",
    "joy",
    "sadness",
    "fear",
    "surprise",
    "neutral"
]


# =====================================================
# RUN COMPARISON
# =====================================================

def run_model_comparison(
    input_csv: str,
    temp_go_csv: str = "outputs/_temp_go_compare.csv",
    temp_bert_csv: str = "outputs/_temp_bert_compare.csv"
):
    """
    Pokreće oba modela i spaja rezultate.
    """

    # -----------------------------------------
    # GoEmotions -> Ekman
    # -----------------------------------------
    run_goemotions_analysis(
        input_csv=input_csv,
        output_csv=temp_go_csv,
        mode="ekman"
    )

    # -----------------------------------------
    # BERT
    # -----------------------------------------
    run_bert_analysis(
        input_csv=input_csv,
        output_csv=temp_bert_csv
    )

    # -----------------------------------------
    # Load rezultata
    # -----------------------------------------
    go_df = pd.read_csv(temp_go_csv)
    bert_df = pd.read_csv(temp_bert_csv)

    # -----------------------------------------
    # Rename columns
    # -----------------------------------------
    go_df = go_df.rename(
        columns={"predicted_emotion": "goemotion"}
    )

    bert_df = bert_df.rename(
        columns={"predicted_emotion": "bert"}
    )

    # -----------------------------------------
    # Merge
    # -----------------------------------------
    keep_cols = ["turn_id", "speaker", "text"]

    merged = go_df[keep_cols + ["goemotion"]].merge(
        bert_df[["turn_id", "bert"]],
        on="turn_id",
        how="inner"
    )

    # -----------------------------------------
    # Match?
    # -----------------------------------------
    merged["match"] = merged["goemotion"] == merged["bert"]

    agreement = round(
        merged["match"].mean() * 100,
        2
    )

    return merged, agreement


# =====================================================
# CONFUSION MATRIX
# =====================================================

def get_confusion_df(compare_df):
    """
    DataFrame confusion matrix.
    """

    cm = confusion_matrix(
        compare_df["goemotion"],
        compare_df["bert"],
        labels=LABELS
    )

    cm_df = pd.DataFrame(
        cm,
        index=LABELS,
        columns=LABELS
    )

    return cm_df


# =====================================================
# MISMATCHES
# =====================================================

def get_mismatches(compare_df, limit=20):
    """
    Samo poruke gdje se modeli ne slažu.
    """

    mism = compare_df[
        compare_df["match"] == False
    ].copy()

    return mism.head(limit)


# =====================================================
# SUMMARY STATS
# =====================================================

def get_summary(compare_df):
    """
    Dodatna statistika.
    """

    total = len(compare_df)
    matched = int(compare_df["match"].sum())
    mismatched = total - matched

    return {
        "total": total,
        "matched": matched,
        "mismatched": mismatched
    }