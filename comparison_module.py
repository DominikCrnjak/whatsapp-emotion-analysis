"""
comparison_module.py
=========================================================
USPOREDBA MODELA EMOCIJA (OPTIMIZED)

Uspoređuje:

1. GoEmotions model (Ekman mode)
2. Custom BERT model

Rezultat:
- merged tablica po poruci
- agreement %
- confusion matrix
- mismatch primjeri
- summary statistika

DODATNO:
- sprema oba rezultata za vizualizacije
- robustniji merge
- fallback stupci
- čišći kod

=========================================================
"""

import os
import pandas as pd

from sklearn.metrics import confusion_matrix

from goemotions_module import run_goemotions_analysis
from bert_module import run_bert_analysis


# =====================================================
# GLOBAL POSTAVKE
# =====================================================

OUTPUT_DIR = "outputs"

# CSV datoteke koje će koristiti app.py compare screen
GO_COMPARE_PATH = os.path.join(
    OUTPUT_DIR,
    "comparison_goemotions.csv"
)

BERT_COMPARE_PATH = os.path.join(
    OUTPUT_DIR,
    "comparison_bert.csv"
)

# privremeni merge output (nije nužno potreban ali korisno)
MERGED_COMPARE_PATH = os.path.join(
    OUTPUT_DIR,
    "comparison_merged.csv"
)

# standardni redoslijed labela
LABELS = [
    "anger",
    "joy",
    "sadness",
    "fear",
    "surprise",
    "neutral"
]


# =====================================================
# INTERNAL HELPERS
# =====================================================

def _ensure_output_dir():
    """
    Kreira outputs folder ako ne postoji.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _safe_read_csv(path):
    """
    Učitaj CSV sigurno.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV nije pronađen: {path}")

    return pd.read_csv(path)


def _ensure_required_columns(df):
    """
    Provjera osnovnih stupaca.
    """
    required = ["turn_id", "text"]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Nedostaje stupac: {col}")

    # speaker nije obavezan
    if "speaker" not in df.columns:
        df["speaker"] = "Speaker"

    return df


# =====================================================
# GLAVNA FUNKCIJA
# =====================================================

def run_model_comparison(
    input_csv: str
):
    """
    Pokreće oba modela i vraća usporedbu.

    Također sprema:

    outputs/comparison_goemotions.csv
    outputs/comparison_bert.csv
    outputs/comparison_merged.csv
    """

    _ensure_output_dir()

    # =================================================
    # 1. GOEMOTIONS ANALIZA
    # =================================================
    run_goemotions_analysis(
        input_csv=input_csv,
        output_csv=GO_COMPARE_PATH,
        mode="ekman"
    )

    # =================================================
    # 2. BERT ANALIZA
    # =================================================
    run_bert_analysis(
        input_csv=input_csv,
        output_csv=BERT_COMPARE_PATH
    )

    # =================================================
    # 3. LOAD REZULTATA
    # =================================================
    go_df = _safe_read_csv(GO_COMPARE_PATH)
    bert_df = _safe_read_csv(BERT_COMPARE_PATH)

    go_df = _ensure_required_columns(go_df)
    bert_df = _ensure_required_columns(bert_df)

    # =================================================
    # 4. RENAME STUPACA
    # =================================================
    go_df = go_df.rename(
        columns={
            "predicted_emotion": "goemotion",
            "continuous_score": "go_score"
        }
    )

    bert_df = bert_df.rename(
        columns={
            "predicted_emotion": "bert",
            "continuous_score": "bert_score"
        }
    )

    # =================================================
    # 5. MERGE TABLICA
    # =================================================
    keep_cols = [
        "turn_id",
        "speaker",
        "text"
    ]

    go_keep = keep_cols + ["goemotion"]

    if "go_score" in go_df.columns:
        go_keep.append("go_score")

    bert_keep = ["turn_id", "bert"]

    if "bert_score" in bert_df.columns:
        bert_keep.append("bert_score")

    merged = go_df[go_keep].merge(
        bert_df[bert_keep],
        on="turn_id",
        how="inner"
    )

    # =================================================
    # 6. PODUDARANJE
    # =================================================
    merged["match"] = (
        merged["goemotion"] == merged["bert"]
    )

    agreement = round(
        merged["match"].mean() * 100,
        2
    )

    # =================================================
    # 7. SCORE RAZLIKA (ako postoji)
    # =================================================
    if (
        "go_score" in merged.columns and
        "bert_score" in merged.columns
    ):
        merged["score_difference"] = (
            merged["go_score"] - merged["bert_score"]
        ).round(4)

    # =================================================
    # 8. SAVE MERGED
    # =================================================
    merged.to_csv(
        MERGED_COMPARE_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    return merged, agreement


# =====================================================
# CONFUSION MATRIX
# =====================================================

def get_confusion_df(compare_df):
    """
    Kreira confusion matrix.

    Redovi   = GoEmotions
    Stupci   = BERT
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
# NESLAGANJA
# =====================================================

def get_mismatches(
    compare_df,
    limit=20
):
    """
    Vraća samo poruke gdje se modeli ne slažu.
    """

    mism = compare_df[
        compare_df["match"] == False
    ].copy()

    return mism.head(limit)


# =====================================================
# SUMMARY
# =====================================================

def get_summary(compare_df):
    """
    Osnovna statistika usporedbe.
    """

    total = len(compare_df)

    matched = int(
        compare_df["match"].sum()
    )

    mismatched = total - matched

    return {
        "total": total,
        "matched": matched,
        "mismatched": mismatched
    }


# =====================================================
# EXTRA ANALYTICS
# =====================================================

def get_top_disagreements(compare_df, top_n=10):
    """
    Najčešći parovi neslaganja.

    primjer:
    GoEmotions = joy
    BERT = neutral
    """

    mism = compare_df[
        compare_df["match"] == False
    ].copy()

    if len(mism) == 0:
        return pd.DataFrame()

    out = (
        mism.groupby(
            ["goemotion", "bert"]
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            "count",
            ascending=False
        )
        .head(top_n)
    )

    return out