import streamlit as st
import os
import pandas as pd
from pathlib import Path

# ==================================================
# IMPORT MODULA
# ==================================================

# TXT -> CSV parser
from whatsapp_to_csv import write_one_csv, load_json, save_json

# Analiza modela
from goemotions_module import run_goemotions_analysis
from bert_module import run_bert_analysis

# Vizualizacije
from visualization_module import (
    load_result_csv,
    create_pie_chart,
    create_bar_chart,
    create_timeline_chart
)

# Usporedba modela
from comparison_module import (
    run_model_comparison,
    get_confusion_df,
    get_mismatches,
    get_summary
)

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Analiza emocija WhatsApp razgovora",
    page_icon="📱",
    layout="centered"
)

# ==================================================
# SESSION STATE INIT
# ==================================================
defaults = {
    "step": "upload",
    "csv_path": "",
    "go_result": "",
    "bert_result": "",
    "visual_csv": "",
    "visual_title": "",
    "compare_df": None,
    "agreement": 0.0
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ==================================================
# HELPER FUNCTIONS
# ==================================================
def reset_app():
    """
    Reset cijele aplikacije na početno stanje.
    """
    for key, value in defaults.items():
        st.session_state[key] = value


def show_dataframe(path):
    """
    Učitaj i prikaži CSV.
    """
    df = pd.read_csv(path)
    st.dataframe(df, use_container_width=True)


# ==================================================
# HEADER
# ==================================================
st.title("📱 Analiza emocija WhatsApp razgovora")

# ==================================================
# STEP 1 - UPLOAD WHATSAPP TXT
# ==================================================
if st.session_state.step == "upload":

    st.subheader("1. Učitaj WhatsApp razgovor")

    uploaded_file = st.file_uploader(
        "Odaberi WhatsApp .txt datoteku",
        type=["txt"]
    )

    if uploaded_file:

        # kreiraj foldere
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)
        os.makedirs(".wa_state", exist_ok=True)

        file_path = os.path.join("uploads", uploaded_file.name)

        # spremi uploadani file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("Datoteka učitana.")

        if st.button("Pokreni analizu"):

            with st.spinner("Pretvaram TXT u CSV..."):

                txt_path = Path(file_path)
                out_csv = Path("outputs") / txt_path.with_suffix(".csv").name

                speaker_map_path = Path(".wa_state") / "speaker_map.json"
                speaker_map = load_json(speaker_map_path, default={})

                ok, msg = write_one_csv(
                    txt_path=txt_path,
                    out_csv_path=out_csv,
                    speaker_map=speaker_map,
                    keep_system=False
                )

                save_json(speaker_map_path, speaker_map)

                if ok:
                    st.session_state.csv_path = str(out_csv)
                    st.session_state.step = "results"
                    st.rerun()
                else:
                    st.error(msg)

# ==================================================
# STEP 2 - GLAVNI IZBORNIK
# ==================================================
elif st.session_state.step == "results":

    st.success("CSV uspješno kreiran.")
    st.subheader("2. Odaberi opciju")

    col1, col2, col3, col4 = st.columns(4)

    # ------------------------------------------------
    # OTVORI CSV
    # ------------------------------------------------
    with col1:
        if st.button("📄 CSV"):
            show_dataframe(st.session_state.csv_path)

    # ------------------------------------------------
    # GOEMOTIONS
    # ------------------------------------------------
    with col2:
        if st.button("😊 GoEmotions"):
            st.session_state.step = "goemotions"
            st.rerun()

    # ------------------------------------------------
    # BERT
    # ------------------------------------------------
    with col3:
        if st.button("🤖 BERT"):

            with st.spinner("Pokrećem BERT analizu..."):

                output_path = "outputs/bert_result.csv"

                run_bert_analysis(
                    input_csv=st.session_state.csv_path,
                    output_csv=output_path
                )

                st.session_state.bert_result = output_path
                st.session_state.step = "bert_done"
                st.rerun()

    # ------------------------------------------------
    # USPOREDBA MODELA
    # ------------------------------------------------
    with col4:
        if st.button("⚖️ Usporedba"):

            with st.spinner("Pokrećem oba modela i radim usporedbu..."):

                compare_df, agreement = run_model_comparison(
                    input_csv=st.session_state.csv_path
                )

                st.session_state.compare_df = compare_df
                st.session_state.agreement = agreement
                st.session_state.step = "compare"
                st.rerun()

    st.divider()

    if st.button("⬅ Novi razgovor"):
        reset_app()
        st.rerun()

# ==================================================
# STEP 3 - GOEMOTIONS MENU
# ==================================================
elif st.session_state.step == "goemotions":

    st.subheader("😊 GoEmotions analiza")
    st.write("Odaberi način rada modela:")

    col1, col2 = st.columns(2)

    # FULL MODEL
    with col1:
        if st.button("😊 Original GoEmotions"):

            with st.spinner("Pokrećem analizu..."):

                output_path = "outputs/goemotions_full.csv"

                run_goemotions_analysis(
                    input_csv=st.session_state.csv_path,
                    output_csv=output_path,
                    mode="full"
                )

                st.session_state.go_result = output_path
                st.session_state.step = "go_done"
                st.rerun()

    # EKMAN MODEL
    with col2:
        if st.button("🎭 Ekman model"):

            with st.spinner("Pokrećem analizu..."):

                output_path = "outputs/goemotions_ekman.csv"

                run_goemotions_analysis(
                    input_csv=st.session_state.csv_path,
                    output_csv=output_path,
                    mode="ekman"
                )

                st.session_state.go_result = output_path
                st.session_state.step = "go_done"
                st.rerun()

    st.divider()

    if st.button("⬅ Natrag"):
        st.session_state.step = "results"
        st.rerun()

# ==================================================
# STEP 4 - GO RESULT
# ==================================================
elif st.session_state.step == "go_done":

    st.success("GoEmotions analiza završena.")
    st.write(st.session_state.go_result)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📄 CSV rezultat"):
            show_dataframe(st.session_state.go_result)

    with col2:
        if st.button("📊 Vizualizacije"):
            st.session_state.visual_csv = st.session_state.go_result
            st.session_state.visual_title = "GoEmotions vizualizacija"
            st.session_state.step = "visuals"
            st.rerun()

    st.divider()

    if st.button("⬅ Povratak"):
        st.session_state.step = "results"
        st.rerun()

# ==================================================
# STEP 5 - BERT RESULT
# ==================================================
elif st.session_state.step == "bert_done":

    st.success("BERT analiza završena.")
    st.write(st.session_state.bert_result)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📄 CSV rezultat"):
            show_dataframe(st.session_state.bert_result)

    with col2:
        if st.button("📊 Vizualizacije"):
            st.session_state.visual_csv = st.session_state.bert_result
            st.session_state.visual_title = "BERT vizualizacija"
            st.session_state.step = "visuals"
            st.rerun()

    st.divider()

    if st.button("⬅ Povratak"):
        st.session_state.step = "results"
        st.rerun()

# ==================================================
# STEP 6 - VISUALIZATIONS
# ==================================================
elif st.session_state.step == "visuals":

    st.subheader(f"📊 {st.session_state.visual_title}")

    df = load_result_csv(st.session_state.visual_csv)

    # statistika
    c1, c2 = st.columns(2)

    with c1:
        st.metric("Broj poruka", len(df))

    with c2:
        if "speaker" in df.columns:
            st.metric("Broj sudionika", df["speaker"].nunique())

    st.divider()

    # PIE
    st.markdown("### 🥧 Distribucija emocija")
    st.plotly_chart(create_pie_chart(df), use_container_width=True)

    st.divider()

    # BAR
    st.markdown("### 📊 Broj poruka po emociji")
    st.plotly_chart(create_bar_chart(df), use_container_width=True)

    st.divider()

    # TIMELINE
    st.markdown("### 📈 Emocionalni tok razgovora")
    st.plotly_chart(create_timeline_chart(df), use_container_width=True)

    st.divider()

    if st.button("⬅ Povratak"):
        st.session_state.step = "results"
        st.rerun()

# ==================================================
# STEP 7 - COMPARISON SCREEN
# ==================================================
elif st.session_state.step == "compare":

    st.subheader("⚖️ Usporedba modela")

    compare_df = st.session_state.compare_df
    agreement = st.session_state.agreement

    summary = get_summary(compare_df)

    # ----------------------------------------------
    # METRIKE
    # ----------------------------------------------
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Ukupno poruka", summary["total"])

    with c2:
        st.metric("Podudaranja", summary["matched"])

    with c3:
        st.metric("Nepodudaranja", summary["mismatched"])

    with c4:
        st.metric("Agreement %", f"{agreement}%")

    st.divider()

    # ----------------------------------------------
    # TABLICA REZULTATA
    # ----------------------------------------------
    st.markdown("### 📄 Rezultati po poruci")

    st.dataframe(
        compare_df,
        use_container_width=True
    )

    st.divider()

    # ----------------------------------------------
    # CONFUSION MATRIX
    # ----------------------------------------------
    st.markdown("### 🔥 Confusion Matrix")

    cm_df = get_confusion_df(compare_df)

    st.dataframe(
        cm_df,
        use_container_width=True
    )

    st.divider()

    # ----------------------------------------------
    # NESLAGANJA
    # ----------------------------------------------
    st.markdown("### ⚠️ Primjeri gdje se modeli ne slažu")

    mismatches = get_mismatches(compare_df)

    st.dataframe(
        mismatches,
        use_container_width=True
    )

    st.divider()

    if st.button("⬅ Povratak"):
        st.session_state.step = "results"
        st.rerun()