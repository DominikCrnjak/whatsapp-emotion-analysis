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
    create_timeline_chart,
    create_score_histogram,
    create_speaker_comparison,
    create_vad_3d_chart,
    create_emotion_sankey,
    create_influence_timeline
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
# Mora biti prva Streamlit naredba
st.set_page_config(
    page_title="Analiza emocija WhatsApp razgovora",
    page_icon="📱",
    layout="centered"
)

# ==================================================
# CUSTOM CSS STILIZACIJA
# ==================================================
# Cilj:
# - svi gumbi jednake širine
# - modern dark izgled
# - hover animacije
# - bolji spacing
# - profesionalniji UI
st.markdown("""
<style>

/* Global background spacing */
.main .block-container{
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1050px;
}

/* ---------------------------
   BUTTON STYLE
----------------------------*/
div.stButton > button{
    width:100%;
    height:56px;
    border-radius:14px;
    border:1px solid rgba(255,255,255,0.12);
    background:rgba(255,255,255,0.03);
    color:white;
    font-size:18px;
    font-weight:600;
    transition:all 0.2s ease;
}

/* Hover efekt */
div.stButton > button:hover{
    border:1px solid #4ade80;
    background:rgba(74,222,128,0.08);
    transform:translateY(-2px);
}

/* Klik */
div.stButton > button:active{
    transform:scale(0.98);
}

/* Razmak među kolonama */
div[data-testid="column"]{
    padding:0.25rem;
}

/* Metric kartice */
div[data-testid="metric-container"]{
    border:1px solid rgba(255,255,255,0.08);
    padding:15px;
    border-radius:14px;
    background:rgba(255,255,255,0.03);
}

/* Dataframe ljepši */
div[data-testid="stDataFrame"]{
    border-radius:12px;
    overflow:hidden;
}

/* Horizontal line */
hr{
    border-color:rgba(255,255,255,0.08);
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# SESSION STATE DEFAULTS
# ==================================================
# Streamlit reruna cijeli file nakon svakog klika.
# Zato koristimo session_state za pamćenje stanja app-a.
defaults = {
    "step": "upload",       # trenutni ekran
    "csv_path": "",         # putanja generiranog CSV-a
    "go_result": "",        # rezultat GoEmotions modela
    "bert_result": "",      # rezultat BERT modela
    "visual_csv": "",       # CSV za vizualizacije
    "visual_title": "",     # naslov vizualizacija
    "compare_df": None,     # dataframe usporedbe modela
    "agreement": 0.0        # postotak slaganja modela
}

# Ako vrijednost ne postoji -> postavi default
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ==================================================
# HELPER FUNKCIJE
# ==================================================

def reset_app():
    """
    Reset cijele aplikacije na početno stanje.
    """
    for key, value in defaults.items():
        st.session_state[key] = value


def show_dataframe(path):
    """
    Učitaj CSV i prikaži ga u Streamlit tablici.
    """
    df = pd.read_csv(path)
    st.dataframe(df, use_container_width=True)


# ==================================================
# HEADER
# ==================================================
st.title("📱 Analiza emocija WhatsApp razgovora")

# ==================================================
# STEP 1
# UPLOAD WHATSAPP TXT DATOTEKE
# ==================================================
if st.session_state.step == "upload":

    st.subheader("1. Učitaj WhatsApp razgovor")

    uploaded_file = st.file_uploader(
        "Odaberi WhatsApp .txt datoteku",
        type=["txt"]
    )

    if uploaded_file:

        # Kreiraj potrebne foldere ako ne postoje
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)
        os.makedirs(".wa_state", exist_ok=True)

        # Putanja gdje spremamo uploadani txt
        file_path = os.path.join(
            "uploads",
            uploaded_file.name
        )

        # Spremi datoteku lokalno
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("Datoteka uspješno učitana.")

        # Pretvori TXT u CSV
        if st.button("🚀 Pokreni analizu", use_container_width=True):

            with st.spinner("Pretvaram TXT u CSV..."):

                txt_path = Path(file_path)
                out_csv = Path("outputs") / txt_path.with_suffix(".csv").name

                # mapa sudionika
                speaker_map_path = Path(".wa_state") / "speaker_map.json"
                speaker_map = load_json(speaker_map_path, default={})

                # glavni parser
                ok, msg = write_one_csv(
                    txt_path=txt_path,
                    out_csv_path=out_csv,
                    speaker_map=speaker_map,
                    keep_system=False
                )

                save_json(
                    speaker_map_path,
                    speaker_map
                )

                if ok:
                    st.session_state.csv_path = str(out_csv)
                    st.session_state.step = "results"
                    st.rerun()
                else:
                    st.error(msg)

# ==================================================
# STEP 2
# GLAVNI IZBORNIK
# ==================================================
elif st.session_state.step == "results":

    st.success("CSV uspješno kreiran.")
    st.subheader("2. Odaberi opciju")

    # 4 jednaka gumba u istom redu
    col1, col2, col3, col4 = st.columns(4, gap="medium")

    # -------------------------
    # CSV pregled
    # -------------------------
    with col1:
        if st.button("📄 CSV", use_container_width=True):
            show_dataframe(st.session_state.csv_path)

    # -------------------------
    # GoEmotions
    # -------------------------
    with col2:
        if st.button("😊 GoEmotions", use_container_width=True):
            st.session_state.step = "goemotions"
            st.rerun()

    # -------------------------
    # BERT
    # -------------------------
    with col3:
        if st.button("🤖 BERT", use_container_width=True):

            with st.spinner("Pokrećem BERT analizu..."):

                output_path = "outputs/bert_result.csv"

                run_bert_analysis(
                    input_csv=st.session_state.csv_path,
                    output_csv=output_path
                )

                st.session_state.bert_result = output_path
                st.session_state.step = "bert_done"
                st.rerun()

    # -------------------------
    # Usporedba modela
    # -------------------------
    with col4:
        if st.button("⚖️ Usporedba", use_container_width=True):

            with st.spinner("Pokrećem oba modela..."):

                compare_df, agreement = run_model_comparison(
                    input_csv=st.session_state.csv_path
                )

                st.session_state.compare_df = compare_df
                st.session_state.agreement = agreement
                st.session_state.step = "compare"
                st.rerun()

    st.divider()

    if st.button("⬅ Novi razgovor", use_container_width=True):
        reset_app()
        st.rerun()

# ==================================================
# STEP 3
# GOEMOTIONS IZBORNIK
# ==================================================
elif st.session_state.step == "goemotions":

    st.subheader("😊 GoEmotions analiza")
    st.write("Odaberi način rada modela:")

    col1, col2 = st.columns(2)

    # Originalni 28-label model
    with col1:
        if st.button("😊 Original GoEmotions", use_container_width=True):

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

    # Ekman 6 emocija
    with col2:
        if st.button("🎭 Ekman model", use_container_width=True):

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

    if st.button("⬅ Natrag", use_container_width=True):
        st.session_state.step = "results"
        st.rerun()

# ==================================================
# STEP 4
# GOEMOTIONS REZULTAT
# ==================================================
elif st.session_state.step == "go_done":

    st.success("GoEmotions analiza završena.")
    st.write(st.session_state.go_result)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📄 CSV rezultat", use_container_width=True):
            show_dataframe(st.session_state.go_result)

    with col2:
        if st.button("📊 Vizualizacije", use_container_width=True):
            st.session_state.visual_csv = st.session_state.go_result
            st.session_state.visual_title = "GoEmotions vizualizacija"
            st.session_state.step = "visuals"
            st.rerun()

    st.divider()

    if st.button("⬅ Povratak", use_container_width=True):
        st.session_state.step = "results"
        st.rerun()

# ==================================================
# STEP 5
# BERT REZULTAT
# ==================================================
elif st.session_state.step == "bert_done":

    st.success("BERT analiza završena.")
    st.write(st.session_state.bert_result)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📄 CSV rezultat", use_container_width=True):
            show_dataframe(st.session_state.bert_result)

    with col2:
        if st.button("📊 Vizualizacije", use_container_width=True):
            st.session_state.visual_csv = st.session_state.bert_result
            st.session_state.visual_title = "BERT vizualizacija"
            st.session_state.step = "visuals"
            st.rerun()

    st.divider()

    if st.button("⬅ Povratak", use_container_width=True):
        st.session_state.step = "results"
        st.rerun()

# ==================================================
# STEP 6
# VIZUALIZACIJE
# ==================================================
elif st.session_state.step == "visuals":

    st.subheader(f"📊 {st.session_state.visual_title}")

    df = load_result_csv(st.session_state.visual_csv)

    # -------------------------
    # Osnovne metrike
    # -------------------------
    c1, c2 = st.columns(2)

    with c1:
        st.metric("Broj poruka", len(df))

    with c2:
        if "speaker" in df.columns:
            st.metric(
                "Broj sudionika",
                df["speaker"].nunique()
            )

    st.divider()

    # Pie
    st.markdown("### 🥧 Distribucija emocija")
    st.plotly_chart(
        create_pie_chart(df),
        use_container_width=True
    )

    st.divider()

    # Bar
    st.markdown("### 📊 Broj poruka po emociji")
    st.plotly_chart(
        create_bar_chart(df),
        use_container_width=True
    )

    st.divider()

    # Timeline
    st.markdown("### 📈 Emocionalni tok razgovora")
    st.plotly_chart(
        create_timeline_chart(df),
        use_container_width=True
    )

    st.divider()

    # VAD
    st.markdown("### 🌌 VAD emocionalni prostor")
    st.plotly_chart(
        create_vad_3d_chart(df),
        use_container_width=True
    )

    st.divider()

    # Sankey
    st.markdown("### 🔀 Sankey dijagram emocija")
    st.plotly_chart(
        create_emotion_sankey(df),
        use_container_width=True
    )

    st.divider()

    # Influence
    st.markdown("### 🎯 Utjecaj sudionika")
    st.plotly_chart(
        create_influence_timeline(df),
        use_container_width=True
    )

    st.divider()

    if st.button("⬅ Povratak", use_container_width=True):
        st.session_state.step = "results"
        st.rerun()

# ==================================================
# STEP 7
# USPOREDBA MODELA (UPGRADED)
# ==================================================
elif st.session_state.step == "compare":

    st.subheader("⚖️ Usporedba modela")

    compare_df = st.session_state.compare_df
    agreement = st.session_state.agreement

    summary = get_summary(compare_df)

    # ==================================================
    # UČITAJ REZULTATE OBA MODELA
    # ==================================================
    # Pretpostavka:
    # comparison_module sprema rezultate ovdje
    # ako koristiš druge nazive samo promijeni putanje
    go_path = "outputs/go_compare.csv"
    bert_path = "outputs/bert_compare.csv"

    go_df = None
    bert_df = None

    if os.path.exists(go_path):
        go_df = load_result_csv(go_path)

    if os.path.exists(bert_path):
        bert_df = load_result_csv(bert_path)

    # ==================================================
    # METRIKE
    # ==================================================
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Ukupno poruka", summary["total"])

    with c2:
        st.metric("Podudaranja", summary["matched"])

    with c3:
        st.metric("Razlike", summary["mismatched"])

    with c4:
        st.metric("Agreement", f"{agreement}%")

    st.divider()


    # ==================================================
    # TABLICA REZULTATA
    # ==================================================
    st.markdown("## 📄 Rezultati po poruci")

    st.dataframe(
        compare_df,
        use_container_width=True
    )

    st.divider()

    # ==================================================
    # CONFUSION MATRIX
    # ==================================================
    st.markdown("## 🔥 Confusion Matrix")

    st.dataframe(
        get_confusion_df(compare_df),
        use_container_width=True
    )

    st.divider()

    # ==================================================
    # NESLAGANJA
    # ==================================================
    st.markdown("## ⚠️ Primjeri neslaganja")

    st.dataframe(
        get_mismatches(compare_df),
        use_container_width=True
    )

    st.divider()

    # ==================================================
    # VIZUALNA USPOREDBA
    # ==================================================
    go_path = "outputs/comparison_goemotions.csv"
    bert_path = "outputs/comparison_bert.csv"

    try:
        go_df = load_result_csv(go_path)
        bert_df = load_result_csv(bert_path)

        # ==================================================
        # PIE ostaje side-by-side
        # ==================================================
        st.markdown("## 🥧 Distribucija emocija")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 😊 GoEmotions")
            st.plotly_chart(
                create_pie_chart(go_df),
                use_container_width=True
            )

        with col2:
            st.markdown("### 🤖 BERT")
            st.plotly_chart(
                create_pie_chart(bert_df),
                use_container_width=True
            )

        st.divider()

        # ==================================================
        # TIMELINE jedan ispod drugog
        # ==================================================
        st.markdown("## 📈 Emotional Timeline")

        st.markdown("### 😊 GoEmotions")
        st.plotly_chart(
            create_timeline_chart(go_df),
            use_container_width=True
        )

        st.markdown("### 🤖 BERT")
        st.plotly_chart(
            create_timeline_chart(bert_df),
            use_container_width=True
        )

        st.divider()

        # ==================================================
        # VAD 3D jedan ispod drugog
        # ==================================================
        st.markdown("## 🌐 VAD Emotional Space")

        st.markdown("### 😊 GoEmotions")
        st.plotly_chart(
            create_vad_3d_chart(go_df),
            use_container_width=True
        )

        st.markdown("### 🤖 BERT")
        st.plotly_chart(
            create_vad_3d_chart(bert_df),
            use_container_width=True
        )

        st.divider()

        # ==================================================
        # SANKEY jedan ispod drugog
        # ==================================================
        st.markdown("## 🔀 Tok emocija")

        st.markdown("### 😊 GoEmotions")
        st.plotly_chart(
            create_emotion_sankey(go_df),
            use_container_width=True
        )

        st.markdown("### 🤖 BERT")
        st.plotly_chart(
            create_emotion_sankey(bert_df),
            use_container_width=True
        )

        st.divider()

        # ==================================================
        # INFLUENCE jedan ispod drugog
        # ==================================================
        st.markdown("## 🔄 Utjecaj sudionika")

        st.markdown("### 😊 GoEmotions")
        st.plotly_chart(
            create_influence_timeline(go_df),
            use_container_width=True
        )

        st.markdown("### 🤖 BERT")
        st.plotly_chart(
            create_influence_timeline(bert_df),
            use_container_width=True
        )

    except Exception as e:
        st.warning("Vizualizacije nisu pronađene.")
        st.code(str(e))

    st.divider()

    # ==================================================
    # POVRATAK
    # ==================================================
    if st.button("⬅ Povratak", use_container_width=True):
        st.session_state.step = "results"
        st.rerun()

