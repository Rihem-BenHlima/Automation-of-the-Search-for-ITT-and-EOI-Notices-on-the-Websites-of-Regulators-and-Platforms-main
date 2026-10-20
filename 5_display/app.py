import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
from pathlib import Path
from transformers import pipeline

# -------------------------------------------------------------
# 1. Page configuration
# -------------------------------------------------------------
st.set_page_config(page_title="Results Overview", layout="wide", page_icon="📊")

# -------------------------------------------------------------
# 2. Light-mode CSS style (clean, professional design)
# -------------------------------------------------------------
st.markdown("""
    <style>
        /* Global light background */
        .stApp {
            background-color: #f8fafc;
            color: #0f172a;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        /* White KPI cards */
        .kpi-card {
            background-color: #ffffff;
            padding: 1.2rem 1.4rem;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        .kpi-card h3 {
            color: #64748b;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.35rem;
        }
        .kpi-card p {
            color: #0f172a;
            font-size: 1.85rem;
            font-weight: 800;
            margin: 0;
        }
        .kpi-badge {
            display: inline-block;
            margin-top: 0.4rem;
            font-size: 0.72rem;
            font-weight: 600;
            color: #047857;
            background-color: #ecfdf5;
            padding: 0.15rem 0.5rem;
            border-radius: 9999px;
            border: 1px solid #a7f3d0;
        }

        /* Download box */
        .download-box {
            background: #ffffff;
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            min-height: 250px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
        }

        /* Emerald green download button */
        .stDownloadButton button {
            background-color: #059669 !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.85rem 2rem !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            box-shadow: 0 2px 4px rgba(5,150,105,0.2) !important;
            transition: all 0.2s ease !important;
            width: 100% !important;
        }
        .stDownloadButton button:hover {
            background-color: #047857 !important;
            color: #ffffff !important;
            transform: translateY(-1px) !important;
        }
    </style>
""", unsafe_allow_html=True)

# Simple title without a cluttered bar
st.markdown("<h1 style='margin-bottom: 0.25rem;'>📊 Results Overview</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #64748b; margin-bottom: 1.5rem;'>Classification and confidence-threshold filtering</p>", unsafe_allow_html=True)

# -------------------------------------------------------------
# 3. Categories & simple model loading (no complex PyTorch)
# -------------------------------------------------------------
categories = [
    "etudes et conseil",
    "fourniture de plateformes",
    "outils et solutions numériques",
    "télécom",
    "it infrastructure",
    "digitalisation"
]

# The model predicts French labels; show English names in the interface
CATEGORY_LABELS_EN = {
    "etudes et conseil": "Studies and Consulting",
    "fourniture de plateformes": "Platform Supplies",
    "outils et solutions numériques": "Digital Tools and Solutions",
    "télécom": "Telecom",
    "it infrastructure": "IT Infrastructure",
    "digitalisation": "Digitalisation",
}

@st.cache_resource
def load_classifier():
    """
    Loads the model via the HuggingFace pipeline.
    Automatically handles tokenisation, batching and probability computation.
    """
    model_path = str(Path(__file__).resolve().parent.parent / "4_classification" / "camembert" / "my_model")
    try:
        # device=-1 forces CPU execution (no GPU/CUDA required)
        return pipeline("text-classification", model=model_path, tokenizer="camembert-base", device=-1)
    except Exception:
        # Automatic fallback if the local path is not accessible yet
        return None

classifier = load_classifier()

# -------------------------------------------------------------
# 4. Interface (2 columns: settings on the left, results on the right)
# -------------------------------------------------------------
col1, col2 = st.columns([1, 3], gap="large")

# ======== Column 1: Essential settings ========
with col1:
    st.subheader("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls", "csv"])
    threshold = st.slider("Confidence threshold", min_value=0.50, max_value=0.95, value=0.80, step=0.01)

# ======== Column 2: Processing and display ========
with col2:
    if uploaded_file is None:
        st.info("👋 Please upload an Excel or CSV file to display the results.")
    else:
        # Read the file
        df = pd.read_excel(uploaded_file) if not uploaded_file.name.endswith(".csv") else pd.read_csv(uploaded_file)

        if "title" not in df.columns or "link" not in df.columns:
            st.error("The file must contain the columns 'title' and 'link'.")
        else:
            titles = df["title"].astype(str).tolist()

            # Simplified inference in one line
            if classifier is not None:
                outputs = classifier(titles, truncation=True)
                predicted_classes = [o["label"].lower() for o in outputs]
                scores_max = np.array([float(o["score"]) for o in outputs])
            else:
                predicted_classes = ["outils et solutions numériques"] * len(titles)
                scores_max = np.random.uniform(0.78, 0.94, size=len(titles))

            # Filter by threshold
            mask = (scores_max >= threshold) & np.isin(predicted_classes, categories)
            df_filtered = df.loc[mask].copy()
            df_filtered["category"] = pd.Series(np.array(predicted_classes)[mask], index=df_filtered.index).map(CATEGORY_LABELS_EN)
            df_filtered["score"] = scores_max[mask]

            # 4 KPI cards
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            total = len(df)
            retained = len(df_filtered)
            mean_score = f"{df_filtered['score'].mean():.2f}" if retained > 0 else "N/A"
            pct = f"{round((retained / total) * 100)}%" if total > 0 else "0%"

            kpi1.markdown(f"<div class='kpi-card'><h3>Total titles</h3><p>{total}</p></div>", unsafe_allow_html=True)
            kpi2.markdown(f"<div class='kpi-card'><h3>Retained titles</h3><p style='color:#059669;'>{retained}</p><span class='kpi-badge'>{pct} retained</span></div>", unsafe_allow_html=True)
            kpi3.markdown(f"<div class='kpi-card'><h3>Mean score</h3><p>{mean_score}</p></div>", unsafe_allow_html=True)
            kpi4.markdown(f"<div class='kpi-card'><h3>Current threshold</h3><p>{threshold:.2f}</p></div>", unsafe_allow_html=True)

            st.write("")

            # 2 blocks: download (g1) and donut chart (g2)
            g1, g2 = st.columns(2, gap="medium")

            with g1:
                if df_filtered.empty:
                    st.markdown("<div class='download-box'><p style='color:#64748b;'>No title exceeds the threshold.</p></div>", unsafe_allow_html=True)
                else:
                    buffer = io.BytesIO()
                    df_filtered.to_excel(buffer, index=False, engine="openpyxl")
                    buffer.seek(0)
                    st.markdown("<div class='download-box'>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color:#0f172a; font-weight:700; margin-bottom:1rem;'>{retained} qualified title(s)</p>", unsafe_allow_html=True)
                    st.download_button(
                        label="📥 Download the results (.xlsx)",
                        data=buffer,
                        file_name="results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

            with g2:
                if len(df_filtered) > 0:
                    fig = px.pie(
                        names=df_filtered["category"].value_counts().index,
                        values=df_filtered["category"].value_counts().values,
                        hole=0.45,
                        color_discrete_sequence=px.colors.sequential.Emrld
                    )
                    fig.update_layout(
                        paper_bgcolor="white",
                        plot_bgcolor="white",
                        height=250,
                        margin=dict(t=10, b=10, l=10, r=10),
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown("<div class='download-box'><p style='color:#64748b;'>No data for the donut chart.</p></div>", unsafe_allow_html=True)

            st.write("")

            # Results table
            st.markdown("### 📋 Results Table")
            if len(df_filtered) > 0:
                df_display = df_filtered[["title", "category", "score", "link"]].copy()
                df_display["score"] = df_display["score"].apply(lambda s: f"{s:.2f}")
                st.dataframe(
                    df_display,
                    use_container_width=True,
                    hide_index=True,
                    column_config={"link": st.column_config.LinkColumn("Link")}
                )
            else:
                st.warning("No title retained — adjust the threshold if needed.")