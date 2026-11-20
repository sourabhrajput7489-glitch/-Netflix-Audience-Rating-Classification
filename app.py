
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Netflix Audience Rating Predictor",
    page_icon="🎬",
    layout="wide"
)

# Load trained model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "netflix_rating_model.pkl"
DATA_PATH = BASE_DIR / "data" / "Dataset.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()

# Header
st.title("🎬 Netflix Audience Rating Classification")
st.markdown(
    "Predict the audience rating of Netflix Movies and TV Shows "
    "using a Machine Learning classification model."
)

st.divider()

# Sidebar
st.sidebar.header("📊 Model Information")
st.sidebar.write("Algorithm: Logistic Regression")
st.sidebar.metric("Test Accuracy", "53.58%")
st.sidebar.write("Dataset: 8,790 Netflix titles")
st.sidebar.info(
    "Predictions are estimates based on content metadata, "
    "not official Netflix age ratings."
)

# Input section
st.subheader("Enter Netflix Content Details")

col1, col2 = st.columns(2)

with col1:
    content_type = st.selectbox(
        "Content Type",
        ["Movie", "TV Show"]
    )

    release_year = st.number_input(
        "Release Year",
        min_value=1900,
        max_value=2030,
        value=2022
    )

    countries = sorted(
        df["country"].dropna().unique().tolist()
    )

    country = st.selectbox(
        "Country",
        countries
    )

with col2:
    genre = st.selectbox(
        "Genre / Listed In",
        sorted(df["listed_in"].dropna().unique().tolist())
    )

    duration = st.text_input(
        "Duration",
        value="120 min" if content_type == "Movie" else "2 Seasons"
    )

    director = st.text_input(
        "Director",
        value="Unknown"
    )

st.divider()

# Prediction
if st.button("🔮 Predict Audience Rating", type="primary"):

    input_data = pd.DataFrame({
        "type": [content_type],
        "release_year": [release_year],
        "country": [country],
        "listed_in": [genre],
        "duration": [duration],
        "director": [director]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Audience Rating: {prediction}")

    st.subheader("Prediction Details")

    st.dataframe(input_data, use_container_width=True)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        classes = model.classes_

        probability_df = pd.DataFrame({
            "Rating": classes,
            "Probability": probabilities
        }).sort_values("Probability", ascending=False)

        st.subheader("Prediction Probability by Rating")
        st.bar_chart(
            probability_df.set_index("Rating")
        )

        confidence = probability_df.iloc[0]["Probability"]
        st.metric(
            "Top Predicted Probability",
            f"{confidence * 100:.2f}%"
        )

st.divider()
st.caption(
    "Netflix Audience Rating Classification | "
    "Machine Learning Internship Project"
)
