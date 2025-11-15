import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import base64
from io import StringIO

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Disease Detection AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Custom CSS (pastel hospital theme)
# -------------------------
st.markdown(
    """
    <style>
    /* page bg */
    .stApp {
        background: #fffbe6;
        color: #1f2937;
    }

    /* central container card */
    .card {
        background: #ffffff;
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 8px 22px rgba(28, 78, 120, 0.06);
        border: 1px solid rgba(0,0,0,0.03);
        margin-bottom: 18px;
    }

    /* hero */
    .hero-title {
        color: #0f4c81;
        font-size: 38px;
        font-weight: 800;
        margin: 8px 0 0 0;
    }
    .hero-sub {
        color: #22577a;
        opacity: 0.85;
        margin-top: 6px;
        font-size: 16px;
    }

    /* success banner */
    .success-banner {
        background: #e9fff0;
        border: 1px solid #c8f3d7;
        color: #0b7a3f;
        padding: 12px 16px;
        border-radius: 10px;
        display: inline-block;
        margin-bottom: 10px;
        font-weight: 600;
    }

    /* styled table */
    table.styled-table {
        border-collapse: collapse;
        width: 100%;
        font-size: 14px;
        border-radius: 8px;
        overflow: hidden;
    }
    table.styled-table thead tr {
        background: #f1f6fb;
        color: #0f4c81;
        font-weight: 700;
        text-align: left;
    }
    table.styled-table th, table.styled-table td {
        padding: 10px 12px;
        border-bottom: 1px solid #eee;
    }
    table.styled-table tbody tr:hover {
        background: #fcffef;
    }

    /* download button */
    .download-btn {
        background: #e6faf0;
        border: 1px solid #c7f3d9;
        color: #0b7a3f;
        padding: 10px 16px;
        border-radius: 10px;
        font-weight: 700;
    }

    /* footer */
    .footer {
        color: #6b7280;
        font-size: 13px;
        text-align: center;
        margin-top: 18px;
    }

    /* small responsive tweaks */
    @media (max-width: 800px) {
        .hero-title { font-size: 28px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Header / Hero (centered)
# -------------------------
st.markdown(
    """
    <div style='text-align:center; margin-top:10px; margin-bottom:10px;'>
    """,
    unsafe_allow_html=True
)

st.image("hospital_banner.png", width=130)

st.markdown(
    """
        <div class='hero-title'>🩺 Disease Detection AI</div>
        <div class='hero-sub'>Upload patient data to predict disease risk using machine learning</div>
    </div>
    """,
    unsafe_allow_html=True
)



st.sidebar.markdown("## 📁 Upload Patient CSV")
uploaded_file = st.sidebar.file_uploader("Choose CSV file", type=["csv"])


st.sidebar.markdown("---")
st.sidebar.markdown("Model file (optional):")
model_uploader = st.sidebar.file_uploader("Upload Random_Forest_model.pkl (optional)", type=["pkl", "joblib"])



model = None
if model_uploader is not None:
    try:
        with open("Random_Forest_model.pkl", "wb") as f:
            f.write(model_uploader.getbuffer())
        model = joblib.load("Random_Forest_model.pkl")
    except Exception as e:
        st.sidebar.error(f"Failed to load uploaded model: {e}")

if model is None:
    try:
        model = joblib.load("Random_Forest_model.pkl")
    except Exception:
        
        pass

left_col, right_col = st.columns([1, 2])


with left_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### 🗂️ Upload & Controls")
    st.markdown(
        """
        - Upload data in the sidebar.
        - Supported columns: Glucose, BloodPressure, SkinThickness, Insulin, BMI, Outcome (optional).
        - Model should be present as `Random_Forest_model.pkl`.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

#
with right_col:
    
    if not uploaded_file:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.info("👈 Upload a CSV file from the sidebar to begin predictions.", icon="ℹ️")
        if model is None:
            st.warning("Model not found. Upload `Random_Forest_model.pkl` via the sidebar or place it in the app folder.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Read CSV
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        # Display success banner
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='success-banner'>✔ File uploaded successfully</div>", unsafe_allow_html=True)

        # Uploaded data card
        st.markdown("<h3 style='margin-top:6px; margin-bottom:6px; color:#0f4c81;'>📊 Uploaded Patient Data</h3>", unsafe_allow_html=True)

        # Small cleaning similar to original script
        cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        present_cols = [c for c in cols_with_zero if c in data.columns]
        if present_cols:
            data[present_cols] = data[present_cols].replace(0, np.nan)
            data.fillna(data.mean(numeric_only=True), inplace=True)

        # Render styled table via HTML
        try:
            # convert dataframe to html with classes
            table_html = data.to_html(classes="styled-table", index=False, na_rep="", justify="left")
            st.markdown(table_html, unsafe_allow_html=True)
        except Exception:
            st.dataframe(data)

        # Prediction logic (if model present)
        if model is None:
            st.error("❌ Model file not found. Please upload `Random_Forest_model.pkl` in the sidebar or put it in the app directory.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            # Prepare features
            X = data.drop("Outcome", axis=1, errors='ignore')
            # If there are non-numeric columns, try to drop them (or you can add encoding later)
            X = X.select_dtypes(include=[np.number])

            
            scaler = StandardScaler()
            try:
                X_scaled = scaler.fit_transform(X)
            except Exception as e:
                st.error(f"Failed to preprocess data for prediction: {e}")
                st.markdown("</div>", unsafe_allow_html=True)
                st.stop()

            # Predictions
            try:
                predictions = model.predict(X_scaled)
                probs = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(predictions))
            except Exception as e:
                st.error(f"Model prediction failed: {e}")
                st.markdown("</div>", unsafe_allow_html=True)
                st.stop()

            # Build results dataframe
            result_df = pd.DataFrame({
                "Prediction": ["Positive" if p == 1 else "Negative" for p in predictions],
                "Probability (%)": np.round(probs * 100, 2)
            })

            # Prediction results card
            st.markdown("<hr style='margin:12px 0 14px 0;'>", unsafe_allow_html=True)
            st.markdown("<h3 style='color:#0f4c81;'>🔍 Prediction Results</h3>", unsafe_allow_html=True)

            
            st.markdown("<div class='success-banner'>✔ Prediction complete</div>", unsafe_allow_html=True)

            # Render results as html styled table
            try:
                res_html = result_df.to_html(classes="styled-table", index=False, justify="left")
                st.markdown(res_html, unsafe_allow_html=True)
            except Exception:
                st.dataframe(result_df)

            # Download button (csv)
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name="prediction_results.csv",
                mime="text/csv"
            )

            st.markdown("</div>", unsafe_allow_html=True)  # close card

# -------------------------
# Footer
# -------------------------
st.markdown(
    """
    <div class='footer'>
        Made with Streamlit & Machine Learning
    </div>
    """,
    unsafe_allow_html=True,
)
