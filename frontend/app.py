import streamlit as st
import requests
import time

st.markdown("""
<style>

/* FULLSCREEN OVERLAY */
#overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;

    background: rgba(0,0,0,0.55);
    backdrop-filter: blur(10px);

    display: flex;
    align-items: center;
    justify-content: center;

    z-index: 999999;
    flex-direction: column;

    color: white;
    font-size: 22px;

    opacity: 1;
    transition: opacity 0.6s ease;
}

/* fade out */
#overlay.fade-out {
    opacity: 0;
    pointer-events: none;
}

/* spinner */
.spinner {
    width: 70px;
    height: 70px;
    border: 6px solid rgba(255,255,255,0.2);
    border-top: 6px solid #00ffd5;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-bottom: 20px;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* glow card */
.glow-card {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    padding: 30px;
    border-radius: 20px;
    color: white;
    text-align: center;
    animation: glow 2s infinite ease-in-out;
    margin-top: 20px;
}

@keyframes glow {
    0% { box-shadow: 0 0 10px rgba(0,255,200,0.4); }
    50% { box-shadow: 0 0 25px rgba(0,150,255,0.8); }
    100% { box-shadow: 0 0 10px rgba(0,255,200,0.4); }
}

.price {
    font-size: 52px;
    font-weight: bold;
    margin-top: 10px;
    color: #00ffd5;
}

.subtitle {
    opacity: 0.8;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)

# Desired model selection
option_map = {
    0: "Random Forest",
    1: "Gradient Boosting",
    2: "XGBoost"
}
model_options = list(option_map.values())


# ======================
# INPUT UI
# ======================
col1, col2 = st.columns(2)

with col1:
    year_built = st.number_input("Year Built", 1900, 2026, 2000)
    bedroom_abvgr = st.slider("Bedrooms", 0, 10, 3)
    garage_area = st.number_input("Garage Area", 0, 2000, 500)
    overall_qual = st.slider("Overall Quality", 1, 10, 5)

with col2:
    gr_liv_area = st.number_input("Ground Living Area (sq ft)", 300, 10000, 1500)
    garage_cars = st.slider("Garage Capacity", 0, 5, 2)
    total_bsmt_sf = st.number_input("Basement Area", 0, 5000, 800)
    full_bath = st.slider("Full Bathrooms", 0, 5, 2)

# ======================
# SIDEBAR
# ======================
st.sidebar.title("About")
st.sidebar.markdown("This app predicts house prices using Machine Learning.")

st.sidebar.subheader("Models tested")
st.sidebar.markdown(
    """
    | Model | RMSE |
    |----|----|
    | Random Forest | 0.080 |
    | Gradient Boosting | 0.147 |
    | XGBoost | 0.114 |
    """
)

selection = st.sidebar.selectbox(
    "Select Regression Model",
    options=model_options,
    index=2,
)
st.write(
    "Your selected option: "
    f"{selection}"
)

st.sidebar.subheader("Built with:")
st.sidebar.info("""
- Streamlit
- FastAPI
- Scikit-learn
- MLflow
""")

# ======================
# OUTPUT PLACEHOLDERS
# ======================
overlay = st.empty()
result = st.empty()
feedback = st.empty()

# ======================
# PREDICTION
# ======================

st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)

if st.button("Predict Price", use_container_width=True):

    model_selection = selection

    payload = {
        "Model": model_selection,
        "OverallQual": overall_qual,
        "GrLivArea": gr_liv_area,
        "GarageCars": garage_cars,
        "TotalBsmtSF": total_bsmt_sf,
        "FullBath": full_bath,
        "YearBuilt": year_built,
        "BedroomAbvGr": bedroom_abvgr,
        "GarageArea": garage_area,
        "HouseAge": 2010 - year_built,
        "TotalArea": gr_liv_area + garage_area
    }

    # FULLSCREEN BLUR + SPINNER
    overlay.markdown("""
    <div id="overlay">
        <div class="spinner"></div>
        <div style="margin-top:20px;">
            AI is estimating property value...
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload,
            timeout=10
        )

        overlay.empty()

        # SUCCESS
        if response.status_code == 200:

            prediction = response.json()["predicted_price"]

            result.markdown(
                f"""
                <div class="glow-card">
                    <div class="subtitle">🏠 Predicted House Value</div>
                    <div class="price">${prediction:,.0f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.toast("Prediction completed successfully")

        # API ERROR
        else:
            st.error(
                f"API Error ({response.status_code})"
            )

    # API OFFLINE
    except requests.exceptions.ConnectionError:

        overlay.empty()

        st.error("""
        Cannot connect to prediction API.

        Make sure FastAPI server is running:
        
        python3 -m uvicorn src.api.main:app --reload
        """)

    # TIMEOUT
    except requests.exceptions.Timeout:

        overlay.empty()

        st.error("""
        Request timed out.
        
        The model took too long to respond.
        """)

    # OTHER ERRORS
    except Exception as e:

        overlay.empty()

        st.error(f"Unexpected error: {e}")