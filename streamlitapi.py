import pickle
import streamlit as st
import datetime
import os
import requests

MODEL_URL = "https://storage.googleapis.com/sales_prediction_bucket/model.pkl"
MODEL_PATH = "model.pkl"

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Rossmann Sales Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODEL LOADING ---
def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        try:
            st.info("Downloading model file from cloud storage...")
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            st.success("Model file downloaded successfully.")
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            return False
    return True

@st.cache_data
def load_model(): 
    if not download_model_if_needed():
        return None
    try:
        return pickle.load(open(MODEL_PATH, 'rb'))
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

def predict_sales(features):
    """
    Predict sales based on input features.
    """
    if model is None:
        st.error("Model not available. Please check model download or upload.")
        return None
    prediction = model.predict([features])
    return prediction[0]

# --- PAGE CONTENTS ---
def page_home():
    st.title("Rossmann Sales Prediction")
    st.markdown("""
    <span style='font-size:18px;'>Predict daily sales for Rossmann stores using a machine learning model. Adjust the inputs to see how different factors affect sales!</span>
    """, unsafe_allow_html=True)
    st.markdown("---")

    if model is None:
        st.warning("⚠️ Model not loaded. Please check the model file or download link.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Store Information")
        day_of_week = st.selectbox("Day of Week", 
                                  options=[(1, "Monday"), (2, "Tuesday"), (3, "Wednesday"), 
                                          (4, "Thursday"), (5, "Friday"), (6, "Saturday"), (7, "Sunday")],
                                  format_func=lambda x: x[1], index=0)
        customers = st.number_input("Number of Customers", min_value=0, value=0, 
                                  help="This has the biggest impact on sales prediction!")
        storetype = st.selectbox("Store Type", options=["a", "b", "c", "d"], index=0)
        assortment = st.selectbox("Assortment", options=["a", "b", "c"], index=0)
    with col2:
        st.subheader("Promotional Information")
        promo = st.slider("Promo (%)", min_value=0, max_value=100, value=0)
        promo2 = st.checkbox("Promo2", value=False)
        school_holiday = st.checkbox("School Holiday", value=False)
        competition_distance = st.number_input("Competition Distance (km)", min_value=0, value=0,
                                             help="Distance to nearest competitor - affects sales significantly!")
        month = st.number_input("Month", min_value=1, max_value=12, value=datetime.date.today().month)
        day = st.number_input("Day", min_value=1, max_value=31, value=datetime.date.today().day)

    # Build features list
    features = []
    features.append(day_of_week[0])  # Day of week
    features.append(customers)        # Customers
    features.append(promo / 100)      # Promo
    features.append(1 if school_holiday else 0)  # School holiday
    features.append(competition_distance)  # Competition distance
    features.append(1 if promo2 else 0)    # Promo2
    features.append(month)            # Month
    features.append(day)              # Day
    # Store type one-hot encoding
    features.append(1 if storetype == "b" else 0)
    features.append(1 if storetype == "c" else 0)
    features.append(1 if storetype == "d" else 0)
    # Assortment one-hot encoding
    features.append(1 if assortment == "b" else 0)
    features.append(1 if assortment == "c" else 0)

    st.markdown("---")
    st.subheader("Sales Prediction")
    if st.button("Predict Sales", type="primary"):
        result = predict_sales(features)
        if result is not None:
            st.success(f"**Predicted Sales: ${result:,.2f}**")
            st.markdown("### What's affecting this prediction:")
            base_result = result
            # Test with more customers
            if customers < 1000:
                test_features = features.copy()
                test_features[1] = 1000
                test_result = predict_sales(test_features)
                if test_result is not None and abs(test_result - base_result) > 1:
                    st.write(f"• **More customers (1000)**: ${test_result:,.2f} (${test_result - base_result:+,.2f})")
            # Test with different competition distance
            if competition_distance < 1000:
                test_features = features.copy()
                test_features[4] = 1000
                test_result = predict_sales(test_features)
                if test_result is not None and abs(test_result - base_result) > 1:
                    st.write(f"• **More competition (1000km)**: ${test_result:,.2f} (${test_result - base_result:+,.2f})")
            # Test with promo
            if promo == 0:
                test_features = features.copy()
                test_features[2] = 0.5
                test_result = predict_sales(test_features)
                if test_result is not None and abs(test_result - base_result) > 1:
                    st.write(f"• **With 50% promo**: ${test_result:,.2f} (${test_result - base_result:+,.2f})")
    if st.button("Clear All"):
        st.session_state.clear()
        st.success("All inputs cleared!")
        st.rerun()
    with st.expander("Debug Information"):
        st.write("Feature values:", features)
        st.write("Feature names:", ["DayOfWeek", "Customers", "Promo", "SchoolHoliday", 
                                   "CompetitionDistance", "Promo2", "Month", "Day", 
                                   "StoreType_b", "StoreType_c", "StoreType_d", 
                                   "Assortment_b", "Assortment_c"])

def page_about():
    st.title("About This Project")
    st.markdown("""
    ### Rossmann Sales Prediction
    This web application predicts daily sales for Rossmann stores using a machine learning model trained on historical data. The project demonstrates the power of data-driven decision making in retail, allowing users to experiment with different store and promotional scenarios to see their impact on sales.

    **Author:** Abdullahy Bashir  
    **Technologies:** Python, Streamlit, scikit-learn, Machine Learning

    #### About the Author
    Abdullahy Bashir is a passionate data scientist and machine learning engineer with a keen interest in building impactful solutions for real-world business problems. With a strong background in predictive modeling and analytics, Abdullahy enjoys sharing knowledge through technical writing and open-source projects.
    """)

def page_analysis():
    st.title("Analysis & Technical Walkthrough")
    st.markdown("""
    ### End-to-End Sales Prediction for Rossmann Stores
    Dive deep into the technical details, data exploration, feature engineering, and modeling process behind this app in the following Hashnode article:
    """)
    st.markdown("""
    <div style='margin: 20px 0;'>
        <a href="https://abdullahybashir.hashnode.dev/end-to-end-sales-prediction-for-rossmann-stores-a-detailed-technical-walkthrough" target="_blank" style="font-size:18px; color:#2563eb; font-weight:bold;">Read the full article on Hashnode</a>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <iframe src="https://abdullahybashir.hashnode.dev/end-to-end-sales-prediction-for-rossmann-stores-a-detailed-technical-walkthrough" height="600" width="100%" style="border:none;"></iframe>
    """, unsafe_allow_html=True)
    st.info("If the article does not display above, please use the link to open it in a new tab.")

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["Home", "About", "Analysis"], index=0)
    st.sidebar.markdown("---")
    st.sidebar.write("Built by Abdullahy Bashir")
    if page == "Home":
        page_home()
    elif page == "About":
        page_about()
    elif page == "Analysis":
        page_analysis()

if __name__ == "__main__":
    main()