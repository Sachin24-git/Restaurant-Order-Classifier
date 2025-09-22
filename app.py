# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Restaurant Order Classifier", page_icon="🍕", layout="wide")

FEATURE_ORDER = [
    "order_time",
    "cuisine",
    "meal_type",
    "payment_method",
    "country",
    "order_value",
    "order_month",
    "order_day",
    "order_dayofweek"
]

MODELS_DIR = "models"
DATA_PATH = "data/restaurant_orders.csv"

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

@st.cache_resource
def load_models():
    try:
        knn = joblib.load(os.path.join(MODELS_DIR, 'knn_model.pkl'))
        rf = joblib.load(os.path.join(MODELS_DIR, 'random_forest_model.pkl'))
        xgb_model = joblib.load(os.path.join(MODELS_DIR, 'xgboost_model.pkl'))
        scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
        label_encoders = joblib.load(os.path.join(MODELS_DIR, 'label_encoders.pkl'))
        le_target = joblib.load(os.path.join(MODELS_DIR, 'target_encoder.pkl'))
        feature_columns = joblib.load(os.path.join(MODELS_DIR, 'feature_columns.pkl'))
        reports = None
        try:
            reports = joblib.load(os.path.join(MODELS_DIR, 'classification_reports.pkl'))
        except:
            reports = None
        return knn, rf, xgb_model, scaler, label_encoders, le_target, feature_columns, reports
    except FileNotFoundError:
        st.error("Models or preprocessing objects not found. Run data_generation.py and data_training.py first.")
        st.stop()

knn, rf, xgb_model, scaler, label_encoders, le_target, feature_columns, reports = load_models()

def preprocess_input(inputs, label_encoders, feature_columns):
    processed = {}
    for feature in FEATURE_ORDER:
        if feature in label_encoders:
            le = label_encoders[feature]
            val = inputs[feature]
            if val not in le.classes_:
                val = le.classes_[0]
            processed[feature] = le.transform([val])[0]
        else:
            processed[feature] = inputs[feature]

    # DataFrame with correct columns
    df_input = pd.DataFrame([processed])
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)
    return df_input

def predict_order(features_df):
    # KNN needs scaled
    features_scaled = scaler.transform(features_df)
    knn_pred = knn.predict(features_scaled)

    # RF & XGB on raw
    rf_pred = rf.predict(features_df)
    xgb_pred = xgb_model.predict(features_df)

    return {
        "K-NN": le_target.inverse_transform(knn_pred)[0],
        "Random Forest": le_target.inverse_transform(rf_pred)[0],
        "XGBoost": le_target.inverse_transform(xgb_pred)[0]
    }

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔮 Order Prediction", "📊 Model Performance", "📂 Data Overview"])

if page == "🏠 Home":
    st.title("🍕 Restaurant Order Classification App")
    st.markdown("""
    ### Welcome!
    - Predict order status using ML models
    - Compare models (K-NN, Random Forest, XGBoost)
    - Explore dataset & model performance
    """)

elif page == "🔮 Order Prediction":
    st.title("🔮 Order Status Prediction")

    cuisine = st.selectbox("Cuisine", label_encoders['cuisine'].classes_)
    meal_type = st.selectbox("Meal Type", label_encoders['meal_type'].classes_)
    payment_method = st.selectbox("Payment Method", label_encoders['payment_method'].classes_)
    country = st.selectbox("Country", label_encoders['country'].classes_)

    order_value = st.slider("Order Value", 5, 500, 50)
    order_time = st.slider("Order Hour", 0, 23, 12)
    date = st.date_input("Order Date")
    order_month = date.month
    order_day = date.day
    order_dayofweek = date.weekday()

    if st.button("Predict"):
        inputs = {
            "order_time": order_time,
            "cuisine": cuisine,
            "meal_type": meal_type,
            "payment_method": payment_method,
            "country": country,
            "order_value": order_value,
            "order_month": order_month,
            "order_day": order_day,
            "order_dayofweek": order_dayofweek
        }
        features_df = preprocess_input(inputs, label_encoders, feature_columns)
        predictions = predict_order(features_df)

        st.subheader("Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("K-NN", predictions["K-NN"])
        col2.metric("Random Forest", predictions["Random Forest"])
        col3.metric("XGBoost", predictions["XGBoost"])

elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    st.markdown("### Accuracy Comparison")

    if reports is not None:
        accs = {}
        for name, rpt in reports.items():
            accs[name] = rpt.get('accuracy', 0.0)
        df_acc = pd.DataFrame({
            "Model": list(accs.keys()),
            "Accuracy": list(accs.values())
        })
    else:
        df_acc = pd.DataFrame({
            "Model": ["K-NN", "Random Forest", "XGBoost"],
            "Accuracy": [0.0, 0.0, 0.0]
        })

    st.bar_chart(df_acc.set_index("Model"))

elif page == "📂 Data Overview":
    st.title("📂 Data Overview")
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        st.dataframe(df.head())

        st.markdown("### Cuisine Distribution")
        st.bar_chart(df['cuisine'].value_counts())

        st.markdown("### Order Value Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['order_value'], bins=20, ax=ax, kde=True)
        st.pyplot(fig)

        st.markdown("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.error("Dataset not found. Run data_generation.py first.")
