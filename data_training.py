# data_training.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

DATA_PATH = "data/restaurant_orders.csv"

def load_and_preprocess_data(csv_path=DATA_PATH):
    df = pd.read_csv(csv_path)

    df['order_date'] = pd.to_datetime(df['order_date'])
    df['order_month'] = df['order_date'].dt.month
    df['order_day'] = df['order_date'].dt.day
    df['order_dayofweek'] = df['order_date'].dt.dayofweek
    df.drop('order_date', axis=1, inplace=True)

    categorical_cols = ['cuisine', 'meal_type', 'payment_method', 'country']
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    X = df.drop(['order_id', 'status'], axis=1)
    y = df['status'].astype(str)

    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save preprocessed objects
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(label_encoders, "models/label_encoders.pkl")
    joblib.dump(le_target, "models/target_encoder.pkl")
    joblib.dump(list(X_train.columns), "models/feature_columns.pkl")

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, le_target, scaler, label_encoders

def train_models():
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, le_target, scaler, label_encoders = load_and_preprocess_data()

    # K-NN uses scaled features
    print("Training K-NN...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    knn_pred = knn.predict(X_test_scaled)
    knn_accuracy = accuracy_score(y_test, knn_pred)
    print(f"K-NN Accuracy: {knn_accuracy:.4f}")

    # Random Forest using unscaled features (tree-based)
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

    # XGBoost using unscaled features
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=6,
        use_label_encoder=False, eval_metric='mlogloss', random_state=42
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")

    # Save models
    joblib.dump(knn, "models/knn_model.pkl")
    joblib.dump(rf, "models/random_forest_model.pkl")
    joblib.dump(xgb_model, "models/xgboost_model.pkl")

    # Reports
    reports = {
        'K-NN': classification_report(y_test, knn_pred, target_names=le_target.classes_, output_dict=True),
        'Random Forest': classification_report(y_test, rf_pred, target_names=le_target.classes_, output_dict=True),
        'XGBoost': classification_report(y_test, xgb_pred, target_names=le_target.classes_, output_dict=True)
    }
    joblib.dump(reports, "models/classification_reports.pkl")

    print("\nSaved models and preprocessing objects to models/")

    return knn, rf, xgb_model, reports

if __name__ == "__main__":
    # If dataset not present, generate it first
    if not os.path.exists(DATA_PATH):
        from data_generation import generate_restaurant_data
        generate_restaurant_data(10000, DATA_PATH)

    train_models()
