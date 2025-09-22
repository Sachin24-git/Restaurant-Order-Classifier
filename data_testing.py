# data_testing.py
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

def test_models(num_test_samples=2000):
    knn = joblib.load('models/knn_model.pkl')
    rf = joblib.load('models/random_forest_model.pkl')
    xgb_model = joblib.load('models/xgboost_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    label_encoders = joblib.load('models/label_encoders.pkl')
    le_target = joblib.load('models/target_encoder.pkl')

    # Generate fresh test data
    from data_generation import generate_restaurant_data
    test_df = generate_restaurant_data(num_test_samples, output_path="data/test_restaurant_orders.csv")

    test_df['order_date'] = pd.to_datetime(test_df['order_date'])
    test_df['order_month'] = test_df['order_date'].dt.month
    test_df['order_day'] = test_df['order_date'].dt.day
    test_df['order_dayofweek'] = test_df['order_date'].dt.dayofweek
    test_df.drop('order_date', axis=1, inplace=True)

    categorical_cols = ['cuisine', 'meal_type', 'payment_method', 'country']
    for col in categorical_cols:
        le = label_encoders[col]
        # map unseen labels to most common trained class before transform
        test_df[col] = test_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        test_df[col] = le.transform(test_df[col].astype(str))

    X_test = test_df.drop(['order_id', 'status'], axis=1)
    y_test = test_df['status'].astype(str)
    y_test_encoded = le_target.transform(y_test)

    X_test_scaled = scaler.transform(X_test)

    knn_pred = knn.predict(X_test_scaled)
    rf_pred = rf.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)

    knn_accuracy = accuracy_score(y_test_encoded, knn_pred)
    rf_accuracy = accuracy_score(y_test_encoded, rf_pred)
    xgb_accuracy = accuracy_score(y_test_encoded, xgb_pred)

    knn_report = classification_report(y_test_encoded, knn_pred, target_names=le_target.classes_)
    rf_report = classification_report(y_test_encoded, rf_pred, target_names=le_target.classes_)
    xgb_report = classification_report(y_test_encoded, xgb_pred, target_names=le_target.classes_)

    print("Test Results:")
    print(f"K-NN Accuracy: {knn_accuracy:.4f}")
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")

    print("\nK-NN Classification Report:")
    print(knn_report)
    print("\nRandom Forest Classification Report:")
    print(rf_report)
    print("\nXGBoost Classification Report:")
    print(xgb_report)

    # Choose best model for confusion matrix
    best_pred = None
    best_name = "K-NN"
    best_acc = knn_accuracy
    if rf_accuracy > best_acc:
        best_acc = rf_accuracy
        best_pred = rf_pred
        best_name = "Random Forest"
    if xgb_accuracy > best_acc:
        best_acc = xgb_accuracy
        best_pred = xgb_pred
        best_name = "XGBoost"
    if best_pred is None:
        best_pred = knn_pred

    cm = confusion_matrix(y_test_encoded, best_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le_target.classes_, yticklabels=le_target.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()

    plt.savefig("models/confusion_matrix.png")
    print("Saved confusion matrix to models/confusion_matrix.png")
    plt.show()

    return {
        'knn_accuracy': knn_accuracy,
        'rf_accuracy': rf_accuracy,
        'xgb_accuracy': xgb_accuracy,
        'knn_report': knn_report,
        'rf_report': rf_report,
        'xgb_report': xgb_report
    }

if __name__ == "__main__":
    if not os.path.exists("models/knn_model.pkl"):
        print("Models not found. Please run data_training.py first.")
    else:
        test_models()
