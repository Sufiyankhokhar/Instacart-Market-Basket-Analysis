import streamlit as st
from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import joblib

st.set_page_config(page_title="Instacart Reorder Prediction", layout="wide")
st.title(" Instacart Customer Reorder Prediction")


# DB CONNECTION

engine = create_engine(
    "mysql+pymysql://root:Sufi%400709@localhost:3306/instacart"
)


# LOAD ML DATASET

@st.cache_data
def load_data():
    df = pd.read_sql("SELECT * FROM ml_dataset", engine)
    return df

df = load_data()
st.write("Data loaded successfully:", df.shape)
st.dataframe(df.head(5))


# FEATURES & TARGET

X = df.drop(columns=["reordered"])
y = df["reordered"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# BASELINE MODEL: LOGISTIC REGRESSION

log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

st.subheader("Baseline Model: Logistic Regression")
st.text(classification_report(y_test, y_pred_log))
st.text("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred_log)))


# RANDOM FOREST MODEL

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

st.subheader("Random Forest Results")
st.text(classification_report(y_test, y_pred_rf))
st.text("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred_rf)))

# Save model
joblib.dump(rf_model, "rf_model.pkl")
st.success("Random Forest model saved as rf_model.pkl ✅")


# MODEL EVALUATION METRICS

y_probs = rf_model.predict_proba(X_test)[:, 1]

# ROC-AUC
roc_score = roc_auc_score(y_test, y_probs)
st.write("ROC-AUC Score:", round(roc_score, 4))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_score:.2f})")
ax.plot([0,1],[0,1],'k--')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()
st.pyplot(fig)

# Precision@K
K = 100
top_k_idx = np.argsort(y_probs)[-K:]
precision_at_k = sum(y_test.iloc[top_k_idx]) / K
st.write(f"Precision@{K}:", round(precision_at_k,4))

# Threshold tuning
st.write("Threshold Tuning Metrics:")
thresholds = np.arange(0.1, 0.9, 0.05)
for t in thresholds:
    y_pred_thresh = (y_probs >= t).astype(int)
    precision = precision_score(y_test, y_pred_thresh)
    recall = recall_score(y_test, y_pred_thresh)
    f1 = f1_score(y_test, y_pred_thresh)
    st.write(f"Threshold={t:.2f} | Precision={precision:.2f} | Recall={recall:.2f} | F1={f1:.2f}")


# INTERACTIVE USER INPUT PREDICTION

st.subheader("Predict if a product will be reordered")
st.markdown("Enter feature values below:")

user_input = {
    "order_id": st.number_input("Order ID", min_value=1, max_value=1000000, value=1),
    "user_id": st.number_input("User ID", min_value=1, max_value=1000000, value=1),
    "product_id": st.number_input("Product ID", min_value=1, max_value=1000000, value=1),
    "total_orders": st.number_input("Total orders by user", min_value=1, max_value=200, value=10),
    "avg_days_between_orders": st.number_input("Avg days between orders", min_value=1, max_value=60, value=7),
    "avg_order_hour": st.number_input("Avg order hour", min_value=0, max_value=23, value=12),
    "add_to_cart_order": st.number_input("Add to cart order", min_value=1, max_value=50, value=1)
}

input_df = pd.DataFrame([user_input])

# Align columns exactly as trained
input_df = input_df[X_train.columns]

# Load trained Random Forest model
rf_model = joblib.load("rf_model.pkl")

# Best threshold from tuning (can adjust)
best_threshold = 0.35

y_prob = rf_model.predict_proba(input_df)[:,1][0]
y_pred = int(y_prob >= best_threshold)

st.write(f"Predicted Reorder Probability: {y_prob:.2f}")
st.write("Reorder Prediction:", " Yes" if y_pred==1 else " No")
