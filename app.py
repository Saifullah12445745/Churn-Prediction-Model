#!/usr/bin/env python
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ================================
# 🔧 PATH CONFIGURATION
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_assets", "xgb_churn_model.pkl")
TRAIN_COLS_PATH = os.path.join(BASE_DIR, "model_assets", "training_columns.json")

# ================================
# ⚙️ STREAMLIT PAGE SETTINGS
# ================================
st.set_page_config(page_title="🎓 LMS Churn Prediction Dashboard", layout="wide")

# ================================
# 🚀 LOAD MODEL
# ================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

# ================================
# 🧹 DATA PREPROCESSING
# ================================
def clean_column_names(df):
    df.columns = df.columns.str.replace('\n', ' ', regex=True).str.strip()
    return df

def basic_preprocessing(df):
    df = clean_column_names(df)

    drop_cols = [
        'S.No.', 'Student-Id', 'Date_registered',
        'Course duration start date', 'Date of submission'
    ]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Add missing engineered columns
    if 'days_between' not in df.columns:
        df['days_between'] = 0
    if 'completion_ratio' not in df.columns:
        df['completion_ratio'] = 0.0

    # Convert columns to numeric if possible
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='ignore')

    return df

def align_with_training(df):
    df = pd.get_dummies(df, drop_first=True)
    if os.path.exists(TRAIN_COLS_PATH):
        with open(TRAIN_COLS_PATH, 'r') as f:
            train_cols = json.load(f)
        for c in train_cols:
            if c not in df.columns:
                df[c] = 0
        df = df.reindex(columns=train_cols)
    else:
        df = df.sort_index(axis=1)
    return df

# ================================
# 🔮 PREDICTION FUNCTION
# ================================
def predict(df, model):
    probs = model.predict_proba(df)
    if probs.shape[1] > 2:
        pred_class = np.argmax(probs, axis=1)
        pred_prob = probs.max(axis=1)
    else:
        pred_class = (probs[:, 1] > 0.5).astype(int)
        pred_prob = probs[:, 1]
    return pred_class, pred_prob

# ================================
# 🖥️ STREAMLIT UI
# ================================
st.title("📊 LMS Churn Prediction Dashboard")
st.markdown("Upload a CSV of students to predict who is likely to **churn** (drop out).")

uploaded_file = st.file_uploader("📤 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    raw = pd.read_csv(uploaded_file)
    st.subheader("🔍 Uploaded Data Preview")
    st.dataframe(raw.head())

    df_proc = basic_preprocessing(raw.copy())
    X = align_with_training(df_proc)

    st.subheader("🧩 Processed Data (Used for Model Input)")
    st.dataframe(X.head())

    if st.button("🚀 Run Predictions"):
        preds, probs = predict(X, model)
        results = raw.copy()
        results['Predicted_Label'] = preds
        results['Churn_Probability'] = probs

        st.success("✅ Prediction Complete!")
        st.subheader("🔥 High-Risk Students (Probability ≥ 0.7)")
        high_risk = results[results['Churn_Probability'] >= 0.7]

        if not high_risk.empty:
            st.dataframe(high_risk.sort_values('Churn_Probability', ascending=False))
        else:
            st.info("No students found with churn probability above 0.7")

        # Full results
        st.subheader("📋 Full Results (Top 50)")
        st.dataframe(results.head(50))

        # Download CSV
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results CSV",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )
else:
    st.info("👆 Please upload a CSV file to start predictions.")
