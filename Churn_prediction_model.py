#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-10-17T06:07:55.694Z
"""

# # Problem Definition
# 
# # Goal: Predict which users are likely to churn (leave the platform).
# # Outcome: Use predictions to design personalized retention strategies (e.g., reminders, offers, or support).



# --- Import Libraries ---
# Data handling
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score

# Advanced ML (optional)
import xgboost as xgb
import lightgbm as lgb
import shap

# Other utilities
import warnings
warnings.filterwarnings('ignore')

print("✅ All libraries installed and imported successfully!")


# # Data Collection


df = pd.read_csv("LMS Student Data Set.csv")

df.head()


df.columns

# # Data Preprocessing


# Check missing values
df.isnull().sum()

# fill missing values (for numeric/categorical)
df['Average Course \nFeedback'].fillna(df['Average Course \nFeedback'].mean(), inplace=True)
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)

# Convert Categorical Features to Numeric
cat_cols = ['Gender', 'Region', 'Age_band', 'Code Module',
             'Label', 'Assessment type']

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])


# Feature Scaling
num_cols = ['Average Course \nFeedback', 'Duration\n(In weeks)',
            'Studied credits', 'No.of previous attempts',
            'No.of days viewed', 'Total clicks',
            'No.of clicks person done', 'No.of exercises']

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])




# Create New Features
# Convert date columns to datetime
df['Date_registered'] = pd.to_datetime(df['Date_registered'], errors='coerce')
df['Date of submission'] = pd.to_datetime(df['Date of submission'], errors='coerce')

# Days between registration and submission (as engagement indicator)
df['days_between'] = (df['Date of submission'] - df['Date_registered']).dt.days

# Completion ratio (using exercises as proxy)
df['completion_ratio'] = df['No.of exercises'] / (df['Studied credits'] + 1)


print("✅ Data preprocessing completed successfully!")
print("Shape after preprocessing:", df.shape)
df.head()


# # Exploratory Data Analysis (EDA)


# Basic structure and summary
print("Shape of dataset:", df.shape)
print("\nDataset info:")
df.info()

print("\nStatistical summary:")
display(df.describe())


missing_percent = df.isnull().mean() * 100
missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x=missing_percent.values, y=missing_percent.index, palette='Reds_r')
plt.title("📊 Percentage of Missing Values per Column", fontsize=14, weight='bold')
plt.xlabel("Percent Missing (%)")
plt.ylabel("Columns")
plt.show()


plt.figure(figsize=(6,4))
sns.countplot(x='Label', data=df, palette='coolwarm')
plt.title('Churn vs Active Users')
plt.xlabel('Label (1 = Churned, 0 = Active)')
plt.ylabel('Count')
plt.show()

print(df['Label'].value_counts(normalize=True) * 100)


# Example: Total clicks vs Churn
plt.figure(figsize=(6,4))
sns.boxplot(x='Label', y='Total clicks', data=df, palette='viridis')
plt.title('Total Clicks by Churn Status')
plt.show()

# Example: No.of days viewed vs Churn
plt.figure(figsize=(6,4))
sns.boxplot(x='Label', y='No.of days viewed', data=df, palette='magma')
plt.title('Days Viewed by Churn Status')
plt.show()


# Correlation Heatmap
# Correlation Heatmap
plt.figure(figsize=(12,8))

# Select only numeric columns
corr = df.select_dtypes(include=['number']).corr()

sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
plt.title("Feature Correlation Heatmap")
plt.show()


# Pairplot
sns.pairplot(df[['Label', 'Total clicks', 'No.of days viewed', 
                 'Studied credits', 'Average Course \nFeedback']], 
             hue='Label', diag_kind='kde', palette='husl')
plt.show()


# # Model Building


# 1️⃣ Drop or convert columns that are not numeric
drop_cols = [
    'Date_registered',
    'Course duration start date',
    'Date of submission'
]
# Drop these if present
for col in drop_cols:
    if col in df.columns:
        df = df.drop(columns=[col])

# 2️⃣ Separate features and target
X = df.drop(columns=['Label', 'S.No.', 'Student-Id'])
y = df['Label']


# 3️⃣ Ensure numeric dtypes
X = pd.get_dummies(X, drop_first=True)  # converts categorical to dummy vars

# 4️⃣ Split into train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("✅ Data prepared successfully!")
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

 # ✅ Fix: specify multi-class objective
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss',  # for multi-class
    objective='multi:softmax',  # multi-class classification
    num_class=len(y_train.unique())  # number of unique target classes
)

# Fit the model
xgb_model.fit(X_train, y_train)

print("✅ XGBoost multi-class model trained successfully!")

import json
import os

# ✅ Create a folder for model assets (same as used in PyCharm)
os.makedirs("model_assets", exist_ok=True)

# ✅ Save the feature column names
train_cols = X_train.columns.tolist()

with open("model_assets/training_columns.json", "w") as f:
    json.dump(train_cols, f)

print("✅ training_columns.json saved successfully in 'model_assets/' folder.")


# Predictions
y_pred = xgb_model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)

# ROC AUC for multi-class
roc = roc_auc_score(y_test, xgb_model.predict_proba(X_test), multi_class='ovr')

# Classification report
report = classification_report(y_test, y_pred)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("🎯 Model Performance:")
print(f"Accuracy: {acc:.4f}")
print(f"ROC-AUC (OVR): {roc:.4f}")
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", cm)

# Feature Importance Visualization
import matplotlib.pyplot as plt
import xgboost as xgb

xgb.plot_importance(xgb_model, max_num_features=10)
plt.title("Top 10 Important Features Influencing Churn")
plt.show()


# ## 📘 Project Summary / Conclusion
# 
# In this project, a **Churn Prediction Model** was developed using **XGBoost** to analyze student engagement patterns within the LMS platform.  
# The model effectively identified key churn indicators such as **low activity (fewer days viewed, total clicks, and exercises)** and **poor course feedback**.  
# 
# With an accuracy and **ROC-AUC score** indicating strong predictive performance, the model provides actionable insights to reduce student attrition.  
# 
# By implementing **personalized learning support**, **early engagement reminders**, and **feedback-driven improvements**, the institution can significantly enhance **user retention** and **learning outcomes**.
# 


import joblib
import os

# ✅ Create a folder to store the model
os.makedirs("model_assets", exist_ok=True)

# ✅ Save the trained model
joblib.dump(xgb_model, "model_assets/xgb_churn_model.pkl")

print("✅ Model saved successfully at: model_assets/xgb_churn_model.pkl")


import os
print(os.getcwd())