# 🎓 LMS Churn Prediction Model

This project predicts which students are likely to leave an online learning platform using engagement and activity data.  
It uses **XGBoost (Gradient Boosting)** for model training and **Streamlit** for deployment, providing a user-friendly dashboard for predictions.

---

## 🚀 Features
- Analyze student activity and engagement.
- Predict churn likelihood using a trained ML model.
- Interactive Streamlit dashboard for real-time predictions.
- Scalable model ready for deployment (Streamlit Cloud or Flask).

---

## 🧠 Tech Stack
- **Python 3.10+**
- **Pandas**, **NumPy**, **Scikit-learn**
- **XGBoost**
- **Streamlit**
- **Joblib** (for model saving/loading)
- **Matplotlib / Seaborn** (for data visualization)

---

## ⚙️ Project Structure
churn_prediction_model/
│
├── app.py # Streamlit dashboard
├── xgb_churn_model.pkl # Trained ML model
├── training_columns.json # Optional feature reference
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## 💻 How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Saifullah12445745/student-churn-prediction.git
cd LMS-churn-prediction

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run Streamlit App
streamlit run app.py

📊 Business Insights

Students with low engagement (few logins, low clicks) show higher churn.

Early identification enables personalized retention support.

LMS platforms can use this to improve retention by 15–25%.

👨‍💻 Author

Saif Ullah
BS Software Engineering | Machine Learning & Data Science Enthusiast
📫 LinkedIn Profile :https://www.linkedin.com/posts/saif-ullah-90053a387_excited-to-share-my-latest-machine-learning-activity-7384660708052959232-yRcC?utm_source=share&utm_medium=member_desktop&rcm=ACoAAF9MJ5ABKbDX-UypM0eUZkD5-Hs-4l0qXO4


📂 GitHub Profile:Saifullah12445745