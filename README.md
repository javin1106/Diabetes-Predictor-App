# 🩺 Diabetes Predictor App

This is a Streamlit web app that predicts whether a person is at risk of diabetes using machine learning.

## 🚀 Features
- Built using **Random Forest Classifier**
- Trained on the Pima Indians Diabetes Dataset
- Hyperparameter tuning using **GridSearchCV**
- Interactive UI built with **Streamlit**
- Displays **Feature Importance** and **Confusion Matrix**
- Provides personalized diabetes risk prediction

## 📂 App Structure
- `app.py`: Main Streamlit app file
- `diabetes.csv`: Dataset used
- `diabetes_model.pkl`: Trained model saved with joblib
- `requirements.txt`: Python dependencies

## 📊 Dataset
- [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- 768 entries, 8 features + 1 target (Outcome)

## 🛠 Installation

```bash
pip install -r requirements.txt
streamlit run app.py
