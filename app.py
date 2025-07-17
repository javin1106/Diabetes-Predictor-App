# 📦 Importing Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ⚙️ Streamlit Page Configuration
st.set_page_config(page_title="🩺 Diabetes Risk Predictor", layout="centered")

# 📥 Load Dataset
df = pd.read_csv('C:/Users/LENOVO/Desktop/Javin Programming/machine learning files/Cohort/Streamlit/Diabetes Prediction/diabetes.csv')

# 🧪 Check Dataset Structure
print(df.shape)  # Show dimensions
print(df.isnull().sum())  # Check for missing values
print(df.columns)  # Display column names

# ⚠️ Replace 0s with NaN in features where 0 is not a valid value
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)

# 🩹 Fill missing values with median
df[cols_with_zero] = df[cols_with_zero].fillna(df[cols_with_zero].median())

# 📊 Display summary statistics
print(df.describe())

# 🧾 Feature Matrix and Target Variable
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 📚 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔁 Cached Function for GridSearchCV Training
@st.cache_resource
def train_model():
    # 🔧 Parameter Grid for Random Forest
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    }
    
    # 🔍 Grid Search with Cross Validation
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring='accuracy'
    )
    grid.fit(X_train, y_train)
    return grid

# 🧠 Train Model and Get Best Estimator
grid_search = train_model()
best_model = grid_search.best_estimator_

# 💾 Save the Best Model
joblib.dump(best_model, "diabetes_model.pkl")

# 🎯 Make Predictions and Evaluate Accuracy
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 💻 Streamlit App UI Starts Here
st.title("🩺 Diabetes Prediction App")
st.divider()

# 📍 Sidebar Navigation
st.sidebar.title("🔎 Navigation")
st.sidebar.markdown("ML-powered Diabetes Risk Prediction")
page = st.sidebar.radio("📑 Navigation", ["🏠 Home", "🧪 Diabetes Prediction", "📂 About Dataset"])

# 🔻 Footer Definition
def footer():
    st.markdown("---")
    st.markdown("""
    <b>© 2025 Javin Chutani | Built with ❤️ using Streamlit</b>  
    Connect: 
    <a href='https://www.linkedin.com/in/javin-chutani/' target='_blank'>LinkedIn</a> | 
    <a href='https://github.com/javin1106' target='_blank'>GitHub</a> | 
    <a href='https://x.com/JavinChutani' target='_blank'>X (Twitter)</a>
    """, unsafe_allow_html=True)

# 🏠 Home Page
if page == "🏠 Home":
    st.title("**😄 Welcome to the Diabetes Risk Predictor App**")
    st.markdown("""
        This app uses a Machine Learning model **(Random Forest)** to predict whether someone is at high risk of diabetes, based on certain health metrics.

        The model was trained using **GridSearchCV** to find the best combination of parameters.
        
        The following health metrics have been taken into account for the prediction:\n
        • Pregnancies\n
        • Glucose\n
        • Blood Pressure\n
        • Skin Thickness\n
        • Insulin\n
        • Body Mass Index (BMI)\n
        • Diabetes Pedigree Function (DPF)\n
        • Age\n
    """)
    
    # 📊 Show Model Accuracy
    st.divider()
    st.title("**📊 Model Performance**")
    st.markdown(f"**🎯 Predictor Accuracy: {accuracy * 100:.2f}%**")
    
    # 📌 Feature Importance Visualization
    importances = grid_search.best_estimator_.feature_importances_
    feature_names = X.columns
    fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
    ax_imp.barh(feature_names, importances, color='skyblue')
    ax_imp.set_xlabel("Feature Importance Score")
    ax_imp.set_title("Feature Importance")
    ax_imp.invert_yaxis()
    st.markdown("**📌 Feature Importance:**")
    st.pyplot(fig_imp)
    
    # 📈 Confusion Matrix
    st.markdown("**📈 Confusion Matrix:**")
    cm = confusion_matrix(y_test, grid_search.predict(X_test))
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    
    footer()

# 🧪 Prediction Page
elif page == "🧪 Diabetes Prediction":
    st.title("🧪 Predict Your Diabetes Risk")
    
    # 📝 User Input for Prediction
    st.write("Fill in the following health metrics to see your predicted risk level.")
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=17, value=0)
    glucose = st.number_input('Glucose', min_value=40, max_value=200, value=120)
    blood_pressure = st.number_input('Blood Pressure', min_value=20, max_value=125, value=72)
    skin_thickness = st.number_input('Skin Thickness', min_value=7, max_value=100, value=30)
    insulin = st.number_input('Insulin', min_value=15, max_value=850, value=125)
    bmi = st.number_input('BMI', min_value=10.0, max_value=70.0, value=32.0, format="%.1f")
    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.05, max_value=3.0, value=0.37, format="%.3f")
    age = st.number_input('Age', min_value=18, max_value=100, value=30)

    # 🔮 Make Prediction on User Input
    if st.button("Predict"):
        user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,insulin, bmi, dpf, age]])
        prediction = grid_search.predict(user_data)[0]  # Predicted class (0 or 1)
        probability = grid_search.predict_proba(user_data)[0][1]  # Probability for class 1 (Diabetes)

        # 🚨 Show Prediction Result
        if prediction == 1:
            st.error(f"⚠️ High Risk of Diabetes! ({probability*100:.2f}%)")
        else:
            st.success(f"✅ Low Risk of Diabetes ({probability*100:.2f}%)")
            
    footer()

# 📂 Dataset Info Page
elif page == "📂 About Dataset":
    st.title("📂 About the Dataset")

    st.markdown("""
    **Dataset Name:** Pima Indians Diabetes Database  
    **Source:** [Kaggle / UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
    **Total Rows:** 768  
    **Features:** 8  
    **Target Variable:** `Outcome` (1 = Diabetes, 0 = No Diabetes)
    
    **Feature Descriptions:**
    - **Pregnancies**: Number of times pregnant
    - **Glucose**: Plasma glucose concentration
    - **BloodPressure**: Diastolic blood pressure (mm Hg)
    - **SkinThickness**: Triceps skin fold thickness (mm)
    - **Insulin**: 2-Hour serum insulin (mu U/ml)
    - **BMI**: Body mass index (weight in kg / (height in m)^2)
    - **DiabetesPedigreeFunction**: Likelihood of diabetes based on family history
    - **Age**: Age in years

    The dataset consists of medical information collected from Pima Indian women aged 21 and older.
    """)
    footer()
