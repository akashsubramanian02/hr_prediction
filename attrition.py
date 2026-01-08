import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="HR Prediction App", layout="wide")

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    return pd.read_csv("D:\Guvi-Projects\Project 3\Employee-Attrition - Employee-Attrition.csv")

df = load_data()

# =====================================================
# LOAD MODELS & SCALERS
# =====================================================
@st.cache_resource
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

attrition_model = load_pickle("attrition_model.pkl")
promotion_model = load_pickle("promotion_model.pkl")
performance_model = load_pickle("performance_model.pkl")

attrition_scaler = load_pickle("attrition_scaler.pkl")
promotion_scaler = load_pickle("promotion_scaler.pkl")
performance_scaler = load_pickle("performance_scaler.pkl")

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("📌 Navigation")

menu = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🔮 Attrition Prediction", "🎓 Promotion Prediction", "⭐ Performance Prediction"]
)

# =====================================================
# HOME PAGE
# =====================================================
if menu == "🏠 Home":
    st.title("🏠 Employee Dataset")
    st.dataframe(df, use_container_width=True)

# =====================================================
# ATTRITION PAGE
# =====================================================
elif menu == "🔮 Attrition Prediction":

    st.title("🔮 Employee Attrition Prediction")

    # =====================================================
    # FEATURES USED IN TRAINING
    # =====================================================
    attrition_features = [
        'Age', 'MonthlyIncome', 'DistanceFromHome', 'TotalWorkingYears',
        'YearsAtCompany', 'YearsInCurrentRole', 'PercentSalaryHike',
        'JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance',
        'OverTime', 'Gender', 'JobRole', 'MaritalStatus', 'BusinessTravel'
    ]

    attrition_numeric = [
        'Age', 'MonthlyIncome', 'DistanceFromHome', 'TotalWorkingYears',
        'YearsAtCompany', 'YearsInCurrentRole', 'PercentSalaryHike',
        'JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance'
    ]

    # =====================================================
    # FIXED JOB ROLE MAPPING (VERY IMPORTANT)
    # =====================================================
    jobrole_mapping = {
        role: idx for idx, role in enumerate(sorted(df["JobRole"].unique()))
    }

    # =====================================================
    # SHOW DATA USED
    # =====================================================
    st.subheader("📊 Employee History")
    st.dataframe(df[attrition_features].head(), use_container_width=True)

    # =====================================================
    # INPUT FORM
    # =====================================================
    st.subheader("📝 Enter Employee Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 18, 60)
        income = st.number_input("Monthly Income", 1000, 60000)
        distance = st.number_input("Distance From Home", 1, 30)
        overtime = st.selectbox("OverTime", ["No", "Yes"])
        gender = st.selectbox("Gender", ["Female", "Male"])

    with col2:
        years_company = st.number_input("Years At Company", 0, 40)
        years_role = st.number_input("Years In Current Role", 0, 20)
        total_work_years = st.number_input("Total Working Years", 0, 40)
        jobrole = st.selectbox("Job Role", sorted(df["JobRole"].unique()))

    with col3:
        salary_hike = st.number_input("Percent Salary Hike", 1, 100)
        job_sat = st.slider("Job Satisfaction", 1, 4)
        env_sat = st.slider("Environment Satisfaction", 1, 4)
        worklife = st.slider("Work-Life Balance", 1, 4)
        marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        travel = st.selectbox(
            "Business Travel",
            ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
        )

    # =====================================================
    # PREDICTION
    # =====================================================
    if st.button("Predict Attrition", use_container_width=True):

        df_input = pd.DataFrame([[
            age, income, distance, total_work_years,
            years_company, years_role, salary_hike,
            job_sat, env_sat, worklife,
            1 if overtime == "Yes" else 0,
            1 if gender == "Male" else 0,
            jobrole, marital, travel
        ]], columns=attrition_features)

        # ---------- Encoding categorical variables ----------
        df_input["JobRole"] = df_input["JobRole"].map(jobrole_mapping)

        df_input["MaritalStatus"] = df_input["MaritalStatus"].map({
            "Single": 0,
            "Married": 1,
            "Divorced": 2
        })

        df_input["BusinessTravel"] = df_input["BusinessTravel"].map({
            "Non-Travel": 0,
            "Travel_Rarely": 1,
            "Travel_Frequently": 2
        })

        # ---------- Scale ONLY numeric columns ----------
        df_input[attrition_numeric] = attrition_scaler.transform(
            df_input[attrition_numeric]
        )

        # ---------- Prediction using probability ----------
        X_final = df_input.values
        prob = attrition_model.predict_proba(X_final)[0][1]  # Attrition = Yes

        if prob >= 0.35:
            st.error(f"⚠ Employee is LIKELY to leave (Risk Score: {prob:.2f})")
        else:
            st.success(f"✔ Employee is NOT likely to leave (Risk Score: {prob:.2f})")

# =====================================================
# PROMOTION PAGE
# =====================================================
elif menu == "🎓 Promotion Prediction":

    st.title("🎓 Promotion Prediction")

    promo_features = [
        'PerformanceRating', 'YearsAtCompany', 'YearsInCurrentRole',
        'TotalWorkingYears', 'JobInvolvement', 'JobSatisfaction',
        'EnvironmentSatisfaction', 'TrainingTimesLastYear',
        'MonthlyIncome', 'JobLevel'
    ]

    st.subheader("📊 Columns Used for Promotion Model")
    st.dataframe(df[promo_features].head(), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        perf = st.slider("Performance Rating", 1, 4)
        years_company = st.number_input("Years At Company", 0, 40)
        years_role = st.number_input("Years In Current Role", 0, 20)
        total_years = st.number_input("Total Working Years", 0, 40)
        job_inv = st.slider("Job Involvement", 1, 4)

    with col2:
        job_sat = st.slider("Job Satisfaction", 1, 4)
        env_sat = st.slider("Environment Satisfaction", 1, 4)
        training = st.number_input("Training Times Last Year", 0, 10)
        income = st.number_input("Monthly Income", 1000, 60000)
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])

    if st.button("Predict Promotion", use_container_width=True):

        X = pd.DataFrame([[perf, years_company, years_role, total_years,
                           job_inv, job_sat, env_sat, training, income, job_level]],
                         columns=promo_features)

        X_scaled = promotion_scaler.transform(X)
        pred = promotion_model.predict(X_scaled)[0]

        if pred == 1:
            st.success("🎉 Employee is LIKELY to get promotion")
        else:
            st.warning("❌ Employee is NOT likely to get promotion")

# =====================================================
# PERFORMANCE PAGE
# =====================================================
elif menu == "⭐ Performance Prediction":

    st.title("⭐ Performance Rating Prediction")

    perf_features = [
        'Education', 'JobInvolvement', 'JobLevel',
        'MonthlyIncome', 'YearsAtCompany', 'YearsInCurrentRole'
    ]

    st.subheader("📊 Columns Used for Performance Model")
    st.dataframe(df[perf_features].head(), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        edu = st.selectbox("Education", [1, 2, 3, 4, 5])
        job_inv = st.slider("Job Involvement", 1, 4)
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])

    with col2:
        income = st.number_input("Monthly Income", 1000, 60000)
        years_company = st.number_input("Years At Company", 0, 40)
        years_role = st.number_input("Years In Current Role", 0, 20)

    if st.button("Predict Performance", use_container_width=True):

        X = pd.DataFrame([[edu, job_inv, job_level,
                           income, years_company, years_role]],
                         columns=perf_features)

        X_scaled = performance_scaler.transform(X)
        pred = performance_model.predict(X_scaled)[0]

        st.success(f"⭐ Predicted Performance Rating: {round(pred, 2)}")
