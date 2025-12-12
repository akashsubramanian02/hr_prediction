import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="HR Prediction App", layout="wide") 

# =====================================================
# LOAD PICKLES
# =====================================================
@st.cache_resource
def load_pickle(path):
    return pickle.load(open(path, "rb"))

# Load all models & scalers
attrition_model = load_pickle("attrition_model.pkl")
attrition_scaler = load_pickle("scaler.pkl")

performance_model = load_pickle("performance_model.pkl")
performance_scaler = load_pickle("performance_scaler.pkl")

promotion_model = load_pickle("promotion_model.pkl")
promotion_scaler = load_pickle("promotion_scaler.pkl")

# Promotion features used during training
promotion_features = [
    'PerformanceRating',
    'YearsAtCompany',
    'YearsInCurrentRole',
    'TotalWorkingYears',
    'JobInvolvement',
    'JobSatisfaction',
    'EnvironmentSatisfaction',
    'TrainingTimesLastYear',
    'MonthlyIncome',
    'JobLevel'
]

# =====================================================
# LOAD DATA
# =====================================================
DATA_PATH = r"D:\Guvi-Projects\Project 3\Employee-Attrition - Employee-Attrition.csv"

@st.cache_resource
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# Recreate promotion label (same as training)
df["Promotion"] = df["YearsSinceLastPromotion"].apply(lambda x: 1 if x == 0 else 0)

# =====================================================
# ATTRITION FEATURE SETUP
# =====================================================
numeric_cols = [
    'Age', 'MonthlyIncome', 'DistanceFromHome', 'TotalWorkingYears',
    'YearsAtCompany', 'YearsInCurrentRole', 'PercentSalaryHike',
    'JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance'
]

def get_feature_columns(df):
    df_enc = df.copy()
    df_enc['OverTime'] = df_enc['OverTime'].map({'No': 0, 'Yes': 1})
    df_enc['Gender'] = df_enc['Gender'].map({'Female': 0, 'Male': 1})
    df_enc = pd.get_dummies(df_enc, columns=['JobRole', 'MaritalStatus', 'BusinessTravel'], drop_first=True)
    df_enc = df_enc.drop(columns=['Attrition'])
    return df_enc.columns.tolist()

attrition_feature_cols = get_feature_columns(
    df[numeric_cols + ['OverTime', 'Gender', 'JobRole', 'MaritalStatus', 'BusinessTravel', 'Attrition']]
)

def build_attrition_input(data_dict):
    df_input = pd.DataFrame([data_dict])
    df_input['OverTime'] = df_input['OverTime'].map({'No': 0, 'Yes': 1})
    df_input['Gender'] = df_input['Gender'].map({'Female': 0, 'Male': 1})
    df_input = pd.get_dummies(df_input, columns=['JobRole', 'MaritalStatus', 'BusinessTravel'], drop_first=True)
    df_input = df_input.reindex(columns=attrition_feature_cols, fill_value=0)
    df_input[numeric_cols] = attrition_scaler.transform(df_input[numeric_cols])
    return df_input

# =====================================================
# SIDEBAR MENU
# =====================================================
menu = st.sidebar.radio(
    "📌 Navigation",
    ["🏠 Home", "🔮 Predict Attrition", "⭐ Performance Prediction", "🎓 Promotion Prediction"]
)

# =====================================================
# HOME PAGE
# =====================================================
if menu == "🏠 Home":
    st.title("📊 Employee Insights Dashboard")
    st.dataframe(df, use_container_width=True)

# =====================================================
# ATTRITION PAGE
# =====================================================
elif menu == "🔮 Predict Attrition":

    st.title("🔮 Employee Attrition Prediction")

    st.write("### Dataset Preview")
    st.dataframe(df, use_container_width=True)

    st.subheader("Enter Employee Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 18, 60)
        years_company = st.number_input("Years At Company", 0, 40)
        income = st.number_input("Monthly Income", 1000, 60000)
        jobrole = st.selectbox("Job Role", sorted(df["JobRole"].unique()))

    with col2:
        distance = st.number_input("Distance From Home", 1, 30)
        years_role = st.number_input("Years In Current Role", 0, 20)
        total_work_years = st.number_input("Total Working Years", 0, 40)
        marital = st.selectbox("Marital Status", sorted(df["MaritalStatus"].unique()))

    with col3:
        salary_hike = st.number_input("Percent Salary Hike", 1, 100)
        job_sat = st.slider("Job Satisfaction", 1, 4)
        env_sat = st.slider("Environment Satisfaction", 1, 4)
        worklife = st.slider("Work-Life Balance", 1, 4)
        travel = st.selectbox("Business Travel", sorted(df["BusinessTravel"].unique()))

    overtime = st.selectbox("OverTime", ["No", "Yes"])
    gender = st.selectbox("Gender", ["Female", "Male"])

    if st.button("Predict Attrition", use_container_width=True):
        input_dict = {
            "Age": age,
            "MonthlyIncome": income,
            "DistanceFromHome": distance,
            "TotalWorkingYears": total_work_years,
            "YearsAtCompany": years_company,
            "YearsInCurrentRole": years_role,
            "PercentSalaryHike": salary_hike,
            "JobSatisfaction": job_sat,
            "EnvironmentSatisfaction": env_sat,
            "WorkLifeBalance": worklife,
            "OverTime": overtime,
            "Gender": gender,
            "JobRole": jobrole,
            "MaritalStatus": marital,
            "BusinessTravel": travel
        }

        X_attr = build_attrition_input(input_dict)
        pred = attrition_model.predict(X_attr)[0]

        if pred == 0:
            st.success("✔ Employee is NOT likely to leave.")
        else:
            st.error("⚠ Employee is LIKELY to leave!")

# =====================================================
# PERFORMANCE PREDICTION
# =====================================================
elif menu == "⭐ Performance Prediction":

    st.title("⭐ Performance Rating Prediction")

    st.write("### Dataset Preview")
    st.dataframe(
        df[["Education", "JobInvolvement", "JobLevel", "MonthlyIncome", "YearsAtCompany", "YearsInCurrentRole"]],
        use_container_width=True
    )

    edu = st.selectbox("Education", sorted(df["Education"].unique()))
    involve = st.selectbox("Job Involvement", sorted(df["JobInvolvement"].unique()))
    level = st.selectbox("Job Level", sorted(df["JobLevel"].unique()))
    perf_income = st.number_input("Monthly Income", 1000, 60000)
    yc = st.number_input("Years At Company", 0, 40)
    yr = st.number_input("Years In Current Role", 0, 20)

    if st.button("Predict Performance Rating"):
        X_perf = pd.DataFrame([{
            "Education": edu,
            "JobInvolvement": involve,
            "JobLevel": level,
            "MonthlyIncome": perf_income,
            "YearsAtCompany": yc,
            "YearsInCurrentRole": yr
        }])

        X_scaled = performance_scaler.transform(X_perf)
        pred = performance_model.predict(X_scaled)[0]

        st.success(f"⭐ Predicted Performance Rating = {pred}")

# =====================================================
# PROMOTION PREDICTION
# =====================================================
elif menu == "🎓 Promotion Prediction":

    st.title("🎓 Promotion Likelihood Prediction")

    st.write("### Dataset Preview")
    st.dataframe(df[promotion_features + ["Promotion"]], use_container_width=True)

    pr = st.selectbox("Performance Rating", sorted(df["PerformanceRating"].unique()))
    yc = st.number_input("Years At Company", 0, 40)
    ycr = st.number_input("Years In Current Role", 0, 20)
    twy = st.number_input("Total Working Years", 0, 40)
    jinvolve = st.selectbox("Job Involvement", sorted(df["JobInvolvement"].unique()))
    jsat = st.selectbox("Job Satisfaction", sorted(df["JobSatisfaction"].unique()))
    esat = st.selectbox("Environment Satisfaction", sorted(df["EnvironmentSatisfaction"].unique()))
    train = st.number_input("Training Times Last Year", 0, 10)
    income = st.number_input("Monthly Income", 1000, 60000)
    jlevel = st.selectbox("Job Level", sorted(df["JobLevel"].unique()))

    if st.button("Predict Promotion", use_container_width=True):
        X_promo = pd.DataFrame([{
            "PerformanceRating": pr,
            "YearsAtCompany": yc,
            "YearsInCurrentRole": ycr,
            "TotalWorkingYears": twy,
            "JobInvolvement": jinvolve,
            "JobSatisfaction": jsat,
            "EnvironmentSatisfaction": esat,
            "TrainingTimesLastYear": train,
            "MonthlyIncome": income,
            "JobLevel": jlevel
        }])

        # Scale
        X_scaled = promotion_scaler.transform(X_promo)
        pred = promotion_model.predict(X_scaled)[0]

        if pred == 1:
            st.success("🎉 Promotion is LIKELY!")
        else:
            st.error("❌ Promotion is NOT likely.")
