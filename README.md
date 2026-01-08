👨‍💼 HR Attrition & Performance Prediction System End-to-End Machine Learning + Streamlit + Python Project

This project demonstrates a complete workflow for predicting:

✔ Employee Attrition ✔ Employee Performance Rating ✔ Employee Promotion Likelihood

Using real-world style HR data and an end-to-end pipeline involving:

Jupyter Notebook (.ipynb)

Python preprocessing

Machine Learning Model training

Pickle model export

Streamlit Web Application (.py)

✅ Step-by-Step Process

Created the Jupyter Notebook (hr_prediction.ipynb) ➤ Loaded the Employee Dataset Using Pandas:

Checked shape, column info, and datatypes

Displayed initial few rows

Identified numeric & categorical features

Performed Data Cleaning & Pre-processing ✔ Removed null values ✔ Cleaned categorical columns ✔ Encoded binary columns like:

OverTime → Yes/No

Gender → Male/Female ✔ One-hot encoded JobRole, MaritalStatus, BusinessTravel ✔ Selected 12–15 best features for modeling ✔ Standardized numeric values using StandardScaler ✔ Split dataset into train & test sets

Created final X (features) and y (target) for:

Targets:

Attrition (Yes/No → 1/0)

PerformanceRating

Promotion (YearsSinceLastPromotion based synthetic label)

Machine Learning Model Training Built and tested multiple ML models:

🔹 Logistic Regression 🔹 KNN Classifier 🔹 Decision Tree 🔹 Random Forest (best performer)

Performed:

Accuracy evaluation

Precision, Recall, F1-score

ROC-AUC score

Confusion matrix

Hyperparameter tuning for RandomForest

Achieved 80-83% accuracy using selected features.

Exported Models & Scalers Using Pickle Saved best models:

attrition_model.pkl performance_model.pkl promotion_model.pkl scaler.pkl performance_scaler.pkl promotion_scaler.pkl

These models are used inside the Streamlit application for real-time prediction.

Built the Streamlit Application (hr_prediction.py) Created a full navigation-based dashboard with 4 pages:

📊 Home Dashboard

Shows useful insights:

High Risk Employees

High Job Satisfaction group

High Performance employees

🔮 Attrition Prediction Page

Clean aligned UI

User enters employee details

Features auto-encoded and scaled

Model predicts whether employee will leave

Displays processed input dataframe

📈 Performance Rating Prediction Page

Predicts employee performance (1–4)

Uses performance model & scaler

Shows input preview table

🎓 Promotion Likelihood Prediction Page

Predicts if employee will be promoted

Uses synthetic Promotion label

Displays input dataframe

Streamlit app also includes:

✔ Sidebar menu ✔ Clean UI alignment ✔ Data previews for each section ✔ Error-free preprocessing pipeline

🧰 Technologies Used Component Technology Data Processing Python, Pandas, NumPy ML Models Logistic Regression, KNN, Decision Tree, Random Forest Feature Scaling StandardScaler Model Storage Pickle Web App Streamlit Notebook Jupyter Notebook Visualization Streamlit DataFrames
