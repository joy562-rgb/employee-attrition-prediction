import streamlit as st
import pandas as pd
import joblib

# Load trained model (with preprocessing pipeline)
MODEL_PATH = "models/attrition_model.pkl"
model = joblib.load(MODEL_PATH)

st.title("🏢 Employee Attrition Prediction App")

# Input form
st.subheader("Enter Employee Information:")

data = {
    "Age": st.number_input("Age", min_value=18, max_value=60, value=30),
    "BusinessTravel": st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]),
    "DailyRate": st.number_input("Daily Rate", min_value=100, max_value=1500, value=800),
    "Department": st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"]),
    "DistanceFromHome": st.number_input("Distance From Home", min_value=1, max_value=30, value=10),
    "Education": st.slider("Education", 1, 5, 3),
    "EducationField": st.selectbox("Education Field",
                                   ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources",
                                    "Other"]),
    "EnvironmentSatisfaction": st.slider("Environment Satisfaction", 1, 4, 3),
    "Gender": st.selectbox("Gender", ["Male", "Female"]),
    "HourlyRate": st.number_input("Hourly Rate", min_value=30, max_value=100, value=65),
    "JobInvolvement": st.slider("Job Involvement", 1, 4, 3),
    "JobLevel": st.slider("Job Level", 1, 5, 3),
    "JobRole": st.selectbox("Job Role",
                            ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director",
                             "Healthcare Representative", "Manager", "Sales Representative", "Research Director",
                             "Human Resources"]),
    "JobSatisfaction": st.slider("Job Satisfaction", 1, 4, 3),
    "MaritalStatus": st.selectbox("Marital Status", ["Single", "Married", "Divorced"]),
    "MonthlyIncome": st.number_input("Monthly Income", min_value=1000, value=5000),
    "MonthlyRate": st.number_input("Monthly Rate", min_value=2000, max_value=30000, value=15000),
    "NumCompaniesWorked": st.number_input("Number of Companies Worked", min_value=0, max_value=10, value=2),
    "OverTime": st.selectbox("OverTime", ["Yes", "No"]),
    "PercentSalaryHike": st.number_input("Percent Salary Hike", min_value=0, max_value=25, value=15),
    "PerformanceRating": st.slider("Performance Rating", 1, 4, 3),
    "RelationshipSatisfaction": st.slider("Relationship Satisfaction", 1, 4, 3),
    "StockOptionLevel": st.slider("Stock Option Level", 0, 3, 1),
    "TotalWorkingYears": st.number_input("Total Working Years", min_value=0, max_value=40, value=10),
    "TrainingTimesLastYear": st.number_input("Training Times Last Year", min_value=0, max_value=10, value=3),
    "WorkLifeBalance": st.slider("Work Life Balance", 1, 4, 3),
    "YearsAtCompany": st.number_input("Years At Company", min_value=0, max_value=40, value=5),
    "YearsInCurrentRole": st.number_input("Years In Current Role", min_value=0, max_value=20, value=3),
    "YearsSinceLastPromotion": st.number_input("Years Since Last Promotion", min_value=0, max_value=20, value=2),
    "YearsWithCurrManager": st.number_input("Years With Current Manager", min_value=0, max_value=20, value=3),

    # ✅ Add missing columns required by the model (likely constant)
    "StandardHours": 80,
    "EmployeeCount": 1,
}

# Predict button
if st.button("Predict Attrition"):
    df = pd.DataFrame([data])

    try:
        prediction = model.predict(df)[0]
        result = "Yes - The employee is likely to leave." if prediction == 1 else "No - The employee is likely to stay."
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
