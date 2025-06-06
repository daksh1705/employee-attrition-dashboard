import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open('logistic_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("🚨 Employee Attrition Prediction Dashboard")

st.sidebar.header("Enter Employee Details")

def user_input_features():
    Age = st.sidebar.slider('Age', 18, 60, 30)
    DistanceFromHome = st.sidebar.slider('Distance From Home (km)', 1, 30, 10)
    MonthlyIncome = st.sidebar.slider('Monthly Income ($)', 1000, 20000, 5000)
    JobSatisfaction = st.sidebar.selectbox('Job Satisfaction (1=Low, 4=High)', [1, 2, 3, 4])
    WorkLifeBalance = st.sidebar.selectbox('Work-Life Balance (1=Bad, 4=Best)', [1, 2, 3, 4])
    OverTime = st.sidebar.selectbox('OverTime', ['No', 'Yes'])
    YearsAtCompany = st.sidebar.slider('Years At Company', 0, 40, 5)

    # Convert OverTime to binary
    OverTime = 1 if OverTime == 'Yes' else 0

    data = {
        'Age': Age,
        'DistanceFromHome': DistanceFromHome,
        'MonthlyIncome': MonthlyIncome,
        'JobSatisfaction': JobSatisfaction,
        'WorkLifeBalance': WorkLifeBalance,
        'OverTime': OverTime,
        'YearsAtCompany': YearsAtCompany
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()
st.subheader('Employee Information:')
st.write(input_df)

# Scale the input
scaled_input = scaler.transform(input_df)

# Make prediction
prediction = model.predict(scaled_input)[0]
proba = model.predict_proba(scaled_input)[0][1]

st.subheader("Prediction Result:")
if prediction == 1:
    st.error(f"🔴 The employee is likely to leave. Probability: {proba:.2f}")
else:
    st.success(f"🟢 The employee is likely to stay. Probability: {1 - proba:.2f}")
