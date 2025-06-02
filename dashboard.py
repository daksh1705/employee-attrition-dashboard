import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model/logistic_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))


st.title("🧑‍💼 Employee Attrition Prediction Dashboard")

st.sidebar.header("Enter Employee Details")

def user_input_features():
    satisfaction = st.sidebar.slider('Satisfaction Level', 0.0, 1.0, 0.5)
    evaluation = st.sidebar.slider('Last Evaluation', 0.0, 1.0, 0.7)
    projects = st.sidebar.slider('Number of Projects', 2, 7, 4)
    avg_monthly_hours = st.sidebar.slider('Average Monthly Hours', 80, 310, 160)
    time_spent = st.sidebar.slider('Years at Company', 1, 10, 3)
    work_accident = st.sidebar.selectbox('Work Accident', (0, 1))
    promotion = st.sidebar.selectbox('Promotion Last 5 Years', (0, 1))
    salary = st.sidebar.selectbox('Salary', ['low', 'medium', 'high'])

    salary_low = 1 if salary == 'low' else 0
    salary_medium = 1 if salary == 'medium' else 0
    salary_high = 1 if salary == 'high' else 0

    features = np.array([[satisfaction, evaluation, projects, avg_monthly_hours,
                          time_spent, work_accident, promotion,
                          salary_low, salary_medium, salary_high]])
    return features

input_data = user_input_features()
input_data_scaled = scaler.transform(input_data[:, :7])
final_input = np.hstack((input_data_scaled, input_data[:, 7:]))

prediction = model.predict(final_input)
probability = model.predict_proba(final_input)

st.subheader("Prediction:")
st.write("🔴 Employee Will Leave" if prediction[0] else "🟢 Employee Will Stay")

st.subheader("Probability:")
st.write(f"Stay: {probability[0][0]*100:.2f} %")
st.write(f"Leave: {probability[0][1]*100:.2f} %")
