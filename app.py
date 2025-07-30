import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💼 Employee Salary Prediction App")
st.write("Enter employee details below to predict if their income is `<=50K` or `>50K`.")

# UI Inputs
age = st.number_input("Age", min_value=17, max_value=90, value=30)
workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay'])
fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=10000, value=200000)
education = st.selectbox("Education", ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th'])
educational_num = st.slider("Education Level (Numeric)", 1, 16, 10)
marital_status = st.selectbox("Marital Status", ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent'])
occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct'])
relationship = st.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
gender = st.selectbox("Gender", ['Male', 'Female'])
capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
hours_per_week = st.slider("Hours per week", 1, 99, 40)
native_country = st.selectbox("Native Country", ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'India', 'Other'])

# Manual encoding (based on label encoding during training)
def encode_input():
    mapping = {
        'workclass': {'Private': 2, 'Self-emp-not-inc': 5, 'Self-emp-inc': 4, 'Federal-gov': 0, 'Local-gov': 1, 'State-gov': 6, 'Without-pay': 7},
        'education': {'Bachelors': 1, 'Some-college': 15, '11th': 0, 'HS-grad': 11, 'Prof-school': 13, 'Assoc-acdm': 7, 'Assoc-voc': 8, '9th': 2, '7th-8th': 3, '12th': 4},
        'marital-status': {'Married-civ-spouse': 2, 'Divorced': 0, 'Never-married': 4, 'Separated': 5, 'Widowed': 6, 'Married-spouse-absent': 3},
        'occupation': {'Tech-support': 13, 'Craft-repair': 4, 'Other-service': 10, 'Sales': 11, 'Exec-managerial': 5, 'Prof-specialty': 9, 'Handlers-cleaners': 6, 'Machine-op-inspct': 6},
        'relationship': {'Wife': 5, 'Own-child': 3, 'Husband': 0, 'Not-in-family': 2, 'Other-relative': 1, 'Unmarried': 4},
        'race': {'White': 4, 'Black': 2, 'Asian-Pac-Islander': 1, 'Amer-Indian-Eskimo': 0, 'Other': 3},
        'gender': {'Male': 1, 'Female': 0},
        'native-country': {'United-States': 38, 'Mexico': 20, 'Philippines': 30, 'Germany': 11, 'Canada': 4, 'India': 14, 'Other': 40}
    }

    return [
        age,
        mapping['workclass'][workclass],
        fnlwgt,
        mapping['education'][education],
        educational_num,
        mapping['marital-status'][marital_status],
        mapping['occupation'][occupation],
        mapping['relationship'][relationship],
        mapping['race'][race],
        mapping['gender'][gender],
        capital_gain,
        capital_loss,
        hours_per_week,
        mapping['native-country'][native_country]
    ]

# Predict
if st.button("Predict Salary"):
    user_input = np.array(encode_input()).reshape(1, -1)
    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)[0]
    st.success(f"💰 Predicted Income: {'>50K' if prediction == 1 else '<=50K'}")
