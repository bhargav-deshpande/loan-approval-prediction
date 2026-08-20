import pandas as pd
import streamlit as st
import joblib

model=joblib.load("stacking_model.pkl")
columns=joblib.load("columns.pkl")
scaler=joblib.load("scaler.pkl")

st.title("Loan Approval Prediction")
st.markdown("Provide Following Details")
st.text_input("Enter Your Loan ID")
gender=st.selectbox("Gender",['Male','Female'])
married=st.selectbox("Married",['Yes','No'])
dependents=st.selectbox("Dependents",['0','1','2','3+'])
education=st.selectbox("Education",['Graduate','Not Graduate'])
self_employed=st.selectbox("Self Employed",['Yes','No'])
app_income=st.number_input("Applicant Income",min_value=0)
coapp_incom=st.number_input("Co-Applicant Income",min_value=0)
loan=st.number_input("Loan Amount",min_value=0)
loan_term=st.number_input("Loan Amount Term",min_value=0)
credit_history=st.selectbox("Credit History",['Yes','No'])
area=st.selectbox("Property Area",['Rural','Urban','SemiUrban'])

if st.button("Predict"):
    raw_input={
        'gender': gender,
        'married': married,
        'dependents': dependents,
        'education': education,
        'self_employed': self_employed,
        'applicantincome': app_income,
        'coapplicantincome': coapp_incom,
        'loanamount': loan,
        'loan_amount_term': loan_term,
        'credit_history': 1 if credit_history=="Yes" else 0,
        'property_area': area
    }
    input_df=pd.DataFrame([raw_input])
    input_df['dependents']=input_df['dependents'].replace("3+","3")
    input_encoded = pd.get_dummies(
        input_df,
        columns=[
            "property_area",
            "gender",
            "married",
            "education",
            "self_employed",
            "dependents"
        ],
        drop_first=True,
        dtype=int
    )

    for col in columns:
        if col not in input_encoded.columns:
            input_encoded[col]=0
    input_encoded=input_encoded[columns]

    numeric_col = [
    "applicantincome",
    "loanamount",
    "loan_amount_term",
    "coapplicantincome"
    ]
    input_encoded[numeric_col] = scaler.transform(input_encoded[numeric_col])
    prediction=model.predict(input_encoded)[0]

    if prediction==1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")