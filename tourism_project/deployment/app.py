import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os

# Download and load the model
model_path = hf_hub_download(
    repo_id="sp1505/tourism_model",
    filename="best_tourism_model_v1.joblib",
    repo_type="model",
    token=os.environ.get("HF_TOKEN")
)
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism App")
st.write("""
This application predicts the likelihood of a a person taking a product.
Please enter the sensor and configuration data below to get a prediction.
""")

st.title("Tourism Prediction App")

# User input
Age = st.number_input("Age")
TypeofContact = st.selectbox("Typeofcontact", ["Company Invited", "Self Inquiry"])
CityTier = st.number_input("CityTier", min_value=1, max_value=3)
DurationOfPitch = st.number_input("DurationOfPitch", min_value=5, max_value=127)
Occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Bussiness", "Large Bussiness"])
NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting", min_value=1, max_value=5)
NumberOfFollowups = st.number_input("NumberOfFollowups", min_value=1, max_value=6)
ProductPitched = st.selectbox("ProductPitched", ["Basic", "Delux", "Standard", "King", "Super Deluxe"])
PreferredPropertyStar = st.number_input("PreferredPropertyStar", min_value=3, max_value=5)
MaritalStatus = st.selectbox("MaritalStatus", ["Married", "Single", "Divorced", "Unmarried"])
NumberOfTrips = st.number_input("NumberOfrips", min_value=1, max_value=22)
Passport = st.number_input("Passport", min_value=0, max_value=1)
PitchSatisfactionScore = st.number_input("PitchSatisfactionScore", min_value=1, max_value=5)
OwnCar = st.number_input("OwnCar", min_value=0, max_value=1)
NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=3)
MonthlyIncome = st.number_input("MonthlyIncome")   
Gender = st.selectbox("Gender", ["Male", "Female"])
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'Gender': Gender,
    'Designation': Designation,
  }])

if st.button("Predict Product Taken"):
    prediction = model.predict(input_data)[0]
    result = "Product Taken" if prediction == 1 else "No Failure"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
