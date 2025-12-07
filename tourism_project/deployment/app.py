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
    filename="best_tourism_model_v1.joblib",
    repo_type="model",
    token=os.environ.get("HF_TOKEN")  # required if private
)
model = joblib.load(model_path)

st.title("Tourism Prediction App")

# User input
Age = st.number_input("Age")
TypeofContact = st.selectbox("Type_contact", ["Company Invited", "Self Inquiry"])
CityTier = st.number_input("City Tier", min_value=1, max_value=3)
DurationOfPitch = st.number_input("duration_pitch", min_value=5, max_value=127)
Occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Bussiness", "Large Bussiness"])
NumberOfPersonVisiting = st.number_input("number_person", min_value=1, max_value=5)
NumberOfFollowups = st.number_input("Followups", min_value=1, max_value=6)
ProductPitched = st.selectbox("Product_pitch", ["Basic", "Delux", "Standard", "King", "Super Deluxe"])
PreferredPropertyStar = st.number_input("property_Star", min_value=3, max_value=5)
MaritalStatus = st.selectbox("Marital_status", ["Married", "Single", "Divorced", "Unmarried"])
NumberOfTrips = st.number_input("number_trips", min_value=1, max_value=22)
Passport = st.number_input("passport", min_value=0, max_value=1)
PitchSatisfactionScore = st.number_input("pitchscore", min_value=1, max_value=5)
OwnCar = st.number_input("owncar", min_value=0, max_value=1)
NumberOfChildrenVisiting = st.number_input("childvisitor", min_value=0, max_value=3)
MonthlyIncome = st.number_input("Income")

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'City Tier': CityTier,
    'duration_pitch': DurationOfPitch,
    'Occupation': Occupation,
    'number_person': NumberOfPersonVisiting,
    'Followups': NumberOfFollowups,
    'Product_pitched': ProductPitched,
    'property_star': PreferredPropertyStar,
    'Marital_status': MaritalStatus,
    'number_trips': NumberOfTrips,
    'passport': Passport,
    'pitchscore': PitchSatisfactionScore,
    'owncar': OwnCar,
    'childvisitor': NumberOfChildrenVisiting,
    'Income': MonthlyIncome,
  }])

if st.button("Predict Product Taken"):
    prediction = model.predict(input_data)[0]
    result = "Product Taken" if prediction == 1 else "No Failure"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
