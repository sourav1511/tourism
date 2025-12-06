import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os
# Download and load the model
model_path = hf_hub_download(repo_id="sp1505/tourism", filename="best_tourism_model_v1.joblib")
model = joblib.load(model_path)

model_path = hf_hub_download(
    repo_id="sourav1511/tourism",   
    filename="best_tourism_model_v1.joblib",
    repo_type="model",
    token=os.environ.get("HF_TOKEN")
)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism App")
st.write("""
This application predicts the likelihood of a a person taking a product.
Please enter the sensor and configuration data below to get a prediction.
""")

# User input
city_tier = st.number_input("City Tier", min_value=1, max_value=3)
duration_pitch = st.number_input("duration_pitch", min_value=5, max_value=127)
number_person = st.number_input("number_person", min_value=1, max_value=5)
Followups = st.number_input("Followups", min_value=1, max_value=6)
property_star = st.number_input("property_Star", min_value=3, max_value=5)
number_trips = st.number_input("number_trips", min_value=1, max_value=22)
passport = st.number_input("passport", min_value=0, max_value=1)
pitchscore = st.number_input("pitchscore", min_value=1, max_value=5)
owncar = st.number_input("owncar", min_value=0, max_value=1)
childvisitor = st.number_input("childvisitor", min_value=0, max_value=3)
Monthly_income = st.number_input("Income")
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
Marital_status = st.selectbox("Marital_status", ["Married", "Single", "Divorced", "Unmarried"])
Product_pitched = st.selectbox("Product_pitch", ["Basic", "Delux", "Standard", "King", "Super Deluxe"])
Gender = st.selectbox("Gender", ["Male", "Female"])
Occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Bussiness", "Large Bussiness"])
Type_contact = st.selectbox("Type_contact", ["Company Invited", "Self Inquiry"])

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'City Tier': city_tier,
    'duration_pitch': duration_pitch,
    'number_person': number_person,
    'Followups': Followups,
    'property_star': property_star,
    'number_trips': number_trips,
    'passport': passport,
    'pitchscore': pitchscore,
    'owncar': owncar,
    'childvisitor': childvisitor,
    'Monthly_income': Monthly_income,
    'Designation': Designation,
    'Marital_status': Marital_status,
    'Product_pitched': Product_pitched,
    'Gender': Gender,
    'Occupation': Occupation,
    'Type_contact': Type_contact
  }])


if st.button("Predict Product Taken"):
    prediction = model.predict(input_data)[0]
    result = "Product Taken" if prediction == 1 else "No Failure"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
