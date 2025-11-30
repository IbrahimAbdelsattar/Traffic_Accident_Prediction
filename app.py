import streamlit as st
import pandas as pd
import pickle

# -------------------------------------
# Page Config
# -------------------------------------
st.set_page_config(page_title="Accident Severity Predictor", page_icon="🛣️", layout="centered")

# -------------------------------------
# Load Model + Encoders + Data
# -------------------------------------
with open("xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

df = pd.read_csv("cleaned.csv")

# -------------------------------------
# Helper: encode user input
# -------------------------------------
def encode_input(user_dict):
    encoded_row = {}

    for col, val in user_dict.items():
        encoder = encoders[col]              # استخدم نفس الـ encoder من التدريب
        encoded_row[col] = encoder.transform([val])[0]

    return pd.DataFrame([encoded_row])

# -------------------------------------
# UI Title
# -------------------------------------
st.markdown("<h1 style='text-align:center;'>🛣️ Road Accident Severity Predictor</h1>", unsafe_allow_html=True)
st.write("Fill in the accident details to predict the severity level.")

# -------------------------------------
# Options from data
# -------------------------------------
def ops(col):
    return sorted(df[col].dropna().unique())

# -------------------------------------
# Input Form
# -------------------------------------
with st.form("accident_form"):

    st.subheader("👤 Driver & Vehicle Info")
    c1, c2 = st.columns(2)
    with c1:
        age = st.selectbox("Age band of driver", ops("Age_band_of_driver"))
        sex = st.selectbox("Sex of driver", ops("Sex_of_driver"))
        edu = st.selectbox("Educational level", ops("Educational_level"))
    with c2:
        relation = st.selectbox("Vehicle-driver relation", ops("Vehicle_driver_relation"))
        exp = st.selectbox("Driving experience", ops("Driving_experience"))

    st.subheader("🛣️ Road & Environment")
    c3, c4 = st.columns(2)
    with c3:
        lanes = st.selectbox("Lanes or medians", ops("Lanes_or_Medians"))
        junction = st.selectbox("Type of junction", ops("Types_of_Junction"))
        surface = st.selectbox("Road surface type", ops("Road_surface_type"))
    with c4:
        light = st.selectbox("Light conditions", ops("Light_conditions"))
        weather = st.selectbox("Weather conditions", ops("Weather_conditions"))

    st.subheader("🚗 Movements & Collision")
    c5, c6 = st.columns(2)
    with c5:
        veh_move = st.selectbox("Vehicle movement", ops("Vehicle_movement"))
        ped_move = st.selectbox("Pedestrian movement", ops("Pedestrian_movement"))
    with c6:
        collision = st.selectbox("Type of collision", ops("Type_of_collision"))
        cause = st.selectbox("Cause of accident", ops("Cause_of_accident"))

    submitted = st.form_submit_button("🔍 Predict")

# -------------------------------------
# Prediction
# -------------------------------------
if submitted:
    user_input = {
        "Age_band_of_driver": age,
        "Sex_of_driver": sex,
        "Educational_level": edu,
        "Vehicle_driver_relation": relation,
        "Driving_experience": exp,
        "Lanes_or_Medians": lanes,
        "Types_of_Junction": junction,
        "Road_surface_type": surface,
        "Light_conditions": light,
        "Weather_conditions": weather,
        "Type_of_collision": collision,
        "Vehicle_movement": veh_move,
        "Pedestrian_movement": ped_move,
        "Cause_of_accident": cause
    }

    encoded_df = encode_input(user_input)
    pred = model.predict(encoded_df)[0]

    severity_map = {0: "Fatal injury", 1: "Serious injury", 2: "Slight injury"}

    st.success(f"### 🧮 Predicted Severity: **{severity_map[pred]}** (Class {pred})")
