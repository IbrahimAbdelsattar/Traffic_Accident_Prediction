import streamlit as st
import pandas as pd
import pickle

# =========================
# Page Config & Style
# =========================
st.set_page_config(
    page_title="Road Accident Severity Predictor",
    page_icon="🛣️",
    layout="wide"
)

# Custom CSS to center the main container
center_style = """
    <style>
    .main-block {
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
        padding: 2rem;
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .css-18e3th9 {
        padding-top: 1rem;
    }
    .css-1d391kg {
        padding-top: 1rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
"""
st.markdown(center_style, unsafe_allow_html=True)


# =========================
# Load Model & Data
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned.csv")
    return df

@st.cache_resource
def load_model_and_columns():
    with open("xgboost-model.pkl", "rb") as f:
        model = pickle.load(f)

    df = load_data()
    # We assume you trained with get_dummies(drop_first=True)
    X = pd.get_dummies(df.drop("Accident_severity", axis=1), drop_first=True)
    feature_columns = X.columns
    return model, feature_columns, df

model, feature_columns, df = load_model_and_columns()


# =========================
# Helper: Preprocess Input
# =========================
def preprocess_user_input(input_dict):
    """
    Takes a dict of raw user inputs (categorical), returns a single-row
    dataframe encoded with get_dummies and aligned to training columns.
    """
    input_df = pd.DataFrame([input_dict])

    # Same encoding logic as training
    input_encoded = pd.get_dummies(input_df, drop_first=True)

    # Align with training columns (any missing columns -> 0)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

    return input_encoded


# =========================
# UI Header
# =========================
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <h1>🛣️ Road Accident Severity Predictor</h1>
        <p style="font-size: 1.05rem; color: #555;">
            Use this tool to estimate the severity of a road accident based on driver, vehicle, 
            road, and environmental conditions. This model is trained on real data from Addis Ababa.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-block">', unsafe_allow_html=True)

st.subheader("📋 Enter Accident Details")

# =========================
# Dynamic options from dataset
# =========================

def unique_sorted(col):
    return sorted(df[col].dropna().unique().tolist())

age_options = unique_sorted("Age_band_of_driver")
sex_options = unique_sorted("Sex_of_driver")
edu_options = unique_sorted("Educational_level")
rel_options = unique_sorted("Vehicle_driver_relation")
exp_options = unique_sorted("Driving_experience")
lane_options = unique_sorted("Lanes_or_Medians")
junc_options = unique_sorted("Types_of_Junction")
surface_options = unique_sorted("Road_surface_type")
light_options = unique_sorted("Light_conditions")
weather_options = unique_sorted("Weather_conditions")
collision_options = unique_sorted("Type_of_collision")
veh_move_options = unique_sorted("Vehicle_movement")
ped_move_options = unique_sorted("Pedestrian_movement")
cause_options = unique_sorted("Cause_of_accident")

# =========================
# Centered Form Layout
# =========================
left_spacer, form_col, right_spacer = st.columns([1, 3, 1])

with form_col:
    with st.form("accident_form"):

        st.markdown("### 👤 Driver Information")
        c1, c2 = st.columns(2)
        with c1:
            age_band = st.selectbox("Age band of driver", age_options)
            sex = st.selectbox("Sex of driver", sex_options)
        with c2:
            edu = st.selectbox("Educational level", edu_options)
            experience = st.selectbox("Driving experience", exp_options)

        st.markdown("---")
        st.markdown("### 🚗 Vehicle & Relation")
        c3, c4 = st.columns(2)
        with c3:
            relation = st.selectbox("Vehicle-driver relation", rel_options)
            veh_move = st.selectbox("Vehicle movement", veh_move_options)
        with c4:
            ped_move = st.selectbox("Pedestrian movement", ped_move_options)

        st.markdown("---")
        st.markdown("### 🛣️ Road & Junction Conditions")
        c5, c6 = st.columns(2)
        with c5:
            lanes = st.selectbox("Lanes or medians", lane_options)
            junction = st.selectbox("Type of junction", junc_options)
        with c6:
            surface = st.selectbox("Road surface type", surface_options)

        st.markdown("---")
        st.markdown("### 🌤️ Environment & Collision")
        c7, c8 = st.columns(2)
        with c7:
            light = st.selectbox("Light conditions", light_options)
            weather = st.selectbox("Weather conditions", weather_options)
        with c8:
            collision = st.selectbox("Type of collision", collision_options)
            cause = st.selectbox("Cause of accident", cause_options)

        st.markdown("---")
        submitted = st.form_submit_button("🔍 Predict Accident Severity")

# =========================
# On Submit -> Predict
# =========================
if submitted:
    user_input = {
        "Age_band_of_driver": age_band,
        "Sex_of_driver": sex,
        "Educational_level": edu,
        "Vehicle_driver_relation": relation,
        "Driving_experience": experience,
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

    input_encoded = preprocess_user_input(user_input)
    pred = model.predict(input_encoded)[0]

    severity_map = {
        0: "Fatal injury",
        1: "Serious injury",
        2: "Slight injury"
    }

    severity_text = severity_map.get(pred, "Unknown")

    st.markdown("### 🧮 Prediction Result")
    st.success(f"Predicted Accident Severity: **{severity_text}** (class {pred})")

    # Optional explanation
    st.info(
        "⚠️ This prediction is based on historical data and a machine learning model. "
        "It should be used for analysis and educational purposes, not as a definitive decision-making tool."
    )

st.markdown('</div>', unsafe_allow_html=True)

