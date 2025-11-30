import streamlit as st
import pandas as pd
import pickle

# =========================
# Basic Page Config
# =========================
st.set_page_config(
    page_title="Road Accident Severity Predictor",
    page_icon="🛣️",
    layout="wide"
)

# بسيط خالص لتجميل البوكس في النص
st.markdown(
    """
    <style>
    .main-block {
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
        padding: 1.5rem;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.06);
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Load Model & Data (simple)
# =========================
df = pd.read_csv("cleaned.csv")

with open("xgboost-model.pkl", "rb") as f:
    model = pickle.load(f)

# نفس الترميز اللي اتدرّب عليه الموديل
X = pd.get_dummies(df.drop("Accident_severity", axis=1), drop_first=True)
feature_columns = X.columns

# =========================
# Helper: preprocess single input
# =========================
def preprocess_user_input(user_input_dict):
    input_df = pd.DataFrame([user_input_dict])
    input_encoded = pd.get_dummies(input_df, drop_first=True)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
    return input_encoded

# =========================
# Header
# =========================
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 1rem;">
        <h1>🛣️ Road Accident Severity Predictor</h1>
        <p style="font-size: 0.95rem; color: #555;">
            Estimate the severity of a road accident using driver, vehicle, road, and environment information.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-block">', unsafe_allow_html=True)

st.subheader("📋 Enter Accident Details")

# =========================
# Options from dataset
# =========================
def unique_sorted(col):
    return sorted(df[col].dropna().unique().tolist())

age_options        = unique_sorted("Age_band_of_driver")
sex_options        = unique_sorted("Sex_of_driver")
edu_options        = unique_sorted("Educational_level")
rel_options        = unique_sorted("Vehicle_driver_relation")
exp_options        = unique_sorted("Driving_experience")
lane_options       = unique_sorted("Lanes_or_Medians")
junc_options       = unique_sorted("Types_of_Junction")
surface_options    = unique_sorted("Road_surface_type")
light_options      = unique_sorted("Light_conditions")
weather_options    = unique_sorted("Weather_conditions")
collision_options  = unique_sorted("Type_of_collision")
veh_move_options   = unique_sorted("Vehicle_movement")
ped_move_options   = unique_sorted("Pedestrian_movement")
cause_options      = unique_sorted("Cause_of_accident")

# =========================
# Centered Form
# =========================
_, form_col, _ = st.columns([1, 3, 1])

with form_col:
    st.markdown("### 👤 Driver Information")
    c1, c2 = st.columns(2)
    with c1:
        age_band = st.selectbox("Age band of driver", age_options)
        sex      = st.selectbox("Sex of driver", sex_options)
    with c2:
        edu        = st.selectbox("Educational level", edu_options)
        experience = st.selectbox("Driving experience", exp_options)

    st.markdown("---")
    st.markdown("### 🚗 Vehicle & Pedestrian")
    c3, c4 = st.columns(2)
    with c3:
        relation = st.selectbox("Vehicle-driver relation", rel_options)
        veh_move = st.selectbox("Vehicle movement", veh_move_options)
    with c4:
        ped_move = st.selectbox("Pedestrian movement", ped_move_options)

    st.markdown("---")
    st.markdown("### 🛣️ Road & Junction")
    c5, c6 = st.columns(2)
    with c5:
        lanes    = st.selectbox("Lanes or medians", lane_options)
        junction = st.selectbox("Type of junction", junc_options)
    with c6:
        surface = st.selectbox("Road surface type", surface_options)

    st.markdown("---")
    st.markmarkdown("### 🌤️ Environment & Collision")
    c7, c8 = st.columns(2)
    with c7:
        light   = st.selectbox("Light conditions", light_options)
        weather = st.selectbox("Weather conditions", weather_options)
    with c8:
        collision = st.selectbox("Type of collision", collision_options)
        cause     = st.selectbox("Cause of accident", cause_options)

    st.markdown("---")
    if st.button("🔍 Predict Accident Severity"):
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

        X_input = preprocess_user_input(user_input)
        pred = model.predict(X_input)[0]

        severity_map = {
            0: "Fatal injury",
            1: "Serious injury",
            2: "Slight injury"
        }
        severity_text = severity_map.get(pred, "Unknown")

        st.markdown("### 🧮 Prediction Result")
        st.success(f"Predicted Accident Severity: **{severity_text}** (class {pred})")
        st.info(
            "⚠️ This prediction is for analysis and educational purposes only, "
            "and should not be used as a final decision-making tool."
        )

st.markdown('</div>', unsafe_allow_html=True)
