import streamlit as st
import pandas as pd
import xgboost as xgb

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Road Accident Severity Predictor",
    page_icon="🛣️",
    layout="wide"
)

# بسيط لتوسيط البوكس
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

# -----------------------------
# Load Data & Model
# -----------------------------
df = pd.read_csv("cleaned.csv")

# الأعمدة التفسيرية (نفس اللي استخدمتها في التدريب)
FEATURE_COLS = [
    'Age_band_of_driver', 'Sex_of_driver', 'Educational_level',
    'Vehicle_driver_relation', 'Driving_experience', 'Lanes_or_Medians',
    'Types_of_Junction', 'Road_surface_type', 'Light_conditions',
    'Weather_conditions', 'Type_of_collision', 'Vehicle_movement',
    'Pedestrian_movement', 'Cause_of_accident'
]

# نفس الترميز اللي اتدرّب عليه الموديل: get_dummies(drop_first=True)
X_train_like = pd.get_dummies(df[FEATURE_COLS], drop_first=True)
feature_columns = X_train_like.columns  # الأعمدة اللي الموديل متعود عليها

# Load XGBoost model from JSON
model = xgb.XGBClassifier()
model.load_model("xgboost_model.json")


# -----------------------------
# Helper: preprocess user input
# -----------------------------
def preprocess_user_input(user_input_dict):
    """
    يحوّل الـ inputs الcategorical لـ one-hot encoded
    بنفس شكل الداتا وقت التدريب، ويعمل reindex على الأعمدة.
    """
    input_df = pd.DataFrame([user_input_dict])
    input_encoded = pd.get_dummies(input_df, drop_first=True)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
    return input_encoded


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 1rem;">
        <h1>🛣️ Road Accident Severity Predictor</h1>
        <p style="font-size: 0.95rem; color: #555;">
            Estimate the severity of a road accident using driver, vehicle, road, and environment information.
            The model is trained on real accident data from Addis Ababa.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-block">', unsafe_allow_html=True)
st.subheader("📋 Enter Accident Details")

# -----------------------------
# Options from dataset
# -----------------------------
def ops(col):
    return sorted(df[col].dropna().unique().tolist())

age_options        = ops("Age_band_of_driver")
sex_options        = ops("Sex_of_driver")
edu_options        = ops("Educational_level")
rel_options        = ops("Vehicle_driver_relation")
exp_options        = ops("Driving_experience")
lane_options       = ops("Lanes_or_Medians")
junc_options       = ops("Types_of_Junction")
surface_options    = ops("Road_surface_type")
light_options      = ops("Light_conditions")
weather_options    = ops("Weather_conditions")
collision_options  = ops("Type_of_collision")
veh_move_options   = ops("Vehicle_movement")
ped_move_options   = ops("Pedestrian_movement")
cause_options      = ops("Cause_of_accident")

# -----------------------------
# Centered Form
# -----------------------------
_, form_col, _ = st.columns([1, 3, 1])

with form_col:
    st.markdown("### 👤 Driver & Vehicle Info")
    c1, c2 = st.columns(2)
    with c1:
        age_band = st.selectbox("Age band of driver", age_options)
        sex      = st.selectbox("Sex of driver", sex_options)
        edu      = st.selectbox("Educational level", edu_options)
    with c2:
        relation   = st.selectbox("Vehicle-driver relation", rel_options)
        experience = st.selectbox("Driving experience", exp_options)

    st.markdown("---")
    st.markdown("### 🛣️ Road & Environment")
    c3, c4 = st.columns(2)
    with c3:
        lanes    = st.selectbox("Lanes or medians", lane_options)
        junction = st.selectbox("Type of junction", junc_options)
        surface  = st.selectbox("Road surface type", surface_options)
    with c4:
        light   = st.selectbox("Light conditions", light_options)
        weather = st.selectbox("Weather conditions", weather_options)

    st.markdown("---")
    st.markdown("### 🚗 Movements & Collision")
    c5, c6 = st.columns(2)
    with c5:
        veh_move = st.selectbox("Vehicle movement", veh_move_options)
        ped_move = st.selectbox("Pedestrian movement", ped_move_options)
    with c6:
        collision = st.selectbox("Type of collision", collision_options)
        cause     = st.selectbox("Cause of accident", cause_options)

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Accident Severity")

# -----------------------------
# Prediction
# -----------------------------
if predict_btn:
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
    severity_text = severity_map.get(int(pred), "Unknown")

    st.markdown("### 🧮 Prediction Result")
    st.success(f"Predicted Accident Severity: **{severity_text}** (class {pred})")
    st.info(
        "⚠️ This prediction is based on a machine learning model trained on historical data. "
        "It should be used for analysis and educational purposes only."
    )

st.markdown('</div>', unsafe_allow_html=True)

