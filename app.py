import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime as dt

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="UK Traffic Accident Severity Prediction",
    page_icon="🚦",
    layout="wide"
)

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    # غيّر اسم الملف حسب الموديل بتاعك
    with open("lightgbm_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# =========================
# Sidebar - Project Info
# =========================
st.sidebar.title("📊 Project Overview")
st.sidebar.markdown(
    """
    ### UK Traffic and Accidents Dataset (2000–2016)
    This app uses a machine learning model trained on UK police-reported accident data (2005–2014)  
    to **predict accident severity or risk patterns** based on:
    
    - Traffic & road features  
    - Weather & light conditions  
    - Urban/rural characteristics  
    - Date & time information  
    
    Data source: UK Department of Transport (Open Government Licence).
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 Built by: **Ibrahim Abdelsattar**")

# =========================
# Main Title
# =========================
st.title("🚦 UK Road Accident Risk / Severity Prediction")
st.markdown(
    """
    Use this interface to simulate a road accident scenario in the UK and get a model-based prediction.  
    Fill in the details about **location, road, environment, and time**, then click **Predict**.
    """
)

# =========================
# Centered Input Form
# =========================
left_col, center_col, right_col = st.columns([1, 2, 1])

with center_col:
    st.markdown("### 📝 Accident Details")

    with st.form("prediction_form"):

        # ---------- Location ----------
        st.subheader("📍 Location")
        col_loc1, col_loc2 = st.columns(2)
        with col_loc1:
            Location_Easting_OSGR = st.number_input(
                "Location Easting OSGR",
                value=443940.0,
                help="OSGR easting coordinate of the accident location"
            )
            Longitude = st.number_input(
                "Longitude",
                value=-1.349860,
                format="%.6f"
            )
        with col_loc2:
            Location_Northing_OSGR = st.number_input(
                "Location Northing OSGR",
                value=262705.0,
                help="OSGR northing coordinate of the accident location"
            )
            Latitude = st.number_input(
                "Latitude",
                value=52.249074,
                format="%.6f"
            )

        st.markdown("---")

        # ---------- Date & Time ----------
        st.subheader("⏱ Date & Time")

        default_date = dt.date(2010, 1, 1)
        accident_date = st.date_input("Date of Accident", value=default_date)

        default_time = dt.time(17, 0)
        accident_time = st.time_input("Time of Accident", value=default_time)

        # مشتقات التاريخ والوقت
        Year = accident_date.year
        Month = accident_date.month
        Day = accident_date.day
        Day_of_Week = accident_date.strftime("%A")  # e.g. Monday, Tuesday...

        Hour = accident_time.hour
        Minute = accident_time.minute

        # cyclical encoding للوقت (حسب الساعة فقط كما في الداتا)
        angle = 2 * np.pi * Hour / 24
        Time_sin = float(np.sin(angle))
        Time_cos = float(np.cos(angle))

        st.caption(
            f"📌 Derived features — Year: {Year}, Month: {Month}, Day: {Day}, "
            f"Day of Week: {Day_of_Week}, Hour: {Hour}, Minute: {Minute}"
        )

        st.markdown("---")

        # ---------- Road & Traffic ----------
        st.subheader("🚗 Road & Traffic")

        col_rt1, col_rt2 = st.columns(2)
        with col_rt1:
            Police_Force = st.number_input(
                "Police Force Code",
                min_value=1,
                max_value=99,
                value=1,
                step=1,
                help="Numeric code identifying the police authority"
            )

            Number_of_Vehicles = st.number_input(
                "Number of Vehicles Involved",
                min_value=1,
                max_value=70,
                value=2,
                step=1
            )

            Number_of_Casualties = st.number_input(
                "Number of Casualties",
                min_value=1,
                max_value=100,
                value=1,
                step=1
            )

            Speed_limit = st.selectbox(
                "Speed Limit (mph)",
                options=[10, 15, 20, 30, 40, 50, 60, 70],
                index=3  # 30 mph by default
            )

            Road_Type = st.selectbox(
                "Road Type",
                options=[
                    "Single carriageway",
                    "Dual carriageway",
                    "Roundabout",
                    "One way street",
                    "Slip road",
                    "Unknown"
                ]
            )

        with col_rt2:
            col_rc1, col_rc2 = st.columns(2)
            with col_rc1:
                first_road_class = st.selectbox(
                    "1st Road Class",
                    options=[1, 2, 3, 4, 5, 6],
                    index=2
                )
            with col_rc2:
                first_road_number = st.number_input(
                    "1st Road Number",
                    min_value=0,
                    value=0,
                    step=1
                )

            col_sr1, col_sr2 = st.columns(2)
            with col_sr1:
                second_road_class = st.selectbox(
                    "2nd Road Class",
                    options=[-1, 1, 2, 3, 4, 5, 6],
                    index=0,
                    help="-1 means no second road"
                )
            with col_sr2:
                second_road_number = st.number_input(
                    "2nd Road Number",
                    min_value=-1,
                    value=0,
                    step=1
                )

            Junction_Control = st.selectbox(
                "Junction Control",
                options=[
                    "Giveway or uncontrolled",
                    "Automatic traffic signal",
                    "Stop Sign",
                    "Authorised person"
                ]
            )

        st.markdown("---")

        # ---------- Authorities & LSOA ----------
        st.subheader("🏢 Administrative Information")
        col_ad1, col_ad2 = st.columns(2)

        with col_ad1:
            local_auth_district = st.text_input(
                "Local Authority (District) Code",
                value="300",
                help="e.g. 300, 204, 102..."
            )

            local_auth_highway = st.text_input(
                "Local Authority (Highway) Code",
                value="E10000016",
                help="e.g. E10000016, E10000030..."
            )

        with col_ad2:
            lsoa_code = st.text_input(
                "LSOA of Accident Location",
                value="E01018648",
                help="e.g. E01018648..."
            )

        st.markdown("---")

        # ---------- Environment ----------
        st.subheader("🌦 Environment & Conditions")

        col_env1, col_env2 = st.columns(2)

        with col_env1:
            Light_Conditions = st.selectbox(
                "Light Conditions",
                options=[
                    "Daylight: Street light present",
                    "Darkness: Street lights present and lit",
                    "Darkeness: No street lighting",
                    "Darkness: Street lighting unknown",
                    "Darkness: Street lights present but unlit",
                ]
            )

            Weather_Conditions = st.selectbox(
                "Weather Conditions",
                options=[
                    "fine without high winds",
                    "raining without high winds",
                    "other",
                    "unknown",
                    "raining with high winds",
                    "fine with high winds",
                    "snowing without high winds",
                    "fog or mist",
                    "snowing with high winds",
                ]
            )

            Road_Surface_Conditions = st.selectbox(
                "Road Surface Conditions",
                options=[
                    "Dry",
                    "Wet/Damp",
                    "Frost/Ice",
                    "Snow",
                    "Flood (Over 3cm of water)",
                ]
            )

        with col_env2:
            Ped_Human = st.selectbox(
                "Pedestrian Crossing - Human Control",
                options=[
                    "None within 50 metres",
                    "Control by other authorised person",
                    "Control by school crossing patrol",
                ]
            )

            Ped_Physical = st.selectbox(
                "Pedestrian Crossing - Physical Facilities",
                options=[
                    "No physical crossing within 50 meters",
                    "Pedestrian phase at traffic signal junction",
                    "non-junction pedestrian crossing",
                    "Zebra crossing",
                    "Central refuge",
                    "Footbridge or subway",
                ]
            )

            Urban_or_Rural_Area = st.selectbox(
                "Urban or Rural Area",
                options=[1, 2, 3],
                index=0,
                help="1=Urban, 2=Rural, 3=Unallocated"
            )

            Did_Police_Officer_Attend_Scene_of_Accident = st.selectbox(
                "Did a Police Officer Attend the Scene?",
                options=["Yes", "No"]
            )

        # ---------- Submit Button ----------
        submitted = st.form_submit_button("🚀 Predict")

    # =========================
    # Build Feature Row & Predict
    # =========================
    if submitted:
        # Date in YYYY-MM-DD format (as in your dataset)
        Date_str = accident_date.strftime("%Y-%m-%d")

        # IMPORTANT: لازم الأعمدة تكون بنفس أسماء وترتيب التدريب
        input_data = {
            "Location_Easting_OSGR": Location_Easting_OSGR,
            "Location_Northing_OSGR": Location_Northing_OSGR,
            "Longitude": Longitude,
            "Latitude": Latitude,
            "Police_Force": Police_Force,
            "Number_of_Vehicles": Number_of_Vehicles,
            "Number_of_Casualties": Number_of_Casualties,
            "Date": Date_str,
            "Day_of_Week": Day_of_Week,
            "Time": accident_time.strftime("%H:%M"),
            "Local_Authority_(District)": local_auth_district,
            "Local_Authority_(Highway)": local_auth_highway,
            "1st_Road_Class": first_road_class,
            "1st_Road_Number": first_road_number,
            "Road_Type": Road_Type,
            "Speed_limit": Speed_limit,
            "Junction_Control": Junction_Control,
            "2nd_Road_Class": second_road_class,
            "2nd_Road_Number": second_road_number,
            "Pedestrian_Crossing-Human_Control": Ped_Human,
            "Pedestrian_Crossing-Physical_Facilities": Ped_Physical,
            "Light_Conditions": Light_Conditions,
            "Weather_Conditions": Weather_Conditions,
            "Road_Surface_Conditions": Road_Surface_Conditions,
            "Urban_or_Rural_Area": Urban_or_Rural_Area,
            "Did_Police_Officer_Attend_Scene_of_Accident": Did_Police_Officer_Attend_Scene_of_Accident,
            "LSOA_of_Accident_Location": lsoa_code,
            "Year": Year,
            "Month": Month,
            "Day": Day,
            "Hour": Hour,
            "Minute": Minute,
            "Time_sin": Time_sin,
            "Time_cos": Time_cos,
        }

        # Convert to DataFrame (single row)
        input_df = pd.DataFrame([input_data])

        st.markdown("### 🔍 Input Summary")
        st.dataframe(input_df)

        # Make prediction
        try:
            y_pred = model.predict(input_df)[0]

            st.markdown("## 🧠 Model Prediction")
            st.success(f"Predicted output: **{y_pred}**")

            # لو الموديل classifier وعنده predict_proba
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_df)[0]
                prob_df = pd.DataFrame(
                    [probs],
                    columns=[f"Class {c}" for c in range(len(probs))]
                )
                st.markdown("### 📈 Class Probabilities")
                st.dataframe(prob_df)

        except Exception as e:
            st.error("❌ An error occurred while making the prediction.")
            st.exception(e)
