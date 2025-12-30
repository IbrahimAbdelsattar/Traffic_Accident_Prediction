# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os

st.set_page_config(page_title="Traffic Accident Prediction", layout="centered")

MODEL_PATHS = ["models/model.pkl", "model.pkl", "models/model.joblib", "model.joblib"]

@st.cache_resource
def load_model():
    for p in MODEL_PATHS:
        if os.path.exists(p):
            try:
                # try joblib first
                model = joblib.load(p)
                return model, p
            except Exception:
                try:
                    with open(p, "rb") as f:
                        model = pickle.load(f)
                    return model, p
                except Exception:
                    st.warning(f"Found {p} but couldn't load it. Ensure it is a pickled/sklearn model.")
    return None, None

def get_feature_names(model):
    # scikit-learn 1.0+ uses feature_names_in_
    if hasattr(model, "feature_names_in_"):
        try:
            return list(model.feature_names_in_)
        except Exception:
            pass
    # some pipelines keep feature names in named_steps or preprocessor; user may need to provide features.txt
    return None

def predict_dataframe(model, df):
    # Ensure ordering if model has feature_names_in_
    fn = get_feature_names(model)
    if fn:
        missing = [c for c in fn if c not in df.columns]
        if missing:
            raise ValueError(f"Input is missing features: {missing}")
        df = df[fn]
    # try predict_proba else predict
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df)
        preds = model.predict(df)
        return pd.DataFrame({
            "prediction": preds,
            "probability": [p.max() for p in probs]
        })
    else:
        preds = model.predict(df)
        return pd.DataFrame({"prediction": preds})

st.title("Traffic Accident Prediction")
st.write("A small Streamlit UI to run predictions with your trained model. Upload a model at `models/model.pkl` or `model.pkl` in the repository, or use the file uploader below.")

model, model_path = load_model()
if model is None:
    st.error("No model file found. Place a pickled sklearn model at one of: " + ", ".join(MODEL_PATHS))
    st.info("If you don't have a saved model yet, train/export one and save it as `models/model.pkl` or `model.pkl`. A typical way: `joblib.dump(your_model, 'models/model.pkl')`.")
else:
    st.success(f"Loaded model from: {model_path}")

    # attempt to get feature names
    features = get_feature_names(model)
    if features:
        st.sidebar.write("Detected feature names for the model:")
        st.sidebar.write(features)
    else:
        st.sidebar.info("Feature names not detected automatically. You can upload a CSV for batch prediction or paste a single-row CSV/JSON with matching columns.")

    mode = st.radio("Mode", ("Single prediction (manual)", "Batch prediction (CSV upload)"))

    if mode.startswith("Single"):
        if features:
            st.subheader("Enter feature values")
            inputs = {}
            with st.form("single_form"):
                for f in features:
                    # create numeric input; if column name contains 'hour' or 'time', allow float; otherwise float default
                    inputs[f] = st.number_input(label=f, format="%.6f", key=f)
                submitted = st.form_submit_button("Predict")
            if submitted:
                df = pd.DataFrame([inputs])
                try:
                    out = predict_dataframe(model, df)
                    st.write("Prediction result")
                    st.table(out)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
        else:
            st.subheader("Provide one-row input (CSV row or JSON)")
            st.info("Enter a single-row CSV header+row or a JSON object with feature names as keys.")
            raw = st.text_area("CSV row (header + single row) or JSON", height=200, placeholder='e.g.\ncol1,col2,col3\n1,2,3\n\nor\n{"col1":1, "col2":2, "col3":3}')
            if st.button("Predict from text"):
                try:
                    if raw.strip().startswith("{"):
                        obj = pd.json_normalize(pd.read_json(raw, typ='series'))
                        df = pd.DataFrame([obj])
                    else:
                        from io import StringIO
                        df = pd.read_csv(StringIO(raw))
                        # if user gave header+row it's fine; if just values, this may fail
                    out = predict_dataframe(model, df)
                    st.write("Prediction result")
                    st.table(out)
                except Exception as e:
                    st.error(f"Failed to parse or predict: {e}")

    else:
        st.subheader("Upload CSV for batch prediction")
        uploaded = st.file_uploader("Upload a CSV with columns matching training features", type=["csv"])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                out = predict_dataframe(model, df)
                result = pd.concat([df.reset_index(drop=True), out.reset_index(drop=True)], axis=1)
                st.success("Predictions completed")
                st.dataframe(result)
                csv = result.to_csv(index=False).encode('utf-8')
                st.download_button("Download results CSV", csv, "predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.markdown("---")
st.write("Notes:")
st.write("- Make sure the model expects the same feature columns and preprocessing used at training time.")
st.write("- If your model is a pipeline (preprocessor + estimator), saving the whole pipeline with joblib/pickle is recommended.")
st.write("- To deploy on Streamlit Cloud: push this repo and connect it in https://share.streamlit.io/ (see README).")
