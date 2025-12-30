# Deploying Traffic Accident Prediction with Streamlit

This folder contains a Streamlit app (app.py) to serve your trained model.

Files:
- `app.py` - Streamlit application.
- `requirements.txt` - Python packages required.

How the app expects your model:
- Place a serialized model at either `models/model.pkl` or `model.pkl` (joblib or pickle formats are supported).
- If your trained model is a scikit-learn Pipeline, include the preprocessing steps inside the pipeline and save the entire pipeline (recommended).
- If your model was trained with scikit-learn >= 1.0 and preserved `feature_names_in_`, the app will detect expected features automatically and render input fields. Otherwise provide CSVs matching the original training features.

Run locally:
1. Create a virtualenv and install requirements:
   python -m venv venv
   source venv/bin/activate  # Windows: venv\\Scripts\\activate
   pip install -r requirements.txt
2. Run:
   streamlit run app.py
3. Open the shown localhost URL (usually http://localhost:8501).

Deploy to Streamlit Community Cloud:
1. Push these files to the repository root (or a subfolder) in GitHub.
2. Go to https://share.streamlit.io/ and sign in with GitHub.
3. Click "New app", select the repository `IbrahimAbdelsattar/Traffic_Accident_Prediction`, branch, and path to `app.py`.
4. Click "Deploy". Streamlit will install `requirements.txt` and run `streamlit run app.py` for you.

Notes for reliable predictions:
- Ensure the model and any preprocessing expect the same columns and encodings as at training (one-hot encoded categories, label encoders, scaling, etc.).
- If your pipeline uses external files (e.g., encoders saved separately), package them in the repo and adapt `app.py` to load them.
- If feature names are not auto-detected, create a `features.txt` (one column per line) or provide a CSV header row on single-prediction mode.
