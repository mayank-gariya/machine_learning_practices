import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# --- Re-initialize model and data for self-containment ---
# Load the full dataset to get feature names and for re-training
df_full = load_breast_cancer(as_frame=True).frame
X = df_full.drop('target', axis=1)
y = df_full['target']

# Split data - important to use the same random_state as before
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the StandardScaler (though not used directly in this model's prediction path)
scaler = StandardScaler()
scaler.fit(X_train)

# Initialize and train the GradientBoostingClassifier with best parameters
# Best trial parameters: {'classifier': 'GradientBoosting', 'n_estimators': 166, 'learning_rate': 0.06015407899366217, 'max_depth': 17, 'min_samples_split': 7, 'min_samples_leaf': 5}
model_params = {
    'n_estimators': 166,
    'learning_rate': 0.06015407899366217,
    'max_depth': 17,
    'min_samples_split': 7,
    'min_samples_leaf': 5,
    'random_state': 42 # Ensure reproducibility
}
model = GradientBoostingClassifier(**model_params)
model.fit(X_train, y_train) # Train on the original (unscaled) X_train

# --- Streamlit App Layout ---
st.set_page_config(layout="wide") # Use wide layout
st.title("Breast Cancer Prediction App")
st.write("Enter the values for the 30 features to get a prediction for Malignant (0) or Benign (1).")
st.markdown("---")

# Create input fields for each feature dynamically
feature_names = X.columns
input_data = {}

st.header("Input Features")
cols = st.columns(3) # 3 columns for inputs

for i, feature in enumerate(feature_names):
    with cols[i % 3]: # Cycle through columns
        # Use the mean of the training data as a default value
        default_value = float(X_train[feature].mean())
        input_data[feature] = st.number_input(f"{feature}", value=default_value, key=f"input_{feature}")

input_df = pd.DataFrame([input_data])

st.markdown("---")
st.subheader("Review Input Values:")
st.dataframe(input_df)

st.markdown("---")
if st.button("Predict"):
    # No scaling applied here, consistent with the last fitted model in the notebook (cell rbV-gGIM7-sW)
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Prediction Result:")
    if prediction[0] == 0:
        st.error(f"The model predicts: **Malignant** (Probability: {prediction_proba[0][0]:.4f})")
    else:
        st.success(f"The model predicts: **Benign** (Probability: {prediction_proba[0][1]:.4f})")

    st.info("A classification of 0 indicates Malignant, and 1 indicates Benign.")
