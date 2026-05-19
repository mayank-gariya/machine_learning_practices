import streamlit as st
import pandas as pd
import numpy as np
import os 
from pathlib import Path
# 1. Page Configuration
st.set_page_config(
    page_title="Steel Defects Detector",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 Steel Plates Faults Detection Dashboard")
st.markdown("""
This self-contained application utilizes an **Ensemble Stacking Machine Learning model** 
(Random Forest + XGBoost) to predict manufacturing defects directly from steel plate dimensions.
""")

st.write("---")


APP_DIR = Path(__file__).resolve().parent
P2_DIR = APP_DIR.parent
MODELS_DIR = P2_DIR / "models"

# 2. Load the Pipeline Artifacts Locally
MODEL_PATH = str(MODELS_DIR / "model.joblib")          
SCALER_PATH = str(MODELS_DIR / "scaler.joblib")
ENCODER_PATH = str(MODELS_DIR / "label_encoder.joblib")

@st.cache_resource
def load_pipeline():
    """Loads and caches the model artifacts so they don't reload on every slider move."""
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(ENCODER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
        return model, scaler, label_encoder
    else:
        st.error("❌ Pipeline artifacts missing in 'models/' directory. Please make sure you ran your training script first!")
        return None, None, None

model, scaler, label_encoder = load_pipeline()

# 3. Setup Columns for Input Layout
col1, col2, col3 = st.columns(3)

with col1:
    st.header("📐 Geometry & Coordinates")
    x_min = st.number_input("X Minimum Coordinate", value=42.0)
    x_max = st.number_input("X Maximum Coordinate", value=134.0)
    y_min = st.number_input("Y Minimum Coordinate", value=872000.0)
    y_max = st.number_input("Y Maximum Coordinate", value=872200.0)
    pixels_area = st.number_input("Pixels Area", value=500.0)
    x_perimeter = st.number_input("X Perimeter", value=35.0)
    y_perimeter = st.number_input("Y Perimeter", value=40.0)
    length_conveyer = st.number_input("Length of Conveyer", value=1350.0)

with col2:
    st.header("💡 Luminosity & Indexing")
    sum_luminosity = st.number_input("Sum of Luminosity", value=55000.0)
    min_luminosity = st.number_input("Minimum Luminosity", value=80.0)
    max_luminosity = st.number_input("Maximum Luminosity", value=145.0)
    steel_thickness = st.number_input("Steel Plate Thickness (mm)", value=40.0)
    edges_index = st.slider("Edges Index", 0.0, 1.0, 0.1)
    empty_index = st.slider("Empty Index", 0.0, 1.0, 0.3)
    square_index = st.slider("Square Index", 0.0, 1.0, 0.5)

with col3:
    st.header("📊 Logistical & Shape Ratios")
    outside_x_index = st.slider("Outside X Index", 0.0, 1.0, 0.02)
    edges_x_index = st.slider("Edges X Index", 0.0, 1.0, 0.6)
    edges_y_index = st.slider("Edges Y Index", 0.0, 1.0, 0.9)
    outside_global_index = st.selectbox("Outside Global Index", [0.0, 0.5, 1.0], index=1)
    log_areas = st.number_input("Log of Areas", value=2.7)
    log_x_index = st.number_input("Log X Index", value=1.3)
    log_y_index = st.number_input("Log Y Index", value=1.4)
    orientation_index = st.slider("Orientation Index", -1.0, 1.0, -0.2)
    luminosity_index = st.slider("Luminosity Index", -1.0, 1.0, -0.1)
    sigmoid_areas = st.slider("Sigmoid of Areas", 0.0, 1.0, 0.5)
    
    st.markdown("**Steel Type Designation:**")
    steel_a300 = st.checkbox("Type A300", value=1)
    steel_a400 = 0 if steel_a300 else 1

st.write("---")

# 4. Trigger Prediction Logic (Now entirely in-app)
if st.button("🚀 Analyze Steel Plate Defect", use_container_width=True):
    if model is not None:
        with st.spinner("Processing features through Stacking Ensemble..."):
            try:
                # Structure the input data into a DataFrame
                input_data = {
                    "X_Minimum": x_min, "X_Maximum": x_max, "Y_Minimum": y_min, "Y_Maximum": y_max,
                    "Pixels_Areas": pixels_area, "X_Perimeter": x_perimeter, "Y_Perimeter": y_perimeter,
                    "Sum_of_Luminosity": sum_luminosity, "Minimum_of_Luminosity": min_luminosity,
                    "Maximum_of_Luminosity": max_luminosity, "Length_of_Conveyer": length_conveyer,
                    "TypeOfSteel_A300": int(steel_a300), "TypeOfSteel_A400": int(steel_a400),
                    "Steel_Plate_Thickness": steel_thickness, "Edges_Index": edges_index, "Empty_Index": empty_index,
                    "Square_Index": square_index, "Outside_X_Index": outside_x_index, "Edges_X_Index": edges_x_index,
                    "Edges_Y_Index": edges_y_index, "Outside_Global_Index": outside_global_index, "LogOfAreas": log_areas,
                    "Log_X_Index": log_x_index, "Log_Y_Index": log_y_index, "Orientation_Index": orientation_index,
                    "Luminosity_Index": luminosity_index, "SigmoidOfAreas": sigmoid_areas
                }
                input_df = pd.DataFrame([input_data])
                
                # Apply the explicit Feature Engineering matching train.py
                input_df['Fault_Width'] = np.abs(input_df['X_Maximum'] - input_df['X_Minimum'])
                input_df['Fault_Length'] = np.abs(input_df['Y_Maximum'] - input_df['Y_Minimum'])
                input_df['Fault_Area_Estimate'] = input_df['Fault_Width'] * input_df['Fault_Length']
                input_df['Thickness_to_Area'] = input_df['Steel_Plate_Thickness'] / (input_df['Pixels_Areas'] + 1e-5)
                
                # Scale features using the saved production scaler
                scaled_features = scaler.transform(input_df)
                
                # Run predictions and extract probabilities
                prediction_encoded = model.predict(scaled_features)
                probabilities = model.predict_proba(scaled_features)[0]
                
                # Decode integer back to string fault label
                predicted_class_name = label_encoder.inverse_transform(prediction_encoded)[0]
                prediction_label = predicted_class_name.upper().replace("_", " ")
                confidence = float(np.max(probabilities)) * 100
                
                # Display Results UI
                st.success("### Analysis Complete!")
                metric_col1, metric_col2 = st.columns(2)
                metric_col1.metric(label="Predicted Fault Category", value=prediction_label)
                metric_col2.metric(label="Ensemble Model Confidence", value=f"{confidence:.2f}%")
                
                # Structural context hints
                if "SCATCH" in prediction_label or "SCRATCH" in prediction_label:
                    st.warning("⚠️ High structural risk. Inspect machine rollers for mechanical abrasive points.")
                elif "BUMPS" in prediction_label or "PASTRY" in prediction_label:
                    st.info("ℹ️ Surface irregularity detected. Check raw slab cooling rates.")
                    
            except Exception as e:
                st.error(f"Prediction failed inside Streamlit runtime: {str(e)}")
