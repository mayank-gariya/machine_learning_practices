
import gradio as gr
import joblib
import pandas as pd
import pickle as pkl

# Load the trained Linear Regression model pipeline
# Make sure 'linear_regression_pipeline.joblib' is in the same directory as app.py
try:
    loaded_pipeline = joblib.load('LinearRegressionModel.pkl')
    print("Linear Regression model pipeline loaded successfully.")
except FileNotFoundError:
    print("Error: 'linear_regression_pipeline.joblib' not found. Make sure it's in the same directory as app.py.")
    exit()


# Define the prediction function for Gradio
def predict_car_price(name, company, year, kms_driven, fuel_type):
    # Create a DataFrame from the input data
    input_data = pd.DataFrame({
        'name': [name],
        'company': [company],
        'year': [year],
        'kms_driven': [kms_driven],
        'fuel_type': [fuel_type]
    })

    # Make prediction
    prediction = loaded_pipeline.predict(input_data)[0]
    return f"{prediction:,.2f} INR"


company_options = sorted(['Hyundai', 'Mahindra', 'Ford', 'Maruti', 'Skoda', 'Audi', 'Toyota', 'Renault', 'Honda', 'Datsun', 'Mitsubishi', 'Tata', 'Volkswagen', 'Chevrolet', 'Mini', 'BMW', 'Nissan', 'Hindustan', 'Fiat', 'Force', 'Mercedes', 'Land', 'Jaguar', 'Jeep', 'Volvo'])
fuel_type_options = sorted(['Petrol', 'Diesel', 'LPG'])
year_options = sorted(list(range(1995, 2020)), reverse=True)

inputs = [
    gr.Text(label="Car Name (e.g., 'Maruti Suzuki Swift')"),
    gr.Dropdown(choices=company_options, label="Company"),
    gr.Dropdown(choices=year_options, label="Manufacturing Year"),
    gr.Number(label="Kilometers Driven"),
    gr.Dropdown(choices=fuel_type_options, label="Fuel Type")
]

outputs = gr.Text(label="Predicted Price")

gradio_app = gr.Interface(
    fn=predict_car_price,
    inputs=inputs,
    outputs=outputs,
    title="Car Price Predictor",
    description="Enter car details to get an estimated price using a Linear Regression model."
)

# Launch the Gradio app
if __name__ == '__main__':
    gradio_app.launch(share=True)
