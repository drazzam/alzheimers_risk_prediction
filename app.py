import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Load the saved model and preprocessing objects
@st.cache_resource
def load_model():
    model = joblib.load('best_model.joblib')
    imputer = joblib.load('imputer.joblib')
    scaler = joblib.load('scaler.joblib')
    selector = joblib.load('selector.joblib')
    return model, imputer, scaler, selector

model, imputer, scaler, selector = load_model()

# Load the feature names
@st.cache_data
def load_features():
    with open('top_features.txt', 'r') as f:
        return [line.strip() for line in f]

top_features = load_features()

# Create a dictionary to map original feature names to user-friendly names
feature_name_mapping = {
    "Right-Inf-Lat-Vent_volume": "Right Inferior Lateral Ventricle Volume (in mm³)",
    "rh_middletemporal_thickness": "Right Hemisphere Middle Temporal Gyrus Thickness (in mm)",
    "rh_temporalpole_thickness": "Right Hemisphere Temporal Pole Thickness (in mm)",
    "lh_lateraloccipital_thickness": "Left Hemisphere Lateral Occipital Cortex Thickness (in mm)",
    "rh_lateraloccipital_thickness": "Right Hemisphere Lateral Occipital Cortex Thickness (in mm)",
    "WM-hypointensities_volume": "White Matter Hypointensities Volume (in mm³)",
    "rh_parahippocampal_volume": "Right Hemisphere Parahippocampal Gyrus Volume (in mm³)",
    "3rd-Ventricle_volume": "Third Ventricle Volume (in mm³)",
    "rh_fusiform_thickness": "Right Hemisphere Fusiform Gyrus Thickness (in mm)",
    "lh_supramarginal_thickness": "Left Hemisphere Supramarginal Gyrus Thickness (in mm)"
}

# Get all feature names from the imputer
all_features = imputer.feature_names_in_

# Function to make predictions
def predict_ad_risk(input_data):
    # Ensure all features are present
    for feature in all_features:
        if feature not in input_data.columns:
            input_data[feature] = imputer.statistics_[list(all_features).index(feature)]

    # Reorder columns to match the order used during training
    input_data = input_data[all_features]

    input_imputed = imputer.transform(input_data)
    input_selected = selector.transform(input_imputed)
    input_scaled = scaler.transform(input_selected)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0, 1]
    return prediction[0], probability

# Function to create a gauge chart
def create_gauge_chart(value, title):
    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(projection='polar'))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, 2*np.pi, 4, endpoint=False))
    ax.set_xticklabels(['0%', '25%', '50%', '75%'])
    ax.spines['polar'].set_visible(False)
    ax.grid(False)

    # Create the gauge
    gauge = ax.bar(0, value, width=np.pi*2, bottom=0.0, alpha=0.5, color='lightblue')

    # Add value text
    ax.text(0, 0, f'{value:.1%}', ha='center', va='center', fontsize=20, fontweight='bold')

    # Add title
    plt.title(title, y=1.2, fontsize=14)

    return fig

# Streamlit app
st.title("Alzheimer's Disease Risk Prediction")
st.subheader("Enter the values for the following features:")

# Create input fields
user_input = {}
for feature in top_features:
    user_input[feature] = st.number_input(
        feature_name_mapping.get(feature, feature),
        value=0.0,
        format="%.2f"
    )

if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    
    # Make prediction
    prediction, probability = predict_ad_risk(input_df)
    confidence = max(probability, 1-probability)

    # Create and display gauge charts
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(create_gauge_chart(probability, "Probability of AD"))
    with col2:
        st.pyplot(create_gauge_chart(confidence, "Prediction Confidence"))

    # Display interpretation
    if probability > 0.7:
        st.error("Warning: The model indicates a high risk of developing Alzheimer's Disease. Please consult with a healthcare professional.")
    elif probability > 0.3:
        st.warning("Info: The model indicates a moderate risk of developing Alzheimer's Disease. Regular check-ups are recommended.")
    else:
        st.success("Good news: The model indicates a low risk of developing Alzheimer's Disease. Maintain a healthy lifestyle to keep the risk low.")

# Add information about the model
st.markdown("""
## About This Model

This model uses Support Vector Machine (SVM) algorithm to predict the risk of Alzheimer's Disease based on brain imaging features within the next 2 - 3 years while the individual is totally asymptomatic and has no cognitive decline.

The prediction is based on the 10 most important features identified by our model performance.

**Disclaimer:** While this tool can provide valuable insights, it should not be considered as a definitive diagnosis. This model is for testing and development purposes.
""")
