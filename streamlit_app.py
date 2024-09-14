import streamlit as st
import pickle
import numpy as np

# Load the models and scaler from the pickle file
with open('models.pkl', 'rb') as file:
    loaded_models = pickle.load(file)

# Access individual models and scaler
rf_model_loaded = loaded_models["random_forest"]
knn_model_loaded = loaded_models["knn"]
svm_model_loaded = loaded_models["SVM"]
scaler_loaded = loaded_models["scaler"]

# Define the app
st.set_page_config(
    page_title="Parkinson's Disease Prediction",
    page_icon="🧠",
    layout="wide",  # Use wide layout to maximize space
    initial_sidebar_state="collapsed"
)

st.title("🧠 Parkinson's Disease Prediction")

# Introduction section
st.markdown("""
    ### Welcome to the Parkinson's Disease Prediction App!
    This application uses **Machine Learning** models to predict the likelihood of Parkinson's Disease 
    based on specific voice measurements. Simply enter the patient's details, and let our models do the rest!
""")


# Main input section
st.markdown("## 🔍 Input Patient's Voice Measurements")
col1, col2, col3 = st.columns(3)

# Add inputs to the columns
with col1:
    input_data = []
    input_data.append(st.number_input("MDVP:Jitter(Abs)", min_value=0.0, format="%.4f"))
    input_data.append(st.number_input("MDVP:Shimmer", min_value=0.0, format="%.4f"))
    input_data.append(st.number_input("MDVP:Shimmer(dB)", min_value=0.0, format="%.2f"))
    input_data.append(st.number_input("Shimmer:APQ3", min_value=0.0, format="%.4f"))

with col2:
    input_data.append(st.number_input("Shimmer:APQ5", min_value=0.0, format="%.4f"))
    input_data.append(st.number_input("MDVP:APQ", min_value=0.0, format="%.4f"))
    input_data.append(st.number_input("Shimmer:DDA", min_value=0.0, format="%.4f"))

with col3:
    input_data.append(st.number_input("spread1", min_value=-10.0, max_value=10.0, format="%.6f"))
    input_data.append(st.number_input("spread2", min_value=0.0, format="%.6f"))
    input_data.append(st.number_input("PPE", min_value=0.0, format="%.6f"))
# Convert input data to numpy array and scale it
input_data = np.array(input_data).reshape(1, -1)
input_data_scaled = scaler_loaded.transform(input_data)

# Predictions
rf_prediction = rf_model_loaded.predict(input_data_scaled)[0]
knn_prediction = knn_model_loaded.predict(input_data_scaled)[0]
svm_prediction = svm_model_loaded.predict(input_data_scaled)[0]

# Interpretation of predictions
rf_result = "🟢 Parkinsons" if rf_prediction == 1 else "🟢 Healthy"
knn_result = "🟢 Parkinsons" if knn_prediction == 1 else "🟢 Healthy"
svm_result = "🟢 Parkinsons" if svm_prediction == 1 else "🟢 Healthy"

# Display predictions
st.markdown("## 🧾 Predictions")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"**Random Forest:** {rf_result}")
with col2:
    st.markdown(f"**KNN:** {knn_result}")
with col3:
    st.markdown(f"**SVM:** {svm_result}")

# Display model accuracies
st.markdown("## 📊 Model Accuracies")
rf_accuracy = 86.44  # Example accuracy; replace with actual accuracy
knn_accuracy = 89.83  # Example accuracy; replace with actual accuracy
svm_accuracy = 93.22  # Example accuracy; replace with actual accuracy

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"### Random Forest: {rf_accuracy}%")
with col2:
    st.markdown(f"### KNN: {knn_accuracy}%")
with col3:
    st.markdown(f"### SVM: {svm_accuracy}%")

