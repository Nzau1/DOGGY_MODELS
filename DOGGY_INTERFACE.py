import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import OneHotEncoder

# Step 1: Load Model and Encoder
models_folder = os.path.join(os.path.dirname(__file__), 'KIONGOZI')
model_path = os.path.join(models_folder, 'dog_price_predictor.pkl')
data_sample_path = os.path.join(models_folder, 'dog_price_prediction_dataset.csv')

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        background-color: #f0f2f6;
    }
    .stTitle {
        color: #2C3E50;
        font-size: 2.5em;
        text-align: center;
        padding: 20px;
        background-color: #3498DB;
        color: white;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #2980B9;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #3498DB;
    }
    .stSelectbox, .stSlider {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stSuccess {
        background-color: #27AE60;
        color: white;
        border-radius: 5px;
        padding: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model
model = joblib.load(model_path)

# Load the sample data to initialize encoder
data_sample = pd.read_csv(data_sample_path)

# Define categorical columns
categorical_cols = ['Breed', 'Location', 'Size', 'Gender']

# Initialize the encoder
encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
encoder.fit(data_sample[categorical_cols])

# Streamlit App Title
st.markdown('<h1 class="stTitle">Dog Price Prediction</h1>', unsafe_allow_html=True)
st.write("Developed by Eng. Antony")
st.write("Fill in the details below to get an estimate of the dog's price.")

# Function to process input and make predictions
def predict_dog_price(breed, location, age, gender, size, 
                      vaccination_status, pedigree_certification, 
                      training_level, health_screening, 
                      parent_champion_status, demand):
    # Create input DataFrame
    input_data = pd.DataFrame([[
        breed, location, age, gender, size, 
        vaccination_status, pedigree_certification, 
        training_level, health_screening, 
        parent_champion_status, demand
    ]], columns=[
        'Breed', 'Location', 'Age (Months)', 'Gender', 'Size', 
        'Vaccination Status', 'Pedigree Certification', 
        'Training Level', 'Health Screening', 
        'Parent Champion Status', 'Demand'
    ])
    
    try:
        # Encode categorical variables
        encoded_input = encoder.transform(input_data[categorical_cols])
        encoded_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(categorical_cols))

        # Combine with numerical data
        input_data_processed = pd.concat([input_data.drop(categorical_cols, axis=1), encoded_df], axis=1)

        # Ensure all columns match the trained model input format
        input_data_processed = input_data_processed.reindex(columns=model.feature_names_in_, fill_value=0)

        # Predict using the loaded model
        prediction = model.predict(input_data_processed)

        return f"Estimated Dog Price: Kes{prediction[0]:,.2f}"
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return "Oops! Error: Unable to make prediction. Please check your inputs."

# Streamlit Input Fields
col1, col2 = st.columns(2)

with col1:
    breed = st.selectbox(
        "Select Dog Breed:",
        options=sorted(data_sample['Breed'].unique())
    )
    
    location = st.selectbox(
        "Select Location:",
        options=sorted(data_sample['Location'].unique())
    )
    
    age = st.slider(
        "Dog Age (Months):",
        min_value=2, max_value=120, value=24, step=1
    )
    
    gender = st.selectbox(
        "Select Gender:",
        options=sorted(data_sample['Gender'].unique())
    )

with col2:
    size = st.selectbox(
        "Select Size:",
        options=sorted(data_sample['Size'].unique())
    )
    
    vaccination_status = st.selectbox(
        "Vaccination Status:",
        options=["Complete", "Incomplete"]
    )
    
    pedigree_certification = st.selectbox(
        "Pedigree Certification:",
        options=["Yes", "No"]
    )

# Additional features
st.subheader("Additional Features")
col3, col4 = st.columns(2)

with col3:
    training_level = st.selectbox(
        "Training Level:",
        options=["Advanced", "Intermediate", "Basic", "None"]
    )
    
    health_screening = st.selectbox(
        "Health Screening:",
        options=["Comprehensive", "Basic", "None"]
    )

with col4:
    parent_champion_status = st.selectbox(
        "Parent Champion Status:",
        options=["Yes", "No"]
    )
    
    demand = st.slider(
        "Demand Score:",
        min_value=1, max_value=10, value=5, step=1
    )

# Predict Button
if st.button("Check the Price"):
    if all([breed, location, age, gender, size, 
            vaccination_status, pedigree_certification, 
            training_level, health_screening, 
            parent_champion_status, demand]):
        result = predict_dog_price(
            breed, location, age, gender, size, 
            vaccination_status, pedigree_certification, 
            training_level, health_screening, 
            parent_champion_status, demand
        )
        st.success(result)
    else:
        st.error("Please fill in all the fields.")

# Footer
st.markdown("---")