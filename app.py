import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('regression_model.pkl', 'rb'))

# Define the Streamlit app
st.title("Boston House Price Prediction")

# Define input fields for the 10 features
st.sidebar.header("Enter House Features")
crim = st.sidebar.number_input("CRIM (Per Capita Crime Rate)", value=0.1)
zn = st.sidebar.number_input("ZN (Proportion of Residential Land)", value=0.0)
indus = st.sidebar.number_input("INDUS (Proportion of Non-Retail Business Acres)", value=0.0)
rm = st.sidebar.number_input("RM (Average Number of Rooms)", value=6.0)
age = st.sidebar.number_input("AGE (Proportion of Owner-Occupied Units Built Before 1940)", value=50.0)
dis = st.sidebar.number_input("DIS (Weighted Distance to Employment Centers)", value=4.0)
tax = st.sidebar.number_input("TAX (Property Tax Rate)", value=300)
ptratio = st.sidebar.number_input("PTRATIO (Pupil-Teacher Ratio)", value=18.0)
b = st.sidebar.number_input("B (Proportion of Black Residents)", value=390.0)
lstat = st.sidebar.number_input("LSTAT (Percentage of Lower Status Population)", value=12.0)

# Predict button
if st.sidebar.button("Predict Price"):
    # Ensure correct data format
    input_data = np.array([[crim, zn, indus, rm, age, dis, tax, ptratio, b, lstat]], dtype=np.float64)

    # Check input shape
    if input_data.shape[1] != 10:
        st.error(f"Error: Model expects 10 features, but received {input_data.shape[1]}")
    else:
        # Make prediction
        prediction = model.predict(input_data)[0]

        # Display the result
        st.write(f"### Predicted House Price: ${prediction:.2f}")

# Footer
st.sidebar.markdown("---")


