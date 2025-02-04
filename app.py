import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('regression_model.pkl', 'rb'))

# Set page title
st.title("🏡 Boston House Price Prediction")

# Create input fields in the center using columns
st.markdown("### Enter House Features Below")

# Layout for better centering
col1, col2 = st.columns(2)

with col1:
    crim = st.number_input("CRIM (Per Capita Crime Rate)", value=0.1)
    zn = st.number_input("ZN (Proportion of Residential Land)", value=0.0)
    indus = st.number_input("INDUS (Proportion of Non-Retail Business Acres)", value=0.0)
    chas = st.selectbox("CHAS (Charles River Dummy Variable: 0 or 1)", [0, 1])
    nox = st.number_input("NOX (Nitrogen Oxides Concentration)", value=0.5)
    rm = st.number_input("RM (Average Number of Rooms)", value=6.0)
    age = st.number_input("AGE (Proportion of Owner-Occupied Units Built Before 1940)", value=50.0)

with col2:
    dis = st.number_input("DIS (Weighted Distance to Employment Centers)", value=4.0)
    rad = st.number_input("RAD (Index of Accessibility to Highways)", value=1)
    tax = st.number_input("TAX (Property Tax Rate per $10,000)", value=300)
    ptratio = st.number_input("PTRATIO (Pupil-Teacher Ratio)", value=18.0)
    b = st.number_input("B (Proportion of Black Residents)", value=390.0)
    lstat = st.number_input("LSTAT (Percentage of Lower Status Population)", value=12.0)

# Predict button
if st.button("Predict Price"):
    # Ensure correct input format
    input_data = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]], dtype=np.float64)

    # Check input shape
    if input_data.shape[1] != 13:
        st.error(f"Error: Model expects 13 features, but received {input_data.shape[1]}")
    else:
        # Make prediction
        prediction = model.predict(input_data)[0]

        # Display the result in the center
        st.markdown(f"## 🏠 Predicted House Price: **${prediction:.2f}**")

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit")


