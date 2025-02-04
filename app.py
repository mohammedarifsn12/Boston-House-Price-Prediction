import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('regression_model.pkl', 'rb'))

# Define the Streamlit app
st.title("Boston House Price Prediction")

# Sidebar for input fields
st.sidebar.header("Enter House Features")

crim = st.sidebar.number_input("CRIM (Per Capita Crime Rate)", value=0.1)
zn = st.sidebar.number_input("ZN (Proportion of Residential Land)", value=0.0)
indus = st.sidebar.number_input("INDUS (Proportion of Non-Retail Business Acres)", value=0.0)
chas = st.sidebar.selectbox("CHAS (Charles River Dummy Variable: 0 or 1)", [0, 1])
nox = st.sidebar.number_input("NOX (Nitrogen Oxides Concentration)", value=0.5)
rm = st.sidebar.number_input("RM (Average Number of Rooms)", value=6.0)
age = st.sidebar.number_input("AGE (Proportion of Owner-Occupied Units Built Before 1940)", value=50.0)
dis = st.sidebar.number_input("DIS (Weighted Distance to Employment Centers)", value=4.0)
rad = st.sidebar.number_input("RAD (Index of Accessibility to Highways)", value=1)
tax = st.sidebar.number_input("TAX (Property Tax Rate per $10,000)", value=300)
ptratio = st.sidebar.number_input("PTRATIO (Pupil-Teacher Ratio)", value=18.0)
b = st.sidebar.number_input("B (Proportion of Black Residents)", value=390.0)
lstat = st.sidebar.number_input("LSTAT (Percentage of Lower Status Population)", value=12.0)

# Predict button
if st.sidebar.button("Predict Price"):
    # Ensure correct input format
    input_data = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]], dtype=np.float64)

    # Check input shape
    if input_data.shape[1] != 13:
        st.error(f"Error: Model expects 13 features, but received {input_data.shape[1]}")
    else:
        # Make prediction
        prediction = model.predict(input_data)[0]

        # Display the result
        st.write(f"### Predicted House Price: ${prediction:.2f}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Developed with ❤️ using Streamlit")


