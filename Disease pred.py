# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 20:42:55 2024

@author : AS1209
"""

import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

import pickle

# Load models from the updated path
diabetes_model = pickle.load(open('/Users/anweshaswarup/Downloads/Multiple Disease Prediction Systems/Saved Models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('/Users/anweshaswarup/Downloads/Multiple Disease Prediction Systems/Saved Models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('/Users/anweshaswarup/Downloads/Multiple Disease Prediction Systems/Saved Models/parkinsons_model.sav', 'rb'))
heera_model = pickle.load(open('/Users/anweshaswarup/Downloads/Multiple Disease Prediction Systems/Saved Models/heera_model.sav', 'rb'))
Loan_model = pickle.load(open('/Users/anweshaswarup/Downloads/Multiple Disease Prediction Systems/Saved Models/Loan_model.sav', 'rb'))


def predict_loan_status(data):
    prediction = Loan_model.predict(data)
    return 'Approved' if prediction[0] == 1 else 'Not Approved'


# Sidebar for Navigation
with st.sidebar:
    selected = option_menu(' Health Guard Pro',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            "Parkinson's Disease",
                            
                            ],
                         
                           icons=['capsule', 'heart-pulse', 'virus'],
                           default_index=0)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction')
    col1, col2, col3 = st.columns(3)
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')
    
    diab_diagnosis = ''
    diab_prediction = None  # Initialize diab_prediction variable

    # Creating a button for Prediction
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

        if diab_prediction is not None and diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        elif diab_prediction is not None and diab_prediction[0] == 0:
            diab_diagnosis = 'The person is not diabetic'
        else:
            diab_diagnosis = 'Error in prediction'
    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction')
    col1, col2, col3 = st.columns(3)
    age = st.text_input('Age')
    sex = st.text_input('Sex')
    cp = st.text_input('Chest Pain types')
    trestbps = st.text_input('Resting Blood Pressure')
    chol = st.text_input('Serum Cholestoral in mg/dl')
    fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    restecg = st.text_input('Resting Electrocardiographic results')
    thalach = st.text_input('Maximum Heart Rate achieved')
    exang = st.text_input('Exercise Induced Angina')
    oldpeak = st.text_input('ST depression induced by exercise')
    slope = st.text_input('Slope of the peak exercise ST segment')
    ca = st.text_input('Major vessels colored by flourosopy')
    thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    heart_diagnosis = ''

    # Creating a button for Prediction
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'
    st.success(heart_diagnosis)

# Parkinson's Prediction Page
if selected == "Parkinson's Disease":
    st.title("Parkinson's Disease")
    col1, col2, col3, col4, col5 = st.columns(5)
    fo = st.text_input('MDVP:Fo(Hz)')
    fhi = st.text_input('MDVP:Fhi(Hz)')
    flo = st.text_input('MDVP:Flo(Hz)')
    Jitter_percent = st.text_input('MDVP:Jitter(%)')
    Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
    RAP = st.text_input('MDVP:RAP')
    PPQ = st.text_input('MDVP:PPQ')
    DDP = st.text_input('Jitter:DDP')
    Shimmer = st.text_input('MDVP:Shimmer')
    Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
    APQ3 = st.text_input('Shimmer:APQ3')
    APQ5 = st.text_input('Shimmer:APQ5')
    APQ = st.text_input('MDVP:APQ')
    DDA = st.text_input('Shimmer:DDA')
    NHR = st.text_input('NHR')
    HNR = st.text_input('HNR')
    RPDE = st.text_input('RPDE')
    DFA = st.text_input('DFA')
    spread1 = st.text_input('spread1')
    spread2 = st.text_input('spread2')
    D2 = st.text_input('D2')
    PPE = st.text_input('PPE')

    parkinsons_diagnosis = ''

    # Creating a button for Prediction
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]])

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"
    st.success(parkinsons_diagnosis)
    
# Diamond Price Prediction Page
if selected == 'Diamond Price Prediction':
    st.title('Diamond Price Prediction')
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11 = st.columns(11)
    
    # Gather input features from the user
    carat = st.text_input('Carat')
    cut = st.selectbox('Cut', ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
    color = st.selectbox('Color', ['D', 'E', 'F', 'G', 'H', 'I', 'J'])
    clarity = st.selectbox('Clarity', ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
    depth = st.text_input('Depth')
    table = st.text_input('Table')
    x = st.text_input('Length (x)')
    y = st.text_input('Width (y)')
    z = st.text_input('Height (z)')
    
    diamond_diagnosis = ''

    # Creating a button for Prediction
    if st.button('Predict Diamond Price'):
        # Preprocess the input data and make predictions
        predicted_price = predict_diamond_price(carat, cut, color, clarity, depth, table, x, y, z)
        diamond_diagnosis = f'Predicted Diamond Price: {predicted_price[0]}'

    st.success(diamond_diagnosis)
    
# Loan Status Prediction Page
# Loan Status Prediction Page
if selected == 'Loan Status Prediction':
    st.title('Loan Status Prediction')
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11 = st.columns(11)
    
    # Gather input features from the user
    gender = st.selectbox('Gender', ['Male', 'Female'])
    married = st.selectbox('Marital Status', ['Yes', 'No'])
    dependents = st.slider('Number of Dependents', 0, 3, 1)
    education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
    self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
    applicant_income = st.slider('Applicant Income', 150, 10000, 5000)
    coapplicant_income = st.slider('Coapplicant Income', 0, 10000, 2000)
    loan_amount = st.slider('Loan Amount', 10, 500, 150)
    loan_amount_term = st.slider('Loan Amount Term (months)', 12, 360, 360)
    credit_history = st.selectbox('Credit History', [0, 1])
    property_area = st.selectbox('Property Area', ['Urban', 'Rural', 'Semiurban'])
    
    # Preprocess user input
    gender = 1 if gender == 'Male' else 0
    married = 1 if married == 'Yes' else 0
    education = 1 if education == 'Graduate' else 0
    self_employed = 1 if self_employed == 'Yes' else 0
    property_area_mapping = {'Urban': 0, 'Rural': 1, 'Semiurban': 2}
    property_area = property_area_mapping[property_area]
    
    loan_diagnosis = ''

    # Creating a button for Prediction
    if st.button('Predict Loan Status'):
        # Prepare the input data as a NumPy array
        input_data = np.array([[gender, married, dependents, education, self_employed, applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history, property_area]])
        # Make predictions using the loaded Loan Status Prediction model
        loan_status = predict_loan_status(input_data)
        loan_diagnosis = f'Loan Status: {loan_status}'

    st.success(loan_diagnosis)
