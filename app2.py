# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 22:04:56 2025

@author: Lokesh
"""

import streamlit as st
import pickle

# Title for the app
st.title('Regression Model For Survival Prediction')

# Load the trained model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Function to predict survival
def predict_survival(PassengerId, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked):
    # Use the loaded model to make a prediction
    prediction = loaded_model.predict([[PassengerId, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])
    return prediction[0]  # Assuming the model returns an array, extract the first element

# Main function for the Streamlit app
def main():
    # Input fields
    PassengerId = st.slider("Passenger ID", min_value=1, max_value=1000)
    Pclass = st.selectbox("Passenger Class", [1, 2, 3])
    Sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    Age = st.slider("Age", min_value=1, max_value=100)
    SibSp = st.slider("Number of Siblings/Spouses Aboard", min_value=0, max_value=10)
    Parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10)
    Fare = st.number_input("Fare", min_value=0.0, max_value=600.0)
    Embarked = st.select_slider("Embarked [S = 0, C = 1, Q = 2]", [0, 1, 2])

    # Predict button
    if st.button('Predict'):
        # Make prediction
        prediction = predict_survival(PassengerId, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)

        # Map prediction to a meaningful message
        if prediction == 1:
            result_message = "The person Survived"
        else:
            result_message = "The person did not Survive"

        # Display result
        st.success(f'The Prediction is: {result_message}')

# Run the app
if __name__ == '__main__':
    main()
