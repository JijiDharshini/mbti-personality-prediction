import streamlit as st
import pickle
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))

from src.preprocess.preprocess import clean_text

model = pickle.load(open("Models/mbti_model.pkl", "rb"))
vectorizer = pickle.load(open("Models/vectorizer.pkl", "rb"))

st.title("Personality Prediction from Text")

user_input = st.text_area("Enter your text")

if st.button("Predict Personality"):

    cleaned = clean_text(user_input)

    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized)

    st.success(f"Predicted Personality Type: {prediction[0]}")
    
