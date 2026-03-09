import streamlit as st
import pickle

# Page configuration
st.set_page_config(
    page_title="MBTI Personality Predictor",
    page_icon="🧠",
    layout="centered"
)

# Title
st.title("🧠 MBTI Personality Predictor")
st.write("This AI model predicts your MBTI personality type based on your text.")

# Sidebar
st.sidebar.title("About Project")
st.sidebar.write(
"""
This app predicts **MBTI personality types** using Machine Learning.

Model used:
- TF-IDF Vectorizer
- Machine Learning Classifier

Enter a paragraph about yourself and see your predicted personality type.
"""
)

# Load model and vectorizer
model = pickle.load(open("models/mbti_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# Input text
st.subheader("Enter your text")
user_input = st.text_area(
    "Write something about your thoughts, behaviour, or opinions:",
    height=200
)

# Predict button
if st.button("Predict Personality"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text first.")
    else:
        with st.spinner("Analyzing your personality..."):

            # Transform input
            input_vector = vectorizer.transform([user_input])

            # Predict
            prediction = model.predict(input_vector)[0]

        # Display result
        st.success(f"🎯 Predicted Personality Type: **{prediction}**")

        st.info(
        """
        MBTI Types include:
        INTJ, INTP, ENTJ, ENTP,
        INFJ, INFP, ENFJ, ENFP,
        ISTJ, ISFJ, ESTJ, ESFJ,
        ISTP, ISFP, ESTP, ESFP
        """
        )

# Footer
st.markdown("---")
st.write("Built with ❤️ using Streamlit")
st.markdown("---")
st.markdown("👩‍💻 **Built by J1507Dharshini K** | Data Science Project")
