import streamlit as st
import pickle
import time
import random
import pandas as pd

# PAGE SETTINGS
st.set_page_config(
    page_title="AI MBTI Personality Lab",
    page_icon="🧠",
    layout="wide"
)

# ---------- CSS STYLE ----------

st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#020617,#0f172a,#1e293b);
color:white;
}

/* TITLE */

.title{
font-size:52px;
font-weight:700;
text-align:center;
color:#38bdf8;
}

.subtitle{
text-align:center;
color:#cbd5f5;
margin-bottom:30px;
}

/* CARD */

.card{
background:#111827;
padding:20px;
border-radius:14px;
border:1px solid #334155;
}

/* RESULT */

.result{
background:linear-gradient(135deg,#3b82f6,#2563eb);
padding:25px;
border-radius:14px;
text-align:center;
font-size:28px;
font-weight:600;
}

/* BUTTON */

div.stButton > button{
background:linear-gradient(135deg,#38bdf8,#3b82f6);
color:white;
border:none;
font-weight:bold;
border-radius:10px;
}

/* FOOTER */

.footer{
text-align:center;
margin-top:40px;
color:#94a3b8;
}

</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------

st.markdown('<p class="title">🧠 AI MBTI Personality Lab</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Explore your personality with AI</p>', unsafe_allow_html=True)

# ---------- LOAD MODEL ----------

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR,"../Models/mbti_model.pkl"),"rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR,"../Models/vectorizer.pkl"),"rb"))

# ---------- MBTI DATA ----------

mbti_info = {
"INTJ":("Architect","Strategic planners and deep thinkers."),
"INTP":("Thinker","Curious analysts who love ideas."),
"ENTJ":("Commander","Confident natural leaders."),
"ENTP":("Debater","Innovative and energetic thinkers."),
"INFJ":("Advocate","Insightful idealists."),
"INFP":("Mediator","Creative and empathetic souls."),
"ENFJ":("Protagonist","Charismatic motivators."),
"ENFP":("Campaigner","Energetic and imaginative."),
"ISTJ":("Logistician","Responsible and organized."),
"ISFJ":("Defender","Supportive and loyal."),
"ESTJ":("Executive","Efficient leaders."),
"ESFJ":("Consul","Friendly caretakers."),
"ISTP":("Virtuoso","Hands-on experimenters."),
"ISFP":("Adventurer","Creative explorers."),
"ESTP":("Entrepreneur","Energetic action-takers."),
"ESFP":("Entertainer","Fun loving performers.")
}

# ---------- LAYOUT ----------

col1, col2 = st.columns([2,1])

# ---------- INPUT AREA ----------

with col1:

    st.markdown("### ✍ Describe Yourself")

    text = st.text_area(
        "Write about your hobbies, thoughts, behaviour or decisions",
        height=200
    )

    predict = st.button("🔍 Predict Personality")

# ---------- SIDEBAR MBTI CARDS ----------

with col2:

    st.markdown("### 🧩 Personality Explorer")

    grid = st.columns(2)

    for i,mbti in enumerate(mbti_info):

        with grid[i%2]:

            if st.button(mbti):

                name,desc = mbti_info[mbti]

                st.info(f"**{mbti} — {name}**\n\n{desc}")

# ---------- PREDICTION ----------

if predict:

    if text.strip()=="":
        st.warning("Please write something about yourself.")

    else:

        st.info("🧠 AI is analysing your personality...")

        # MINI GAME
        st.markdown("### 🎮 Quick Game")

        number = random.randint(1,5)
        guess = st.slider("Guess number (1-5)",1,5)

        if st.button("Check"):
            if guess==number:
                st.success("🎉 Correct guess!")
            else:
                st.error("Not correct 😅")

        # LOADING
        with st.spinner("Analyzing text..."):
            time.sleep(2)

        vector = vectorizer.transform([text])
        prediction = model.predict(vector)[0]
        confidence = model.predict_proba(vector).max()*100

        name,desc = mbti_info[prediction]

        # RESULT CARD
        st.markdown(f"""
        <div class="result">
        🎯 {prediction} — {name}
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(confidence))
        st.write(f"Confidence Score: **{confidence:.2f}%**")

        st.markdown(f"""
        <div class="card">
        <b>Description</b><br><br>
        {desc}
        </div>
        """, unsafe_allow_html=True)

        # ---------- TRAIT CHART ----------

        st.markdown("### 📊 Personality Trait Visualization")

        data = pd.DataFrame({
            "Trait":["Introversion","Intuition","Thinking","Judging"],
            "Score":[random.randint(40,100),
                     random.randint(40,100),
                     random.randint(40,100),
                     random.randint(40,100)]
        })

        st.bar_chart(data.set_index("Trait"))

# ---------- HELP DESK ----------

st.divider()

st.markdown("### 💬 Help Desk")

with st.expander("How does this AI work?"):
    st.write(
    "The system uses a machine learning model trained on MBTI datasets to estimate your personality from text patterns."
    )

with st.expander("What should I write?"):
    st.write(
    "Describe your behaviour, hobbies, how you make decisions, and how you interact with people."
    )

with st.expander("Is this accurate?"):
    st.write("It is an AI estimation and should be used for exploration.")

# ---------- FOOTER ----------

st.markdown('<p class="footer">AI Personality Lab • Built with Streamlit by KJ 🩷</p>', unsafe_allow_html=True)
