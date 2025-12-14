import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import speech_recognition as sr
from PIL import Image

# -----------------------------------
# PAGE CONFIG
st.set_page_config(page_title="🧠 ADHD Smart Dashboard", layout="wide")
st.title("🧠 ADHD Smart Monitoring Dashboard Pro")

# -----------------------------------
# LOAD DATA
df = pd.read_excel("ADHD_vs_Control_Sentiment_Dataset_500.xlsx")

# -----------------------------------
# TRAIN MODELS
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Text"])

group_model = LogisticRegression(max_iter=1000)
group_model.fit(X, df["Group"])

mood_model = LogisticRegression(max_iter=1000)
mood_model.fit(X, df["Mood"])

sentiment_model = LogisticRegression(max_iter=1000)
sentiment_model.fit(X, df["Sentiment"])

# -----------------------------------
# INPUT SECTION
st.subheader("🧾 Input Options")

tab1, tab2, tab3 = st.tabs(["✍️ Text / Keywords", "🎤 Voice Input", "🖼️ Image Upload"])

input_text = ""

# -------- TEXT / KEYWORDS
with tab1:
    user_text = st.text_area("Enter your thoughts or behaviour:")
    user_keywords = st.text_input("Or enter keywords (comma separated)")
    if user_text.strip() != "":
        input_text = user_text
    elif user_keywords.strip() != "":
        input_text = user_keywords

# -------- VOICE INPUT
with tab2:
    st.info("Click button and speak clearly (English)")
    if st.button("🎙️ Record Voice"):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Listening...")
            audio = r.listen(source)
        try:
            voice_text = r.recognize_google(audio)
            st.success("Voice Converted to Text:")
            st.write(voice_text)
            input_text = voice_text
        except:
            st.error("Could not recognize voice")

# -------- IMAGE INPUT
with tab3:
    image = st.file_uploader("Upload face/behaviour image (optional)", type=["jpg", "png", "jpeg"])
    if image:
        img = Image.open(image)
        st.image(img, caption="Uploaded Image", width=250)
        st.info("Image added as behavioural context (non-clinical support)")

# -----------------------------------
# ANALYSIS
if st.button("🔍 Analyze"):
    if input_text.strip() == "":
        st.warning("Please provide text, keywords or voice input")
    else:
        vec = vectorizer.transform([input_text])

        group = group_model.predict(vec)[0]
        mood = mood_model.predict(vec)[0]
        sentiment = sentiment_model.predict(vec)[0]

        # -----------------------------------
        # RESULTS
        st.subheader("📊 Analysis Result")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🧠 Condition", group)
        c2.metric("🙂 Mood", mood)
        c3.metric("💭 Sentiment", sentiment)

        # ADHD ALERT LOGIC
        adhd_alert = "LOW"
        if group == "ADHD" and mood in ["Angry", "Frustrated"]:
            adhd_alert = "⚠️ HIGH (Possible Hyper Episode Risk)"

        c4.metric("🚨 ADHD Alert", adhd_alert)

        # -----------------------------------
        # DETAILED INTERPRETATION
        st.subheader("🧩 Interpretation")

        if group == "ADHD":
            st.warning("User shows ADHD-related behavioural patterns")
        else:
            st.success("User currently shows normal behavioural patterns")

        if adhd_alert.startswith("⚠️"):
            st.error("""
            ⚠️ **Warning:**  
            User is currently calm but may become hyperactive later.
            Monitoring and intervention recommended.
            """)

        # -----------------------------------
        # SMART GUIDED SUGGESTIONS
        st.subheader("💡 Personalized Guidance")

        if mood == "Happy":
            st.success("""
            ✅ **You are doing well**
            - Maintain routine
            - Avoid overstimulation
            - Keep sleep schedule fixed
            """)

        elif mood == "Sad":
            st.info("""
            💙 **Low mood detected**
            - Take 5–10 min breaks
            - Light physical activity
            - Talk to a trusted person
            """)

        elif mood == "Angry":
            st.warning("""
            🔕 **Anger detected**
            - Deep breathing (4-7-8)
            - Reduce screen time
            - Quiet environment recommended
            """)

        elif mood == "Frustrated":
            st.warning("""
            🧩 **Frustration detected**
            - Break task into small steps
            - Use timer (Pomodoro)
            - Remove distractions
            """)

        if group == "ADHD":
            st.markdown("""
            ### 🧠 ADHD-Specific Guidance
            - Use reminders & alarms ⏰  
            - Avoid multitasking  
            - Fixed daily routine  
            - Short focused work sessions  
            - Mindfulness / breathing exercises  
            """)

# -----------------------------------
st.markdown("---")
st.caption("🧠 ADHD Smart Monitoring Dashboard | Academic & Assistive Use")

