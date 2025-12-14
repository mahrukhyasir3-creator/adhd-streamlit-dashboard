import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from PIL import Image

# ----------------------------------
# PAGE CONFIG
st.set_page_config(page_title="🧠 ADHD Smart Dashboard", layout="wide")
st.title("🧠 ADHD Smart Monitoring Dashboard")

# ----------------------------------
# LOAD DATA
df = pd.read_excel("ADHD_vs_Control_Sentiment_Dataset_500.xlsx")

# ----------------------------------
# TRAIN MODELS
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Text"])

group_model = LogisticRegression(max_iter=1000)
group_model.fit(X, df["Group"])

mood_model = LogisticRegression(max_iter=1000)
mood_model.fit(X, df["Mood"])

sentiment_model = LogisticRegression(max_iter=1000)
sentiment_model.fit(X, df["Sentiment"])

# ----------------------------------
# USER INPUT SECTION
st.subheader("📝 Input Options")

colA, colB = st.columns(2)

with colA:
    user_text = st.text_area("Enter text about feelings / behavior")
    user_keywords = st.text_input("Enter keywords (comma separated)")

with colB:
    uploaded_image = st.file_uploader("Upload image (optional)", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        img = Image.open(uploaded_image)
        st.image(img, caption="Uploaded Image", width=200)

# Decide final input
input_text = user_text if user_text.strip() != "" else user_keywords

# ----------------------------------
# ANALYZE BUTTON
if st.button("🔍 Analyze"):
    if input_text.strip() == "":
        st.warning("Please enter text or keywords")
    else:
        vec = vectorizer.transform([input_text])

        group = group_model.predict(vec)[0]
        mood = mood_model.predict(vec)[0]
        sentiment = sentiment_model.predict(vec)[0]

        # ----------------------------------
        # ADHD ALERT LOGIC
        alert = "Normal"
        if group == "ADHD" and mood in ["Angry", "Frustrated"]:
            alert = "⚠️ Hyperactivity Possible"
        elif group == "ADHD":
            alert = "⚠️ At Risk"

        # ----------------------------------
        # RESULTS
        st.subheader("📊 Analysis Result")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Patient Type", group)
        c2.metric("Mood", mood)
        c3.metric("Sentiment", sentiment)
        c4.metric("Alert Level", alert)

        # ----------------------------------
        # GUIDANCE & SUGGESTIONS
        st.subheader("💡 Proper Guidance")

        if mood == "Happy":
            st.success("Mood is stable. Maintain routine, sleep schedule, and balanced activities 🌟")

        elif mood == "Sad":
            st.info("Encourage breaks, light exercise, and emotional support 💙")

        elif mood == "Angry":
            st.warning("Recommend breathing exercises, quiet environment, and screen reduction 🔕")

        elif mood == "Frustrated":
            st.warning("Break tasks into small steps and use reminders 🧩")

        if alert == "⚠️ Hyperactivity Possible":
            st.error("""
            **ADHD Hyperactivity Alert**
            - Reduce stimulation  
            - Avoid multitasking  
            - Use grounding techniques  
            - Consider professional consultation
            """)

        elif alert == "⚠️ At Risk":
            st.info("""
            **ADHD Risk Detected**
            - Monitor behavior changes  
            - Maintain structured routine  
            - Use task timers  
            """)

        else:
            st.success("No immediate ADHD risk detected. Maintain healthy habits ✅")

# ----------------------------------
st.markdown("---")
st.caption("ADHD Smart Dashboard • Educational & Support Tool 💙")

