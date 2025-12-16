import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import calendar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from PIL import Image

# ================= PAGE CONFIG =================
st.set_page_config(page_title="🧠 ADHD Smart Support Dashboard", layout="wide")

# ================= BACKGROUND =================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1530023367847-a683933f4178");
    background-size: cover;
    background-position: center;
}

[data-testid="stVerticalBlock"] > div {
    background: rgba(255,255,255,0.93);
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    margin-bottom: 20px;
}

h1 {
    text-align: center;
    color: #4b4b9f;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 ADHD Smart Support Dashboard")

# ================= SIDEBAR =================
st.sidebar.title("🗓️ Daily Monitor")
now = datetime.now()
st.sidebar.markdown(f"**📅 Date:** {now.strftime('%d %B %Y')}")
st.sidebar.markdown(f"**⏰ Time:** {now.strftime('%H:%M')}")

st.sidebar.subheader("👨‍👩‍👧 Parent Reminder")
st.sidebar.info("""
• Observe child calmly  
• Avoid shouting  
• Give one task at a time  
• Appreciate small effort  
""")

# ================= LOAD DATA =================
df = pd.read_excel("ADHD_vs_Control_Sentiment_Dataset_500.xlsx")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Text"])

group_model = LogisticRegression(max_iter=1000)
group_model.fit(X, df["Group"])

mood_model = LogisticRegression(max_iter=1000)
mood_model.fit(X, df["Mood"])

sentiment_model = LogisticRegression(max_iter=1000)
sentiment_model.fit(X, df["Sentiment"])

# ================= INPUT =================
st.subheader("✍️ Enter Child / Person Behavior")

col1, col2 = st.columns(2)

with col1:
    user_text = st.text_area("Describe behavior (or leave empty)")
    keywords = st.text_input("OR enter keywords only")

with col2:
    img = st.file_uploader("Upload image (optional)", type=["jpg","png","jpeg"])
    if img:
        st.image(Image.open(img), width=200)

# ✅ KEYWORDS ONLY WORKING
input_text = user_text.strip() if user_text.strip() else keywords.strip()

# ================= LOG FILE =================
log_file = "behavior_log.csv"

def save_log(date, group, mood, severity):
    row = pd.DataFrame([[date, group, mood, severity]],
        columns=["Date","Group","Mood","Severity"])
    if os.path.exists(log_file):
        row.to_csv(log_file, mode="a", header=False, index=False)
    else:
        row.to_csv(log_file, index=False)

# ================= ANALYZE =================
if st.button("🔍 Analyze Behavior"):
    if input_text == "":
        st.warning("Please enter text or keywords")
    else:
        vec = vectorizer.transform([input_text])

        group = group_model.predict(vec)[0]
        mood = mood_model.predict(vec)[0]
        sentiment = sentiment_model.predict(vec)[0]

        # ---------------- SEVERITY LOGIC ----------------
        hyper_alert = False
        severity = "Low"

        if group == "ADHD" and mood in ["Angry", "Frustrated"]:
            severity = "High"
            hyper_alert = True
        elif group == "ADHD":
            severity = "Medium"

        save_log(now.strftime("%Y-%m-%d"), group, mood, severity)

        # ---------------- RESULT ----------------
        st.subheader("📊 Analysis Result")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Person Type", group)
        c2.metric("Mood", mood)
        c3.metric("Sentiment", sentiment)
        c4.metric("ADHD Severity", severity)

        # ================= GUIDANCE =================
        st.subheader("🧭 Guidance & Support")

        # ---------- NORMAL PERSON ----------
        if group == "Control":
            st.success("""
### ✅ Normal Behavior Detected
**Guidance:**
✔ Maintain routine  
✔ Encourage positive habits  
✔ Balanced screen time  
""")

        # ---------- ADHD PERSON ----------
        if group == "ADHD" and not hyper_alert:
            st.warning("""
### ⚠️ ADHD Detected (Moderate)
**What to do:**
✔ Give one task at a time  
✔ Use visual reminders  
✔ Break work into small steps  
✔ Keep routine consistent  
""")

        # ---------- HYPER ALERT ----------
        if hyper_alert:
            st.error("""
### 🚨 HYPERACTIVITY ALERT
**For Parents / Caregivers:**
1️⃣ Move child to calm place  
2️⃣ Speak softly, no shouting  
3️⃣ Deep breathing together  
4️⃣ Reduce noise & screen  
5️⃣ Observe for next 30 minutes  

⚠️ If repeated daily → consult specialist
""")

        # ================= CHILD EXERCISES =================
        st.subheader("🧩 Child-Friendly Exercises")

        st.markdown("""
**🫁 Calm Breathing**
- Breathe in nose (4 sec)  
- Breathe out mouth (6 sec)  
- Repeat 5 times  

**🎯 Focus Game**
- Ask child to color or draw for 5 minutes  
- No phone / TV  

**🚶 Movement**
- Slow walk  
- Stretch arms  
- Jumping jacks (10 times)  
""")

# ================= WEEKLY LOG =================
st.subheader("📅 Weekly Behavior Summary (Last Section)")

if os.path.exists(log_file):
    log_df = pd.read_csv(log_file)
    st.dataframe(log_df.tail(7))
else:
    st.info("No behavior history available yet.")
