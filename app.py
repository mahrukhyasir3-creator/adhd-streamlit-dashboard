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
st.caption("General-purpose behavior monitoring (child & adult)")

# ================= SIDEBAR =================
st.sidebar.title("🗓️ Daily Monitor")
now = datetime.now()
st.sidebar.markdown(f"**📅 Date:** {now.strftime('%d %B %Y')}")
st.sidebar.markdown(f"**⏰ Time:** {now.strftime('%H:%M')}")

st.sidebar.subheader("🔔 Daily Self-Care Reminder")
st.sidebar.info("""
• One task at a time  
• Short breaks  
• Limit screen overload  
• Regular sleep routine  
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
st.subheader("✍️ Enter Behavior / Feelings")

col1, col2 = st.columns(2)

with col1:
    user_text = st.text_area("Describe behavior or feelings")
    keywords = st.text_input("OR enter keywords only")

with col2:
    img = st.file_uploader("Upload image (optional)", type=["jpg","png","jpeg"])
    if img:
        st.image(Image.open(img), width=200)

# Keywords-only support
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

        # -------- ADHD SEVERITY LOGIC --------
        hyper_alert = False
        severity = "Low"

        if group == "ADHD" and mood in ["Angry", "Frustrated"]:
            severity = "High"
            hyper_alert = True
        elif group == "ADHD":
            severity = "Medium"

        save_log(now.strftime("%Y-%m-%d"), group, mood, severity)

        # -------- RESULTS --------
        st.subheader("📊 Analysis Result")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("User Type", group)
        c2.metric("Mood", mood)
        c3.metric("Sentiment", sentiment)
        c4.metric("ADHD Severity", severity)

        # ================= GUIDANCE =================
        st.subheader("🧭 Guidance")

        # ✅ CONTROL PERSON → NO EXERCISES
        if group == "Control":
            st.success("""
### ✅ Typical Behavior Pattern
No ADHD-related intervention required.

**General Advice:**
✔ Maintain healthy routine  
✔ Balanced workload  
✔ Regular breaks and sleep  
""")

        # ✅ ADHD (MODERATE)
        if group == "ADHD" and not hyper_alert:
            st.warning("""
### ⚠️ ADHD Indicators Detected (Moderate)

**Recommended Actions:**
✔ Use reminders or planners  
✔ Break tasks into small steps  
✔ Reduce distractions  
✔ Maintain consistent routine  
""")

        # ✅ ADHD (HIGH / HYPER ALERT)
        if hyper_alert:
            st.error("""
### 🚨 Hyperactivity Risk Alert

**Immediate Steps:**
1️⃣ Move to a calm environment  
2️⃣ Practice slow breathing  
3️⃣ Avoid multitasking  
4️⃣ Reduce noise & screen exposure  

⚠️ If this pattern repeats frequently, consult a professional.
""")

        # ================= EXERCISES (ONLY FOR ADHD) =================
        if group == "ADHD":
            st.subheader("🧩 Recommended Exercises")

            st.markdown("""
**🫁 Breathing Exercise**
- Inhale slowly for 4 seconds  
- Hold for 2 seconds  
- Exhale for 6 seconds  
- Repeat 5 times  

**🧠 Focus Practice**
- Choose one simple task  
- Set timer for 10 minutes  
- No phone, no interruptions  

**🚶 Physical Movement**
- 10–15 minute walk  
- Gentle stretching  
- Light mobility exercises  
""")

# ================= WEEKLY LOG =================
st.subheader("📅 Recent Behavior Summary")

if os.path.exists(log_file):
    log_df = pd.read_csv(log_file)
    st.dataframe(log_df.tail(7))
else:
    st.info("No behavior history available yet.")

