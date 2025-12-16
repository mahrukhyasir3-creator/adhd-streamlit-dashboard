import streamlit as st
import pandas as pd
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

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
    background: rgba(255,255,255,0.94);
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    margin-bottom: 20px;
}

h1 { text-align:center; color:#4b4b9f; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 ADHD Smart Support Dashboard")
st.caption("Mood-aware guidance for ADHD and non-ADHD users")

# ================= SIDEBAR (DATE & TIME) =================
st.sidebar.title("🗓️ Daily Monitor")

now = datetime.now()
st.sidebar.markdown(f"**📅 Date:** {now.strftime('%A, %d %B %Y')}")
st.sidebar.markdown(f"**⏰ Time:** {now.strftime('%H:%M:%S')}")

st.sidebar.subheader("🔔 Daily Self-Care Reminder")
st.sidebar.info("""
• One task at a time  
• Take short breaks  
• Avoid overload  
• Maintain sleep routine  
""")

# ================= LOAD DATA =================
df = pd.read_excel("ADHD_vs_Control_Sentiment_Dataset_500.xlsx")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Text"])

group_model = LogisticRegression(max_iter=1000).fit(X, df["Group"])
mood_model = LogisticRegression(max_iter=1000).fit(X, df["Mood"])
sentiment_model = LogisticRegression(max_iter=1000).fit(X, df["Sentiment"])

# ================= INPUT =================
st.subheader("✍️ Enter Feelings / Behavior")

user_text = st.text_area("Write behavior or feelings")
keywords = st.text_input("OR enter keywords only")

# Keywords-only support
input_text = user_text.strip() if user_text.strip() else keywords.strip()

# ================= ANALYZE =================
if st.button("🔍 Analyze"):
    if input_text == "":
        st.warning("Please enter text or keywords")
    else:
        vec = vectorizer.transform([input_text])

        group = group_model.predict(vec)[0]
        mood = mood_model.predict(vec)[0]
        sentiment = sentiment_model.predict(vec)[0]

        # ADHD severity
        severity = "Low"
        hyper = False
        if group == "ADHD" and mood in ["Angry", "Frustrated"]:
            severity = "High"
            hyper = True
        elif group == "ADHD":
            severity = "Medium"

        # ================= RESULT =================
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("User Type", group)
        c2.metric("Mood", mood)
        c3.metric("Sentiment", sentiment)
        c4.metric("Severity", severity)

        st.subheader("🧭 Guidance")

        # ============ CONTROL PERSON ============
        if group == "Control":
            st.success(f"""
### ✅ Normal Pattern Detected ({mood})

**General Guidance:**
✔ Maintain routine  
✔ Balanced workload  
✔ Adequate sleep  
✔ Healthy social interaction  

(No exercises required)
""")

        # ============ ADHD PERSON ============
        if group == "ADHD":

            # ---- HAPPY ----
            if mood == "Happy":
                st.success("""
### 😊 ADHD + Happy Mood
You are doing well.

✔ Maintain routine  
✔ Continue positive habits  
✔ No exercise required right now  
""")

            # ---- SAD ----
            elif mood == "Sad":
                st.warning("""
### 😔 ADHD + Sad Mood

**Guidance:**
✔ Emotional support  
✔ Reduce workload  
✔ Stay connected  

**Exercises:**
🫁 Slow breathing (4 sec in, 6 sec out × 5)  
🚶 10-minute light walk  
🧠 Write feelings on paper  
""")

            # ---- ANGRY ----
            elif mood == "Angry":
                st.error("""
### 😠 ADHD + Angry Mood

**Guidance:**
✔ Pause current task  
✔ Calm environment  
✔ Reduce noise & screen  

**Relaxation Exercises:**
🫁 Deep breathing (5 rounds)  
✋ Muscle relaxation  
🚶 Walk before reacting  
""")

            # ---- FRUSTRATED / HYPER ----
            elif mood == "Frustrated":
                st.error("""
### ⚡ ADHD + Frustrated / Hyper Mood
🚨 **Hyperactivity Risk Detected**

**Immediate Steps:**
✔ Stop multitasking  
✔ Quiet space  
✔ One instruction at a time  

**Strong Calming Exercises:**
🫁 Box breathing (4-4-4-4)  
🧠 Grounding: name 5 things you see  
🚶 Slow body movement  

⚠️ Repeated pattern → professional help advised
""")

# ================= END =================
