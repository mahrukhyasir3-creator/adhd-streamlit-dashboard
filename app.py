import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ================= PAGE CONFIG =================
st.set_page_config(page_title="🧠 ADHD Smart Support Dashboard", layout="wide")

# ================= UI THEME =================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1530023367847-a683933f4178");
    background-size: cover;
    background-position: center;
}

[data-testid="stVerticalBlock"] > div {
    background: rgba(255,255,255,0.96);
    padding: 24px;
    border-radius: 18px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    margin-bottom: 22px;
}

h1 { text-align:center; color:#4b4b9f; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 ADHD Smart Support Dashboard")
st.caption("Mood-based guidance with detailed exercise summaries")

# ================= SIDEBAR =================
st.sidebar.title("🗓️ Daily Monitor")
now = datetime.now()
st.sidebar.markdown(f"**📅 Date:** {now.strftime('%A, %d %B %Y')}")
st.sidebar.markdown(f"**⏰ Time:** {now.strftime('%H:%M:%S')}")

st.sidebar.subheader("🔔 Daily Reminder")
st.sidebar.info("""
• One task at a time  
• Short breaks  
• Avoid overload  
• Proper sleep  
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

input_text = user_text.strip() if user_text.strip() else keywords.strip()

# ================= ANALYZE BUTTON =================
analyze_clicked = st.button("🔍 Analyze")

# ================= ANALYSIS =================
if analyze_clicked:

    if input_text == "":
        st.warning("Please enter text or keywords to analyze.")

    elif len(input_text.split()) < 3:
        st.info("Please enter at least 3 words for better analysis.")

    else:
        vec = vectorizer.transform([input_text])

        group = group_model.predict(vec)[0]
        mood = mood_model.predict(vec)[0]
        sentiment = sentiment_model.predict(vec)[0]

        severity = "Low"
        if group == "ADHD" and mood in ["Angry", "Frustrated"]:
            severity = "High"
        elif group == "ADHD":
            severity = "Medium"

        # ================= RESULT =================
        st.subheader("📊 Analysis Result")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("User Type", group)
        c2.metric("Mood", mood)
        c3.metric("Sentiment", sentiment)
        c4.metric("Severity", severity)

        st.subheader("🧭 Guidance & Exercise Summary")

        # ================= CONTROL =================
        if group == "Control":
            st.success(f"""
### ✅ Normal Pattern Detected ({mood})

Your behavior and emotional state fall within a normal range.

✔ Maintain routine  
✔ Balance work and rest  
✔ Continue healthy habits  

(No exercises required)
""")

        # ================= ADHD =================
        if group == "ADHD":

            if mood == "Happy":
                st.success("""
### 😊 ADHD + Happy Mood

You are emotionally stable.

✔ Maintain structure  
✔ Follow sleep routine  
✔ Keep positive habits  

(No exercises needed)
""")

            elif mood == "Sad":
                st.warning("""
### 😔 ADHD + Sad Mood

**Exercises:**
🫁 Slow breathing (4 in / 6 out × 5)  
🚶 10-minute light walk  
🧠 Write thoughts on paper  
""")

            elif mood == "Angry":
                st.error("""
### 😠 ADHD + Angry Mood

**Exercises:**
✋ Muscle relaxation  
🫁 Deep breathing  
🚶 Walk away from trigger  
""")

            elif mood == "Frustrated":
                st.error("""
### ⚡ ADHD + Frustrated / Hyper Mood

**Exercises:**
🧠 Grounding 5-4-3-2-1  
🫁 Box breathing  
🚶 Slow controlled movement  

⚠️ Repeated pattern → professional help advised
""")

