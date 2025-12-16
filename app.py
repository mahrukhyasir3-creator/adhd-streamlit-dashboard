import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ================= PAGE CONFIG =================
st.set_page_config(page_title="🧠 ADHD Smart Support Dashboard", layout="wide")

# ================= BEAUTIFUL UI THEME =================
st.markdown("""
<style>

/* Background */
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1530023367847-a683933f4178");
    background-size: cover;
    background-position: center;
}

/* Main card */
[data-testid="stVerticalBlock"] > div {
    background: rgba(255,255,255,0.95);
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    margin-bottom: 22px;
}

/* Title */
h1 {
    text-align: center;
    color: #4b4b9f;
}

/* Result cards */
.result-box {
    padding: 18px;
    border-radius: 14px;
    text-align: center;
    font-weight: bold;
}

/* Color themes */
.green { background: #e8f8f5; color: #117864; }
.yellow { background: #fef9e7; color: #9a7d0a; }
.red { background: #fdecea; color: #922b21; }
.blue { background: #ebf5fb; color: #154360; }
.purple { background: #f4ecf7; color: #512e5f; }

/* Buttons */
.stButton > button {
    background-color: #6a5acd;
    color: white;
    border-radius: 10px;
    font-size: 16px;
    padding: 8px 20px;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 ADHD Smart Support Dashboard")
st.caption("Mood-based guidance & exercises (clean and calm design)")

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

# ================= ANALYZE =================
if st.button("🔍 Analyze"):
    if input_text == "":
        st.warning("Please enter text or keywords")
    else:
        vec = vectorizer.transform([input_text])
        group = group_model.predict(vec)[0]
        mood = mood_model.predict(vec)[0]
        sentiment = sentiment_model.predict(vec)[0]

        severity = "Low"
        color = "green"

        if group == "ADHD" and mood in ["Angry", "Frustrated"]:
            severity = "High"
            color = "red"
        elif group == "ADHD":
            severity = "Medium"
            color = "yellow"

        # ================= RESULT =================
        st.subheader("📊 Analysis Result")

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='result-box blue'>User Type<br>{group}</div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='result-box purple'>Mood<br>{mood}</div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='result-box blue'>Sentiment<br>{sentiment}</div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='result-box {color}'>Severity<br>{severity}</div>", unsafe_allow_html=True)

        st.subheader("🧭 Guidance")

        # ========== CONTROL ==========
        if group == "Control":
            st.markdown("""
<div class="result-box green">
<b>Normal Pattern</b><br>
Maintain routine, balanced work, and healthy rest.<br>
(No exercises required)
</div>
""", unsafe_allow_html=True)

        # ========== ADHD ==========
        if group == "ADHD":

            if mood == "Happy":
                st.markdown("""
<div class="result-box green">
<b>ADHD + Happy</b><br>
You are stable. Continue positive habits.<br>
(No exercises needed)
</div>
""", unsafe_allow_html=True)

            elif mood == "Sad":
                st.markdown("""
<div class="result-box yellow">
<b>ADHD + Sad</b><br>
• Slow breathing (4 in / 6 out ×5)<br>
• 10-minute light walk<br>
• Write thoughts on paper
</div>
""", unsafe_allow_html=True)

            elif mood == "Angry":
                st.markdown("""
<div class="result-box red">
<b>ADHD + Angry</b><br>
• Deep breathing (5 rounds)<br>
• Muscle relaxation<br>
• Walk away from trigger
</div>
""", unsafe_allow_html=True)

            elif mood == "Frustrated":
                st.markdown("""
<div class="result-box red">
<b>ADHD + Hyper / Frustrated</b><br>
• Grounding 5-4-3-2-1<br>
• Box breathing (4-4-4-4)<br>
• Quiet environment
</div>
""", unsafe_allow_html=True)

# ================= END =================
