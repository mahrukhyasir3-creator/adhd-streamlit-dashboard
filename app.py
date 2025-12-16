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

# ================= SIDEBAR (DATE & TIME) =================
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
        if group == "ADHD" and mood in ["Angry", "Frustrated"]:
            severity = "High"
        elif group == "ADHD":
            severity = "Medium"

        # ================= RESULT =================
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

**Summary:**  
Your behavior and emotional state fall within a normal range. There are no signs of ADHD-related difficulty at this time.

**Why no exercises?**  
Exercises and interventions are only needed when attention or emotional regulation is impaired.

**What to do instead:**  
✔ Maintain your routine  
✔ Balance work and rest  
✔ Continue healthy habits  
""")

        # ================= ADHD =================
        if group == "ADHD":

            # ---------- HAPPY ----------
            if mood == "Happy":
                st.success("""
### 😊 ADHD + Happy Mood

**Summary:**  
You are currently stable and emotionally balanced. Attention and mood regulation appear healthy.

**Why no exercises now?**  
When mood is positive, unnecessary exercises may interrupt natural focus.

**What to do:**  
✔ Maintain structure  
✔ Follow sleep routine  
✔ Keep positive habits consistent  
""")

            # ---------- SAD ----------
            elif mood == "Sad":
                st.warning("""
### 😔 ADHD + Sad Mood

**Summary:**  
Low mood combined with ADHD often reduces motivation and concentration.

**Goal of exercises:**  
✔ Gently lift mood  
✔ Improve emotional regulation  
✔ Restore focus without pressure  

**Exercises & How to Do Them:**

🫁 **Slow Breathing**  
• Inhale 4 seconds → Exhale 6 seconds  
• Repeat 5 times  
• Helps calm the nervous system  

🚶 **Light Walk**  
• Walk slowly for 10 minutes  
• No phone, focus on steps  
• Improves blood flow & mood  

🧠 **Thought Release**  
• Write feelings on paper  
• Do not judge  
• Tear paper after writing  
""")

            # ---------- ANGRY ----------
            elif mood == "Angry":
                st.error("""
### 😠 ADHD + Angry Mood

**Summary:**  
Anger in ADHD often comes from overstimulation and emotional overload.

**Goal of exercises:**  
✔ Reduce emotional intensity  
✔ Prevent impulsive reactions  
✔ Calm the body first  

**Exercises & How to Do Them:**

✋ **Muscle Relaxation**  
• Tighten fists for 5 sec  
• Release slowly  
• Repeat 5 times  

🫁 **Deep Breathing**  
• Inhale 5 sec → Hold 2 sec → Exhale 7 sec  
• Repeat 5 rounds  

🚶 **Cool-Down Walk**  
• Walk away from trigger  
• 5–10 minutes  
• No talking until calm  
""")

            # ---------- FRUSTRATED / HYPER ----------
            elif mood == "Frustrated":
                st.error("""
### ⚡ ADHD + Frustrated / Hyper Mood

**Summary:**  
This indicates high mental overload and risk of hyperactivity.

**Goal of exercises:**  
✔ Ground attention  
✔ Reduce sensory overload  
✔ Prevent escalation  

**Exercises & How to Do Them:**

🧠 **Grounding (5–4–3–2–1)**  
• Name 5 things you see  
• 4 things you touch  
• 3 things you hear  
• 2 things you smell  
• 1 thing you taste  

🫁 **Box Breathing**  
• Inhale 4 sec → Hold 4 sec  
• Exhale 4 sec → Hold 4 sec  
• Repeat 5 cycles  

🚶 **Controlled Movement**  
• Slow stretching  
• No running  
• No screens  

⚠️ **Important:**  
If this pattern repeats daily, professional guidance is recommended.
""")

# ================= END =================
