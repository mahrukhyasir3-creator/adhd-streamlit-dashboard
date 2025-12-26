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
st.caption("Real-time mood-based guidance & exercises")

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

# ================= LOAD DATA (TRAINING) =================
df = pd.read_excel("ADHD_vs_Control_Sentiment_Dataset_500.xlsx")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Text"])

group_model = LogisticRegression(max_iter=1000).fit(X, df["Group"])
mood_model = LogisticRegression(max_iter=1000).fit(X, df["Mood"])
sentiment_model = LogisticRegression(max_iter=1000).fit(X, df["Sentiment"])

# ================= INPUT =================
st.subheader("✍️ Enter Feelings / Behavior (Real-Time Detection)")

user_text = st.text_area("Write behavior or feelings")
keywords = st.text_input("OR enter keywords only")

input_text = user_text.strip() if user_text.strip() else keywords.strip()

# ================= REAL-TIME ANALYSIS =================
# (Detect automatically when text is entered)
if input_text != "" and len(input_text.split()) >= 3:

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

**Summary:**  
Your emotional and behavioral pattern appears normal.

**Why no exercises?**  
Exercises are only needed when attention or emotional regulation is affected.

**What to do:**  
✔ Maintain routine  
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
You are stable and emotionally balanced.

**Why no exercises now?**  
Positive mood does not require intervention.

**What to do:**  
✔ Maintain structure  
✔ Follow sleep routine  
✔ Keep positive habits  
""")

        # ---------- SAD ----------
        elif mood == "Sad":
            st.warning("""
### 😔 ADHD + Sad Mood

**Summary:**  
Low mood with ADHD reduces motivation and focus.

**Goal:**  
✔ Gently lift mood  
✔ Improve emotional regulation  

**Exercises (How to do):**

🫁 **Slow Breathing**  
• Inhale 4 sec → Exhale 6 sec  
• Repeat 5 times  

🚶 **Light Walk**  
• Walk slowly for 10 minutes  
• No phone, focus on steps  

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
Anger often comes from overstimulation.

**Goal:**  
✔ Reduce intensity  
✔ Prevent impulsive reaction  

**Exercises (How to do):**

✋ **Muscle Relaxation**  
• Tighten fists 5 sec → release  
• Repeat 5 times  

🫁 **Deep Breathing**  
• Inhale 5 sec → Hold 2 sec → Exhale 7 sec  
• Repeat 5 rounds  

🚶 **Cool-Down Walk**  
• Walk away from trigger  
• 5–10 minutes  
""")

        # ---------- FRUSTRATED / HYPER ----------
        elif mood == "Frustrated":
            st.error("""
### ⚡ ADHD + Frustrated / Hyper Mood

**Summary:**  
High overload and hyperactivity risk.

**Goal:**  
✔ Ground attention  
✔ Reduce sensory overload  

**Exercises (How to do):**

🧠 **Grounding 5–4–3–2–1**  
• 5 things you see  
• 4 you touch  
• 3 you hear  
• 2 you smell  
• 1 you taste  

🫁 **Box Breathing**  
• Inhale 4 sec → Hold 4 sec  
• Exhale 4 sec → Hold 4 sec  
• Repeat 5 cycles  

🚶 **Controlled Movement**  
• Slow stretching  
• No running, no screens  

⚠️ If this happens daily → professional guidance advised.
""")

elif input_text != "":
    st.info("ℹ️ Please enter at least 3 words for accurate real-time analysis.")
