import streamlit as st
import pandas as pd
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
    background: rgba(255,255,255,0.95);
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    margin-bottom: 20px;
}

h1 { text-align:center; color:#4b4b9f; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 ADHD Smart Support Dashboard")
st.caption("Mood-based guidance & exercises (for ADHD users only)")

# ================= SIDEBAR (DATE & TIME) =================
st.sidebar.title("🗓️ Daily Monitor")
now = datetime.now()
st.sidebar.markdown(f"**📅 Date:** {now.strftime('%A, %d %B %Y')}")
st.sidebar.markdown(f"**⏰ Time:** {now.strftime('%H:%M:%S')}")

st.sidebar.subheader("🔔 Daily Reminder")
st.sidebar.info("""
• One task at a time  
• Take short breaks  
• Avoid overload  
• Sleep well  
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

        st.subheader("🧭 Guidance")

        # ========== CONTROL ==========
        if group == "Control":
            st.success(f"""
### ✅ Normal Pattern ({mood})
No ADHD intervention needed.

✔ Maintain routine  
✔ Balanced work  
✔ Proper rest  
""")

        # ========== ADHD ==========
        if group == "ADHD":

            # ---------- HAPPY ----------
            if mood == "Happy":
                st.success("""
### 😊 ADHD + Happy Mood
You are stable.

✔ Continue routine  
✔ Keep positive habits  
✔ No exercises required  
""")

            # ---------- SAD ----------
            elif mood == "Sad":
                st.warning("""
### 😔 ADHD + Sad Mood

**Why:** Low energy, low motivation, emotional heaviness.

#### 🫁 Exercise 1: Slow Breathing
1. Sit comfortably  
2. Inhale through nose for **4 seconds**  
3. Exhale slowly through mouth for **6 seconds**  
4. Repeat **5 times**

#### 🚶 Exercise 2: Light Walk
1. Walk slowly for **10 minutes**  
2. Focus on breathing  
3. No phone while walking

#### 🧠 Exercise 3: Thought Release
1. Write feelings on paper  
2. Do NOT judge the thoughts  
3. Tear the paper after writing
""")

            # ---------- ANGRY ----------
            elif mood == "Angry":
                st.error("""
### 😠 ADHD + Angry Mood

**Why:** Over-stimulation and emotional overload.

#### ✋ Exercise 1: Muscle Relaxation
1. Tighten fists for **5 seconds**  
2. Release slowly  
3. Repeat **5 times**

#### 🫁 Exercise 2: Deep Breathing
1. Inhale for **5 seconds**  
2. Hold for **2 seconds**  
3. Exhale for **7 seconds**  
4. Repeat **5 rounds**

#### 🚶 Exercise 3: Cool-Down Walk
1. Walk away from trigger  
2. Walk for **5–10 minutes**  
3. Do not talk until calm
""")

            # ---------- FRUSTRATED / HYPER ----------
            elif mood == "Frustrated":
                st.error("""
### ⚡ ADHD + Frustrated / Hyper Mood
🚨 **High Hyperactivity Risk**

#### 🧠 Exercise 1: Grounding (5-4-3-2-1)
• Name **5 things** you see  
• **4 things** you touch  
• **3 things** you hear  
• **2 things** you smell  
• **1 thing** you taste  

#### 🫁 Exercise 2: Box Breathing
1. Inhale **4 sec**  
2. Hold **4 sec**  
3. Exhale **4 sec**  
4. Hold **4 sec**  
5. Repeat **5 cycles**

#### 🚶 Exercise 3: Controlled Movement
• Slow stretching  
• No running  
• No screen use  

⚠️ If this happens daily → seek professional help
""")

# ================= END =================
