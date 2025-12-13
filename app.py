import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="🧠 ADHD Dashboard", layout="wide")
st.title("🧠 ADHD Smart Monitoring Dashboard")

df = pd.read_excel("ADHD_vs_Control_Sentiment_Dataset_500.xlsx")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Text"])

group_model = LogisticRegression()
group_model.fit(X, df["Group"])

mood_model = LogisticRegression()
mood_model.fit(X, df["Mood"])

sentiment_model = LogisticRegression()
sentiment_model.fit(X, df["Sentiment"])

st.subheader("✍️ Enter Text or Keywords")

user_text = st.text_area("Enter your thoughts:")
user_keywords = st.text_input("Enter keywords (optional):")

input_text = user_text if user_text.strip() != "" else user_keywords

if st.button("🔍 Analyze"):
    if input_text.strip() == "":
        st.warning("Please enter text or keywords")
    else:
        vec = vectorizer.transform([input_text])

        group = group_model.predict(vec)[0]
        mood = mood_model.predict(vec)[0]
        sentiment = sentiment_model.predict(vec)[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("Group", group)
        col2.metric("Mood", mood)
        col3.metric("Sentiment", sentiment)

        st.subheader("💡 Suggestions")

        if mood == "Happy":
            st.success("Maintain routine and positive habits 🌟")
        elif mood == "Sad":
            st.info("Take breaks and talk to someone 💙")
        elif mood == "Angry":
            st.warning("Practice breathing and reduce stress 🔕")
        elif mood == "Frustrated":
            st.warning("Break tasks into small steps 🧩")

        if group == "ADHD":
            st.markdown("""
            **ADHD Tips:**
            - Use reminders  
            - Avoid multitasking  
            - Follow sleep routine  
            """)

