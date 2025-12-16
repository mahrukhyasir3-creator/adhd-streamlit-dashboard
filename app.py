import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import calendar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from PIL import Image
from fpdf import FPDF

# ---------------- UI THEME ----------------
st.set_page_config(page_title="🧠 ADHD Smart Dashboard", layout="wide")

st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1581093588401-22a6f9d6a1d4");
    background-size: cover;
}
h1 { text-align:center; color:#4b4b9f; }
div[data-testid="stVerticalBlock"] > div {
    background:white;
    padding:18px;
    border-radius:15px;
    box-shadow:0px 8px 20px rgba(0,0,0,0.1);
}
.stButton > button {
    background-color:#6a5acd;
    color:white;
    border-radius:10px;
    font-size:16px;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 ADHD Smart Monitoring Dashboard")

# ---------------- SIDEBAR ----------------
st.sidebar.title("🗓️ Daily Monitor")
now = datetime.now()
st.sidebar.markdown(f"**📅 Date:** {now.strftime('%A, %d %B %Y')}")
st.sidebar.markdown(f"**⏰ Time:** {now.strftime('%H:%M:%S')}")
st.sidebar.text(calendar.month(now.year, now.month))

st.sidebar.subheader("🔔 Daily Reminder")
st.sidebar.info("✔ Sleep well\n✔ Avoid multitasking\n✔ Take breaks\n✔ Drink water")

# ---------------- LOAD DATA ----------------
df = pd.read_excel("ADHD_vs_Control_Sentiment_Dataset_500.xlsx")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Text"])

group_model = LogisticRegression(max_iter=1000)
group_model.fit(X, df["Group"])

mood_model = LogisticRegression(max_iter=1000)
mood_model.fit(X, df["Mood"])

sentiment_model = LogisticRegression(max_iter=1000)
sentiment_model.fit(X, df["Sentiment"])

# ---------------- INPUT ----------------
st.subheader("✍️ Input Options")
col1, col2 = st.columns(2)

with col1:
    user_text = st.text_area("Enter feelings / behavior")
    keywords = st.text_input("Enter keywords")

with col2:
    img = st.file_uploader("Upload image (optional)", type=["jpg","png","jpeg"])
    if img:
        st.image(Image.open(img), width=200)

input_text = user_text if user_text.strip() else keywords

log_file = "behavior_log.csv"

def save_log(date, mood, sentiment, group, severity):
    row = pd.DataFrame([[date, mood, sentiment, group, severity]],
        columns=["Date","Mood","Sentiment","Group","Severity"])
    if os.path.exists(log_file):
        row.to_csv(log_file, mode="a", header=False, index=False)
    else:
        row.to_csv(log_file, index=False)

# ---------------- ANALYZE ----------------
if st.button("🔍 Analyze"):
    if input_text.strip()=="":
        st.warning("Please enter text or keywords")
    else:
        vec = vectorizer.transform([input_text])
        group = group_model.predict(vec)[0]
        mood = mood_model.predict(vec)[0]
        sentiment = sentiment_model.predict(vec)[0]

        # ADHD Severity
        severity = "Low"
        if group=="ADHD" and mood in ["Angry","Frustrated"]:
            severity = "High"
        elif group=="ADHD":
            severity = "Medium"

        save_log(now.strftime("%Y-%m-%d"), mood, sentiment, group, severity)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Patient Type", group)
        c2.metric("Mood", mood)
        c3.metric("Sentiment", sentiment)
        c4.metric("Severity", severity)

        st.subheader("💡 Guidance")
        if severity=="High":
            st.error("⚠️ High ADHD risk: reduce stimulation, follow routine, consult specialist.")
        elif severity=="Medium":
            st.warning("⚠️ Moderate ADHD signs: monitor focus & stress.")
        else:
            st.success("✅ Stable condition.")

# ---------------- WEEKLY DATA ----------------
st.subheader("📈 Mood Trend (Last 7 Days)")
if os.path.exists(log_file):
    log_df = pd.read_csv(log_file)
    log_df["Date"] = pd.to_datetime(log_df["Date"])
    week_df = log_df.tail(7)
    mood_map = {"Happy":1,"Sad":2,"Frustrated":3,"Angry":4}
    week_df["MoodScore"] = week_df["Mood"].map(mood_map)
    st.line_chart(week_df.set_index("Date")["MoodScore"])
    st.dataframe(week_df)

# ---------------- PDF REPORT ----------------
def create_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0,10,"Weekly ADHD Report", ln=True)
    for _,row in data.iterrows():
        pdf.cell(0,8,f"{row['Date']} | {row['Mood']} | {row['Severity']}", ln=True)
    pdf.output("weekly_report.pdf")

if st.button("📥 Download Weekly Report (PDF)"):
    if os.path.exists(log_file):
        df_pdf = pd.read_csv(log_file).tail(7)
        create_pdf(df_pdf)
        with open("weekly_report.pdf","rb") as f:
            st.download_button("Download PDF", f, file_name="ADHD_Weekly_Report.pdf")


