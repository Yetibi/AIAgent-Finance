import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq
from dotenv import load_dotenv
import os

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# Streamlit UI Styling
st.set_page_config(page_title="📊 Revenue Seasonality Analyzer", page_icon="📅", layout="wide")
st.title("📅 Seasonality & Cycle Analysis of Revenue")

uploaded_file = st.file_uploader("📂 Upload your Excel file with 'Date' and 'Revenue' columns", type=["xlsx", "xls"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    if 'Date' not in df.columns or 'Revenue' not in df.columns:
        st.error("❌ Excel file must contain 'Date' and 'Revenue' columns.")
        st.stop()

    df = df[['Date', 'Revenue']].dropna()
    df = df.rename(columns={'Date': 'ds', 'Revenue': 'y'})
    df['ds'] = pd.to_datetime(df['ds'])
    df['Month'] = df['ds'].dt.month
    df['Weekday'] = df['ds'].dt.day_name()
    df['Year'] = df['ds'].dt.year

    st.subheader("📋 Raw Data Preview")
    st.dataframe(df.head())

    # Seasonality visualizations
    st.subheader("📊 Revenue by Month (Seasonality)")
    monthly_avg = df.groupby('Month')['y'].mean()
    fig1, ax1 = plt.subplots()
    monthly_avg.plot(kind='bar', color='skyblue', ax=ax1)
    ax1.set_title('Average Revenue by Month')
    ax1.set_ylabel('Revenue')
    st.pyplot(fig1)

    st.subheader("📆 Revenue by Day of Week")
    weekday_avg = df.groupby('Weekday')['y'].mean().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    fig2, ax2 = plt.subplots()
    weekday_avg.plot(kind='bar', color='salmon', ax=ax2)
    ax2.set_title('Average Revenue by Weekday')
    ax2.set_ylabel('Revenue')
    st.pyplot(fig2)

    st.subheader("📈 Revenue Over Time")
    fig3, ax3 = plt.subplots()
    ax3.plot(df['ds'], df['y'], color='green')
    ax3.set_title("Revenue Trend Over Time")
    ax3.set_ylabel("Revenue")
    ax3.set_xlabel("Date")
    st.pyplot(fig3)

    # AI Analysis
    st.subheader("🧠 AI-Powered Commentary on Seasonality")
    client = Groq(api_key=GROQ_API_KEY)

    prompt = f"""
    Actúa como un experto en análisis financiero estacional.

    Tu tarea es analizar el comportamiento histórico de ingresos de una empresa.  
    Identifica:
    - Tendencias generales de crecimiento o decrecimiento.
    - Patrones estacionales (mensuales, semanales).
    - Meses o fechas con anomalías.
    - Cualquier ciclo repetitivo importante.
    - Riesgos ocultos en la variabilidad.

    
    """

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a financial seasonality analysis expert."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile"
    )

    ai_output = response.choices[0].message.content
    st.markdown("<div class='ai-commentary'>", unsafe_allow_html=True)
    st.subheader("📄 AI Commentary")
    st.write(ai_output)
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("⬆️ Please upload your Excel file to begin the analysis.")
