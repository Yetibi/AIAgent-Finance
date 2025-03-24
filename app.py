import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import os
from groq import Groq
from dotenv import load_dotenv

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# 🎨 Streamlit UI Styling
st.set_page_config(page_title="AI Forecasting Agent", page_icon="📈", layout="wide")
st.title("📈 Forecasting with Prophet")

# File uploader
uploaded_file = st.file_uploader("Upload your Excel file with Date and Revenue columns", type=["xlsx", "xls"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    if 'Date' not in df.columns or 'Revenue' not in df.columns:
        st.error("❌ Excel file must contain 'Date' and 'Revenue' columns.")
        st.stop()

    df = df[['Date', 'Revenue']].rename(columns={'Date': 'ds', 'Revenue': 'y'})

    st.subheader("🔍 Raw Data Preview")
    st.dataframe(df.head())

    # Forecasting with Prophet
    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    st.subheader("📊 Forecast Plot")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    st.subheader("📉 Forecast Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

    # Send forecast data to AI Agent for commentary
    st.subheader("🧠 AI Forecast Commentary")
    client = Groq(api_key=GROQ_API_KEY)
    forecast_json = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_json(orient='records')

    prompt = f"""
    You are a Financial Analyst specializing in revenue forecasting. Based on the following forecast data, provide:
    - Summary of expected trends.
    - Key fluctuations or anomalies.
    - Risks or uncertainties that could affect the forecast.
    - Executive summary in a concise style.

    Here is the forecast data in JSON:
    {forecast_json}
    """

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a forecasting expert for financial planning."},
            {"role": "user", "content": prompt}
        ],
        model="llama3-8b-8192",
    )

    ai_commentary = response.choices[0].message.content

    st.markdown('<div class="analysis-container">', unsafe_allow_html=True)
    st.subheader("📝 AI Forecast Commentary")
    st.write(ai_commentary)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("📂 Please upload an Excel file to begin.")
