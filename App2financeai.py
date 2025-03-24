import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# Streamlit page config
st.set_page_config(page_title="💅 Análisis Ingresos Servicios", page_icon="💰", layout="wide")
st.title("💅 Análisis Estacional de Servicios por Sede / Manicurista")

# File upload
uploaded_file = st.file_uploader("📂 Sube el archivo Excel con la estructura indicada", type=["xlsx", "xls"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Normalize and clean data
    df = df.rename(columns={
        'FECHA': 'ds',
        'VALOR SERVICIO ($)': 'valor_servicio',
        'CATEGORIA SERVICIO': 'categoria',
        'MANICURISTA': 'manicurista',
        'SEDE': 'sede',
        'FORMA DE PAGO': 'pago',
        'TOTAL DIA': 'total_dia'
    })

    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df = df.dropna(subset=['ds', 'valor_servicio'])

    df['Month'] = df['ds'].dt.month
    df['Weekday'] = df['ds'].dt.day_name()
    df['Year'] = df['ds'].dt.year

    st.subheader("🧾 Vista previa de los datos")
    st.dataframe(df.head())

    # Segment filter
    col1, col2 = st.columns(2)
    with col1:
        selected_sede = st.selectbox("📍 Filtrar por Sede", options=['Todas'] + sorted(df['sede'].dropna().unique().tolist()))
    with col2:
        selected_manicurista = st.selectbox("💅 Filtrar por Manicurista", options=['Todas'] + sorted(df['manicurista'].dropna().unique().tolist()))

    # Apply filters
    if selected_sede != 'Todas':
        df = df[df['sede'] == selected_sede]
    if selected_manicurista != 'Todas':
        df = df[df['manicurista'] == selected_manicurista]

    # 📊 Revenue by Month
    st.subheader("📆 Promedio de Ingresos por Mes")
    monthly_avg = df.groupby('Month')['valor_servicio'].mean()
    fig1, ax1 = plt.subplots()
    monthly_avg.plot(kind='bar', color='orchid', ax=ax1)
    ax1.set_title('Promedio Ingresos por Mes')
    ax1.set_ylabel('Valor Servicio')
    st.pyplot(fig1)

    # 📆 Revenue by Weekday
    st.subheader("📅 Promedio de Ingresos por Día de la Semana")
    weekday_avg = df.groupby('Weekday')['valor_servicio'].mean().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    fig2, ax2 = plt.subplots()
    weekday_avg.plot(kind='bar', color='lightcoral', ax=ax2)
    ax2.set_title('Promedio por Día de la Semana')
    st.pyplot(fig2)

    # 📈 Ingreso Diario
    st.subheader("📈 Evolución Diaria de Ingresos")
    fig3, ax3 = plt.subplots()
    ax3.plot(df['ds'], df['valor_servicio'], marker='o', linestyle='-', color='seagreen')
    ax3.set_title("Ingresos Diarios")
    ax3.set_ylabel("Valor Servicio")
    ax3.set_xlabel("Fecha")
    st.pyplot(fig3)

    # 🧠 AI Commentary
    st.subheader("🧠 Comentario Inteligente de Estacionalidad")
    client = Groq(api_key=GROQ_API_KEY)

    resumen = df[['ds', 'valor_servicio', 'categoria', 'sede']].describe(include='all').to_string()

    prompt = f"""
    Eres un analista experto en servicios de belleza y comportamiento estacional.

    Analiza el comportamiento de ingresos según el siguiente resumen estadístico:
    
    {resumen}

    Incluye:
    - Tendencias generales
    - Patrones estacionales (meses, días)
    - Sedes o manicuristas destacadas
    - Cambios relevantes en el comportamiento
    - Recomendaciones para optimizar horarios o promociones
    """

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Eres un analista experto en negocios de belleza y servicios."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile"
    )

    ai_analysis = response.choices[0].message.content
    st.subheader("📄 Análisis Inteligente")
    st.write(ai_analysis)

else:
    st.info("⬆️ Sube un archivo Excel con columnas como FECHA, VALOR SERVICIO, SEDE...")

