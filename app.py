import streamlit as st
import pandas as pd
import joblib
from transformers import pipeline

# Configuración de la página (Debe ser la primera línea de Streamlit)
st.set_page_config(page_title="FIFA World Cup AI", page_icon="⚽", layout="wide")

# --- CACHE PARA OPTIMIZAR VELOCIDAD ---
@st.cache_data
def load_data():
    cups = pd.read_csv(r'C:\Users\pablo\Desktop\Data Science\Final Project - World Cup\Final-Project---World-Cup\db\WorldCups.csv')
    matches = pd.read_csv(r'C:\Users\pablo\Desktop\Data Science\Final Project - World Cup\Final-Project---World-Cup\db\WorldCupMatches.csv').dropna(how='all')
    matches['Home Team Name'] = matches['Home Team Name'].str.replace('rn">', '').str.strip()
    matches['Away Team Name'] = matches['Away Team Name'].str.replace('rn">', '').str.strip()
    return cups, matches

@st.cache_resource
def load_models():
    modelo = joblib.load('modelo_fifa.pkl')
    encoder = joblib.load('label_encoder.pkl')
    # Cargamos el LLM una sola vez en caché para que la app no sea lenta
    genai = pipeline('text-generation', model='distilgpt2')
    return modelo, encoder, genai

# Cargar todo
cups, matches = load_data()

try:
    modelo, encoder, genai = load_models()
    modelos_listos = True
except Exception as e:
    modelos_listos = False
    st.error(f"🚨 DETALLE DEL ERROR: {e}") # Esto mostrará el error real en la pantalla

# --- SIDEBAR (Menú de navegación) ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa/FIFA_logo_without_slogan.svg/1200px-FIFA_logo_without_slogan.svg.png", width=150)
st.sidebar.title("Navegación")
menu = st.sidebar.radio("Ir a:", ["📊 Dashboard Exploratorio", "🔮 Predicción de Partidos", "🤖 Periodista Deportivo IA"])

# --- PÁGINA 1: DASHBOARD ---
if menu == "📊 Dashboard Exploratorio":
    st.title("📊 Análisis Histórico de los Mundiales")
    st.markdown("Explora los datos históricos de la Copa Mundial de la FIFA.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Mundiales", len(cups))
    col2.metric("Goles Históricos", cups['GoalsScored'].sum())
    col3.metric("Asistencia Total", "40M+") # Aproximación visual

    st.subheader("Goles marcados por Año")
    # Gráfico nativo de Streamlit
    chart_data = cups[['Year', 'GoalsScored']].set_index('Year')
    st.bar_chart(chart_data)

    st.subheader("Datos Crudos (Matches)")
    st.dataframe(matches[['Year', 'Home Team Name', 'Home Team Goals', 'Away Team Goals', 'Away Team Name']].head(10))

# --- PÁGINA 2: MACHINE LEARNING ---
elif menu == "🔮 Predicción de Partidos":
    st.title("🔮 Oráculo de la FIFA")
    st.markdown("Usa nuestro modelo de **Machine Learning (XGBoost)** para predecir el resultado de un partido.")
    
    if not modelos_listos:
        st.warning("⚠️ No se encontraron los archivos 'modelo_fifa.pkl' o 'label_encoder.pkl'. Asegúrate de ejecutar la última celda de tu Jupyter Notebook.")
    else:
        equipos = sorted(matches['Home Team Name'].unique().tolist())
        
        col1, col2 = st.columns(2)
        with col1:
            local = st.selectbox("Equipo Local", equipos, index=equipos.index("Argentina") if "Argentina" in equipos else 0)
            es_anfitrion = st.checkbox("¿Es el país anfitrión?")
        with col2:
            visitante = st.selectbox("Equipo Visitante", equipos, index=equipos.index("France") if "France" in equipos else 1)
            
        if st.button("Predecir Resultado", type="primary"):
            if local == visitante:
                st.error("Un equipo no puede jugar contra sí mismo.")
            else:
                try:
                    loc_id = encoder.transform([local])[0]
                    vis_id = encoder.transform([visitante])[0]
                    host_val = 1 if es_anfitrion else 0
                    
                    # Predicción
                    pred = modelo.predict([[loc_id, vis_id, host_val]])[0]
                    
                    if pred == 2:
                        st.success(f"🏆 Pronóstico: Gana **{local}**")
                    elif pred == 0:
                        st.success(f"🏆 Pronóstico: Gana **{visitante}**")
                    else:
                        st.info("⚖️ Pronóstico: **Empate**")
                except Exception as e:
                    st.error(f"Error en la predicción. Asegúrate de que los equipos existen en el modelo. Error: {e}")

# --- PÁGINA 3: GEN AI ---
elif menu == "🤖 Periodista Deportivo IA":
    st.title("🤖 Generador de Crónicas (GenAI)")
    st.markdown("Usamos **Hugging Face Transformers** para redactar automáticamente el resumen del partido.")
    
    local_ai = st.text_input("Equipo Local", "Brazil")
    goles_l = st.number_input("Goles Local", min_value=0, value=2)
    visitante_ai = st.text_input("Equipo Visitante", "Germany")
    goles_v = st.number_input("Goles Visitante", min_value=0, value=1)
    
    if st.button("Generar Crónica"):
        with st.spinner("La IA está escribiendo el artículo..."):
            prompt = f"In the FIFA World Cup match between {local_ai} and {visitante_ai}, the final score was {goles_l} - {goles_v}. The match was very exciting because"
            
            resultado = genai(prompt, max_new_tokens=60, num_return_sequences=1, pad_token_id=50256)
            
            st.text_area("Crónica del Partido:", value=resultado[0]['generated_text'], height=200)