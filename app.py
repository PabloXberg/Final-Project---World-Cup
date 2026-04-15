import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="FIFA WC Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# CSS PERSONALIZADO
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;600;700&family=Open+Sans:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Open Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Oswald', sans-serif !important;
        letter-spacing: 1px;
    }
    .main { background-color: #0a0a0f; }
    .stApp { background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 100%); }

    /* Métricas */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #e8c04020;
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="metric-container"] label {
        color: #e8c040 !important;
        font-family: 'Oswald', sans-serif;
        font-size: 13px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-family: 'Oswald', sans-serif;
        font-size: 2rem;
    }

    /* Cards de predicción */
    .pred-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 16px;
        padding: 28px;
        border: 1px solid #e8c04030;
        text-align: center;
        margin: 8px 0;
    }
    .pred-winner {
        border: 2px solid #e8c040;
        background: linear-gradient(135deg, #1a1a2e, #1e2d4e);
    }
    .team-name { 
        font-family: 'Oswald', sans-serif;
        font-size: 2rem;
        color: #ffffff;
        letter-spacing: 2px;
    }
    .vs-text {
        font-family: 'Oswald', sans-serif;
        font-size: 1.2rem;
        color: #e8c040;
        letter-spacing: 4px;
        margin: 20px 0;
    }
    .result-badge {
        display: inline-block;
        background: linear-gradient(90deg, #e8c040, #f0a500);
        color: #0a0a0f;
        font-family: 'Oswald', sans-serif;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 8px 24px;
        border-radius: 30px;
        letter-spacing: 2px;
        margin-top: 12px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117, #0a0a0f) !important;
        border-right: 1px solid #e8c04020;
    }
    [data-testid="stSidebar"] * { color: #e0e0e0; }

    /* Botones */
    .stButton > button {
        background: linear-gradient(90deg, #e8c040, #f0a500);
        color: #0a0a0f;
        font-family: 'Oswald', sans-serif;
        font-weight: 600;
        letter-spacing: 1.5px;
        border: none;
        border-radius: 8px;
        padding: 12px 32px;
        font-size: 15px;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px #e8c04050;
    }

    /* Chat */
    .chat-msg-user {
        background: #1a2744;
        border-left: 3px solid #e8c040;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        color: #e0e0e0;
    }
    .chat-msg-ai {
        background: #16213e;
        border-left: 3px solid #4a9eff;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        color: #e0e0e0;
    }

    /* Divisores */
    hr { border-color: #e8c04020; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-family: 'Oswald', sans-serif;
        letter-spacing: 1px;
        color: #888;
    }
    .stTabs [aria-selected="true"] {
        color: #e8c040 !important;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# CARGA DE DATOS Y MODELOS (con caché)
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    results = pd.read_csv('db/results.csv')
    rankings = pd.read_csv('db/ranking.csv')
    results['date'] = pd.to_datetime(results['date'])
    rankings['rank_date'] = pd.to_datetime(rankings['rank_date'])
    return results, rankings

@st.cache_resource
def load_models():
    modelo = joblib.load('modelo_fifa.pkl')
    return modelo

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return None
    return Groq(api_key=api_key)


# ──────────────────────────────────────────────
# FUNCIONES DE ANÁLISIS
# ──────────────────────────────────────────────
def get_form(team, date, df, n=10):
    past = df[
        ((df['home_team'] == team) | (df['away_team'] == team)) &
        (df['date'] < date)
    ].tail(n)
    if len(past) == 0:
        return 0.4
    wins = sum(
        1 for _, r in past.iterrows()
        if (r['home_team'] == team and r['home_score'] > r['away_score']) or
           (r['away_team'] == team and r['away_score'] > r['home_score'])
    )
    return wins / len(past)

def get_avg_goals(team, date, df, n=10):
    past = df[
        ((df['home_team'] == team) | (df['away_team'] == team)) &
        (df['date'] < date)
    ].tail(n)
    if len(past) == 0:
        return 1.0
    total = sum(
        r['home_score'] if r['home_team'] == team else r['away_score']
        for _, r in past.iterrows()
    )
    return total / len(past)

def get_h2h_stats(home, away, df):
    h2h = df[
        ((df['home_team'] == home) & (df['away_team'] == away)) |
        ((df['home_team'] == away) & (df['away_team'] == home))
    ].copy()
    if len(h2h) == 0:
        return None

    h2h['winner'] = h2h.apply(
        lambda r: home if (
            (r['home_team'] == home and r['home_score'] > r['away_score']) or
            (r['away_team'] == home and r['away_score'] > r['home_score'])
        ) else (away if r['home_score'] != r['away_score'] else 'Draw'),
        axis=1
    )
    return h2h

def get_ranking_at_date(team, date, rankings_df):
    team_ranks = rankings_df[
        (rankings_df['country_full'] == team) &
        (rankings_df['rank_date'] <= date)
    ]
    if len(team_ranks) == 0:
        return 100
    return int(team_ranks.sort_values('rank_date').iloc[-1]['rank'])

def build_features_for_match(home, away, results_df, rankings_df):
    today = pd.Timestamp.now()
    home_rank  = get_ranking_at_date(home, today, rankings_df)
    away_rank  = get_ranking_at_date(away, today, rankings_df)
    home_form  = get_form(home, today, results_df)
    away_form  = get_form(away, today, results_df)
    home_goals = get_avg_goals(home, today, results_df)
    away_goals = get_avg_goals(away, today, results_df)

    h2h_df = get_h2h_stats(home, away, results_df)
    if h2h_df is not None and len(h2h_df) > 0:
        total = len(h2h_df)
        hw = sum(h2h_df['winner'] == home)
        dw = sum(h2h_df['winner'] == 'Draw')
        aw = sum(h2h_df['winner'] == away)
        h2h_hw_rate = hw / total
        h2h_dr_rate = dw / total
        h2h_aw_rate = aw / total
    else:
        total, h2h_hw_rate, h2h_dr_rate, h2h_aw_rate = 0, 0.33, 0.33, 0.33

    return np.array([[
        home_rank, away_rank, home_rank - away_rank,
        home_form, away_form, home_form - away_form,
        home_goals, away_goals,
        h2h_hw_rate, h2h_dr_rate, h2h_aw_rate,
        total, 0  # is_neutral = 0 por defecto
    ]])


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style='text-align:center; padding: 20px 0 10px 0;'>
            <div style='font-size:3rem'>⚽</div>
            <div style='font-family:Oswald,sans-serif; font-size:1.3rem; 
                        color:#e8c040; letter-spacing:3px; margin-top:6px;'>
                WORLD CUP
            </div>
            <div style='font-family:Oswald,sans-serif; font-size:1rem; 
                        color:#888; letter-spacing:2px;'>
                AI PREDICTOR
            </div>
        </div>
        <hr>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "Navegación",
        ["📊 Dashboard", "🔮 Predictor", "🤖 Analista IA", "💬 Chat con Datos"],
        label_visibility="collapsed"
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    groq_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        help="Obtén tu clave gratuita en groq.com",
        value=os.getenv("GROQ_API_KEY", "")
    )
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        st.success("✅ API Key configurada")
    else:
        st.warning("Sin API Key, el chat IA no funciona")


# ──────────────────────────────────────────────
# CARGAR DATOS
# ──────────────────────────────────────────────
try:
    results, rankings = load_data()
    datos_ok = True
except Exception as e:
    st.error(f"Error cargando datos: {e}")
    datos_ok = False

try:
    modelo = load_models()
    modelo_ok = True
except:
    modelo_ok = False

wc_matches = results[results['tournament'] == 'FIFA World Cup'].copy() if datos_ok else pd.DataFrame()
todos_equipos = sorted(set(results['home_team'].unique()) | set(results['away_team'].unique())) if datos_ok else []


# ══════════════════════════════════════════════
# PÁGINA 1: DASHBOARD
# ══════════════════════════════════════════════
if menu == "📊 Dashboard":
    st.markdown("# 📊 Dashboard Histórico")
    st.markdown("*Análisis exploratorio completo de la Copa Mundial FIFA*")

    if not datos_ok:
        st.error("No se pudieron cargar los datos. Verifica que los archivos estén en la carpeta `db/`.")
        st.stop()

    # KPIs
    st.markdown("### Cifras Globales")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Partidos Internacionales", f"{len(results):,}")
    col2.metric("Partidos de Copa del Mundo", f"{len(wc_matches):,}")
    col3.metric("Selecciones", f"{len(todos_equipos):,}")
    wc_years = sorted(wc_matches['date'].dt.year.unique()) if len(wc_matches) > 0 else []
    col4.metric("Mundiales en los datos", len(wc_years))

    st.markdown("---")

    # ── Tabs de análisis ──
    tab1, tab2, tab3 = st.tabs(["🏆 Victorias por País", "📈 Tendencias Históricas", "⚔️ Head to Head"])

    with tab1:
        st.subheader("Top 15 — Equipos con más victorias en Mundiales")
        if len(wc_matches) > 0:
            wins_home = wc_matches[wc_matches['home_score'] > wc_matches['away_score']].groupby('home_team').size()
            wins_away = wc_matches[wc_matches['away_score'] > wc_matches['home_score']].groupby('away_team').size()
            total_wins = (wins_home.add(wins_away, fill_value=0)).sort_values(ascending=False).head(15)

            fig = px.bar(
                x=total_wins.values, y=total_wins.index,
                orientation='h',
                color=total_wins.values,
                color_continuous_scale=[[0, '#1a2744'], [0.5, '#e8c040'], [1, '#f0a500']],
                labels={'x': 'Victorias', 'y': ''},
                title=""
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#e0e0e0',
                showlegend=False,
                coloraxis_showscale=False,
                height=450,
                yaxis=dict(autorange='reversed')
            )
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Evolución de goles por Copa del Mundo")
        if len(wc_matches) > 0:
            wc_copy = wc_matches.copy()
            wc_copy['year'] = wc_copy['date'].dt.year
            wc_copy['total_goals'] = wc_copy['home_score'] + wc_copy['away_score']
            by_year = wc_copy.groupby('year').agg(
                total_goals=('total_goals', 'sum'),
                matches=('total_goals', 'count')
            ).reset_index()
            by_year['avg_goals'] = by_year['total_goals'] / by_year['matches']

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=by_year['year'], y=by_year['total_goals'],
                name='Goles totales', marker_color='#1a2744', yaxis='y'
            ))
            fig.add_trace(go.Scatter(
                x=by_year['year'], y=by_year['avg_goals'],
                name='Promedio por partido', line=dict(color='#e8c040', width=3),
                yaxis='y2', mode='lines+markers'
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#e0e0e0',
                legend=dict(bgcolor='rgba(0,0,0,0)'),
                yaxis=dict(title='Goles Totales', gridcolor='#ffffff10'),
                yaxis2=dict(title='Promedio/partido', overlaying='y', side='right', gridcolor='#ffffff10'),
                height=420,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Historial entre dos equipos")
        col_h, col_a = st.columns(2)
        with col_h:
            eq1 = st.selectbox("Equipo 1", todos_equipos, index=todos_equipos.index("Brazil") if "Brazil" in todos_equipos else 0, key="h2h1")
        with col_a:
            eq2 = st.selectbox("Equipo 2", todos_equipos, index=todos_equipos.index("Argentina") if "Argentina" in todos_equipos else 1, key="h2h2")

        if eq1 != eq2:
            h2h_df = get_h2h_stats(eq1, eq2, results)
            if h2h_df is not None and len(h2h_df) > 0:
                w1 = sum(h2h_df['winner'] == eq1)
                w2 = sum(h2h_df['winner'] == eq2)
                draws = sum(h2h_df['winner'] == 'Draw')

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Partidos", len(h2h_df))
                c2.metric(f"Victorias {eq1}", w1)
                c3.metric("Empates", draws)
                c4.metric(f"Victorias {eq2}", w2)

                fig_pie = go.Figure(go.Pie(
                    labels=[eq1, 'Empate', eq2],
                    values=[w1, draws, w2],
                    hole=0.55,
                    marker_colors=['#e8c040', '#555577', '#4a9eff'],
                ))
                fig_pie.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#e0e0e0',
                    showlegend=True,
                    height=320
                )
                st.plotly_chart(fig_pie, use_container_width=True)

                # Últimos 5 enfrentamientos
                st.markdown("**Últimos 5 encuentros:**")
                recent = h2h_df.sort_values('date', ascending=False).head(5)[
                    ['date', 'home_team', 'home_score', 'away_score', 'away_team', 'tournament']
                ].reset_index(drop=True)
                recent['date'] = recent['date'].dt.strftime('%d %b %Y')
                st.dataframe(recent, hide_index=True, use_container_width=True)
            else:
                st.info(f"No hay datos de enfrentamientos entre {eq1} y {eq2}.")


# ══════════════════════════════════════════════
# PÁGINA 2: PREDICTOR ML
# ══════════════════════════════════════════════
elif menu == "🔮 Predictor":
    st.markdown("# 🔮 Oráculo de la FIFA")
    st.markdown("*Predicción basada en Machine Learning con 13 features contextuales*")

    if not datos_ok:
        st.error("Datos no disponibles.")
        st.stop()

    col_l, col_r = st.columns([1, 1])
    with col_l:
        local = st.selectbox("🏠 Equipo Local", todos_equipos,
                             index=todos_equipos.index("Brazil") if "Brazil" in todos_equipos else 0)
        es_neutral = st.checkbox("⚖️ Campo neutral")
    with col_r:
        visitante = st.selectbox("✈️ Equipo Visitante", todos_equipos,
                                 index=todos_equipos.index("Argentina") if "Argentina" in todos_equipos else 1)

    st.markdown("---")

    if st.button("⚡ PREDECIR RESULTADO", type="primary"):
        if local == visitante:
            st.error("Los equipos deben ser diferentes.")
        elif not modelo_ok:
            st.warning("⚠️ Modelo no encontrado. Ejecuta primero el notebook para entrenarlo.")
        else:
            with st.spinner("Calculando..."):
                features = build_features_for_match(local, visitante, results, rankings)
                if es_neutral:
                    features[0][-1] = 1

                pred = modelo.predict(features)[0]
                proba = modelo.predict_proba(features)[0]

                # Labels: 0=away win, 1=draw, 2=home win
                labels = ['away_win', 'draw', 'home_win']
                proba_dict = dict(zip(labels, proba))

            # Resultado
            if pred == 2:
                winner, loser = local, visitante
                result_text = f"🏆 GANA {local.upper()}"
                conf = proba_dict['home_win']
            elif pred == 0:
                winner, loser = visitante, local
                result_text = f"🏆 GANA {visitante.upper()}"
                conf = proba_dict['away_win']
            else:
                winner = None
                result_text = "⚖️ EMPATE PROBABLE"
                conf = proba_dict['draw']

            # Mostrar tarjetas
            c1, c2, c3 = st.columns([2, 1, 2])
            with c1:
                card_class = "pred-card pred-winner" if pred == 2 else "pred-card"
                st.markdown(f"""
                    <div class="{card_class}">
                        <div style='font-size:3rem'>🏠</div>
                        <div class="team-name">{local}</div>
                        <div style='color:#888; margin-top:8px'>Local</div>
                        <div style='color:#e8c040; font-size:1.8rem; font-family:Oswald; margin-top:12px'>
                            {proba_dict['home_win']:.0%}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                    <div style='text-align:center; padding-top:40px'>
                        <div class="vs-text">VS</div>
                        <div style='color:#888; font-size:0.9rem; margin-top:20px'>Empate</div>
                        <div style='color:#aaa; font-size:1.4rem; font-family:Oswald'>
                            {proba_dict['draw']:.0%}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            with c3:
                card_class = "pred-card pred-winner" if pred == 0 else "pred-card"
                st.markdown(f"""
                    <div class="{card_class}">
                        <div style='font-size:3rem'>✈️</div>
                        <div class="team-name">{visitante}</div>
                        <div style='color:#888; margin-top:8px'>Visitante</div>
                        <div style='color:#e8c040; font-size:1.8rem; font-family:Oswald; margin-top:12px'>
                            {proba_dict['away_win']:.0%}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
                <div style='text-align:center; margin:28px 0'>
                    <span class='result-badge'>{result_text}</span>
                    <div style='color:#888; margin-top:10px; font-size:0.9rem'>
                        Confianza del modelo: {conf:.1%}
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Gráfico de probabilidades
            fig = go.Figure(go.Bar(
                x=[f'🏠 {local}', '⚖️ Empate', f'✈️ {visitante}'],
                y=[proba_dict['home_win'], proba_dict['draw'], proba_dict['away_win']],
                marker_color=['#e8c040' if pred == 2 else '#334466',
                              '#e8c040' if pred == 1 else '#334466',
                              '#e8c040' if pred == 0 else '#334466'],
                text=[f"{v:.1%}" for v in [proba_dict['home_win'], proba_dict['draw'], proba_dict['away_win']]],
                textposition='outside', textfont_color='#e0e0e0'
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#e0e0e0',
                yaxis=dict(tickformat='.0%', gridcolor='#ffffff10', range=[0, 1]),
                showlegend=False, height=300,
                title="Probabilidades del modelo"
            )
            st.plotly_chart(fig, use_container_width=True)

            # H2H rápido
            h2h_df = get_h2h_stats(local, visitante, results)
            if h2h_df is not None and len(h2h_df) > 0:
                st.markdown("### 📋 Contexto Histórico")
                w1 = sum(h2h_df['winner'] == local)
                w2 = sum(h2h_df['winner'] == visitante)
                dr = sum(h2h_df['winner'] == 'Draw')
                st.info(f"**Historial H2H:** {local} {w1} — {dr} empates — {w2} {visitante} (total {len(h2h_df)} partidos)")


# ══════════════════════════════════════════════
# PÁGINA 3: ANALISTA IA (POST-PREDICCIÓN)
# ══════════════════════════════════════════════
elif menu == "🤖 Analista IA":
    st.markdown("# 🤖 Analista Deportivo IA")
    st.markdown("*Análisis generado por Llama 3.3-70B (Groq) con datos históricos reales*")

    if not datos_ok:
        st.error("Datos no disponibles.")
        st.stop()

    col_l, col_r = st.columns(2)
    with col_l:
        local_ai = st.selectbox("Equipo Local", todos_equipos,
                                index=todos_equipos.index("Germany") if "Germany" in todos_equipos else 0)
    with col_r:
        visitante_ai = st.selectbox("Equipo Visitante", todos_equipos,
                                    index=todos_equipos.index("France") if "France" in todos_equipos else 1)

    idioma = st.radio("Idioma del análisis", ["Español", "English"], horizontal=True)
    profundidad = st.select_slider("Profundidad del análisis", ["Breve", "Estándar", "Detallado"], value="Estándar")

    if st.button("🎙️ GENERAR ANÁLISIS", type="primary"):
        if local_ai == visitante_ai:
            st.error("Los equipos deben ser diferentes.")
        else:
            client = get_groq_client()
            if not client:
                st.error("Configura tu Groq API Key en la barra lateral para usar esta función.")
            else:
                with st.spinner("El analista IA está preparando su análisis..."):
                    # Recopilar datos para el contexto
                    h2h_df = get_h2h_stats(local_ai, visitante_ai, results)
                    today = pd.Timestamp.now()
                    home_form = get_form(local_ai, today, results)
                    away_form = get_form(visitante_ai, today, results)
                    home_rank = get_ranking_at_date(local_ai, today, rankings)
                    away_rank = get_ranking_at_date(visitante_ai, today, rankings)

                    h2h_summary = "Sin datos H2H disponibles"
                    if h2h_df is not None and len(h2h_df) > 0:
                        w1 = sum(h2h_df['winner'] == local_ai)
                        w2 = sum(h2h_df['winner'] == visitante_ai)
                        dr = sum(h2h_df['winner'] == 'Draw')
                        h2h_summary = f"{local_ai}: {w1} victorias | Empates: {dr} | {visitante_ai}: {w2} victorias (de {len(h2h_df)} partidos)"

                    tokens_map = {"Breve": 200, "Estándar": 400, "Detallado": 700}
                    lang_instruction = "en español" if idioma == "Español" else "in English"

                    prompt = f"""Eres un analista deportivo experto en fútbol internacional con acceso a datos estadísticos reales.

DATOS OBJETIVOS DEL PARTIDO:
- Equipos: {local_ai} (local) vs {visitante_ai} (visitante)
- Ranking FIFA actual: {local_ai} #{home_rank} | {visitante_ai} #{away_rank}
- Forma reciente (% victorias últimos 10 partidos): {local_ai} {home_form:.0%} | {visitante_ai} {away_form:.0%}
- Historial H2H completo: {h2h_summary}

INSTRUCCIONES:
- Escribe un análisis deportivo profesional {lang_instruction}
- Basa tu análisis ÚNICAMENTE en los datos proporcionados
- Incluye: contexto histórico, estado actual de forma, factor psicológico del H2H, y una conclusión
- Longitud: {'2 párrafos concisos' if profundidad == 'Breve' else '3 párrafos' if profundidad == 'Estándar' else '4-5 párrafos detallados'}
- Tono: periodístico, profesional, apasionado"""

                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=tokens_map[profundidad],
                        temperature=0.7
                    )
                    analisis = response.choices[0].message.content

                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #1a1a2e, #16213e);
                                border: 1px solid #e8c04030; border-radius: 16px;
                                padding: 28px; margin-top: 20px;'>
                        <div style='color:#e8c040; font-family:Oswald; font-size:1rem; 
                                    letter-spacing:2px; margin-bottom:16px;'>
                            🎙️ ANÁLISIS — {local_ai.upper()} vs {visitante_ai.upper()}
                        </div>
                        <div style='color:#e0e0e0; line-height:1.8; white-space: pre-wrap;'>{analisis}</div>
                    </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PÁGINA 4: CHAT CON DATOS
# ══════════════════════════════════════════════
elif menu == "💬 Chat con Datos":
    st.markdown("# 💬 Chat con los Datos Históricos")
    st.markdown("*Pregúntale cualquier cosa sobre la historia del fútbol mundial — el LLM consulta los datos reales*")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Historial de chat
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-msg-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-msg-ai">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

    # Input
    pregunta = st.chat_input("Ej: ¿Cuántas veces ganó Brasil el Mundial? ¿Quién tiene mejor record contra Alemania?")

    if pregunta:
        if not datos_ok:
            st.error("Datos no disponibles.")
        else:
            client = get_groq_client()
            if not client:
                st.error("Configura tu Groq API Key en la barra lateral.")
            else:
                st.session_state.messages.append({"role": "user", "content": pregunta})
                st.markdown(f'<div class="chat-msg-user">👤 {pregunta}</div>', unsafe_allow_html=True)

                with st.spinner("Consultando datos..."):
                    # Extraer estadísticas relevantes para el contexto
                    wc = results[results['tournament'] == 'FIFA World Cup'].copy()

                    wins_home = wc[wc['home_score'] > wc['away_score']].groupby('home_team').size()
                    wins_away = wc[wc['away_score'] > wc['home_score']].groupby('away_team').size()
                    total_wins = wins_home.add(wins_away, fill_value=0).sort_values(ascending=False).head(10)

                    context_data = f"""
DATOS HISTÓRICOS DISPONIBLES (Copa del Mundo FIFA):
- Total de partidos en el dataset: {len(wc):,}
- Período: {wc['date'].min().year} - {wc['date'].max().year}

TOP 10 EQUIPOS POR VICTORIAS EN MUNDIALES:
{total_wins.to_string()}

DATOS GENERALES:
- Total partidos internacionales en dataset: {len(results):,}
- Número de selecciones distintas: {len(todos_equipos):,}
                    """

                    # Si mencionan un equipo específico, agregar su historial
                    for equipo in todos_equipos:
                        if equipo.lower() in pregunta.lower():
                            eq_matches = wc[(wc['home_team'] == equipo) | (wc['away_team'] == equipo)]
                            if len(eq_matches) > 0:
                                eq_wins = len(eq_matches[
                                    ((eq_matches['home_team'] == equipo) & (eq_matches['home_score'] > eq_matches['away_score'])) |
                                    ((eq_matches['away_team'] == equipo) & (eq_matches['away_score'] > eq_matches['home_score']))
                                ])
                                context_data += f"\n\nESTADÍSTICAS DE {equipo.upper()} EN MUNDIALES:\n"
                                context_data += f"- Partidos jugados: {len(eq_matches)}\n"
                                context_data += f"- Victorias: {eq_wins}\n"
                                context_data += f"- Goles marcados: {int(eq_matches[eq_matches['home_team']==equipo]['home_score'].sum() + eq_matches[eq_matches['away_team']==equipo]['away_score'].sum())}\n"

                    # Historial de conversación para contexto
                    messages_for_api = [
                        {
                            "role": "system",
                            "content": f"""Eres un experto en historia del fútbol mundial con acceso a una base de datos real de partidos.
Responde en el mismo idioma que el usuario.
Usa los datos proporcionados para dar respuestas precisas y fundamentadas.
Si no tienes el dato exacto, dilo honestamente y da la mejor aproximación posible.

{context_data}"""
                        }
                    ]
                    for m in st.session_state.messages[-6:]:  # Últimos 6 mensajes para contexto
                        messages_for_api.append({"role": m["role"], "content": m["content"]})

                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=messages_for_api,
                        max_tokens=500,
                        temperature=0.5
                    )
                    respuesta = response.choices[0].message.content

                st.session_state.messages.append({"role": "assistant", "content": respuesta})
                st.markdown(f'<div class="chat-msg-ai">🤖 {respuesta}</div>', unsafe_allow_html=True)
                st.rerun()

    if st.button("🗑️ Limpiar chat"):
        st.session_state.messages = []
        st.rerun()
