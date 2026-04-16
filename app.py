import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import time
from wc2026_game import logo_html, render_wc2026_game
from openai import OpenAI
from dotenv import load_dotenv
import base64
from pathlib import Path


load_dotenv()

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="FIFA WC Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;600;700&family=Open+Sans:wght@400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Open Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Oswald', sans-serif !important; letter-spacing: 1px; }
    .main { background-color: #0a0a0f; }
    .stApp { background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 100%); }

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
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117, #0a0a0f) !important;
        border-right: 1px solid #e8c04020;
    }
    [data-testid="stSidebar"] * { color: #e0e0e0; }
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
    hr { border-color: #e8c04020; }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Oswald', sans-serif;
        letter-spacing: 1px;
        color: #888;
    }
    .stTabs [aria-selected="true"] { color: #e8c040 !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# DATA & MODEL LOADING (cached)
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    results  = pd.read_csv('db/results.csv')
    rankings = pd.read_csv('db/ranking.csv')
    results['date']       = pd.to_datetime(results['date'])
    rankings['rank_date'] = pd.to_datetime(rankings['rank_date'])
    results = results.dropna(subset=['home_score', 'away_score']).copy()
    # Add result label
    def classify(row):
        if row['home_score'] > row['away_score']:   return 'Home Win'
        elif row['home_score'] == row['away_score']: return 'Draw'
        else:                                        return 'Away Win'
    results['result_label'] = results.apply(classify, axis=1)
    return results, rankings

@st.cache_resource
def load_model():
    return joblib.load('modelo_fifa.pkl')


# ──────────────────────────────────────────────
# LLM CLIENT & FALLBACK SYSTEM (OpenRouter)
# ──────────────────────────────────────────────
def get_openrouter_client(api_key):
    if not api_key:
        return None
    return OpenAI(base_url='https://openrouter.ai/api/v1', api_key=api_key)

FALLBACK_MODELS = [
    'meta-llama/llama-3.3-70b-instruct:free',
    'nvidia/nemotron-3-super-120b-a12b:free',
    'meta-llama/llama-3.2-3b-instruct:free',
    'google/gemma-3-27b-it:free',
    'google/gemma-3-12b-it:free',
    'google/gemma-3-4b-it:free',
]

def adapt_messages(messages, model):
    """Convert 'system' role to 'user' for models that don't support it (e.g. Gemma)."""
    if 'gemma' not in model:
        return messages
    adapted, sys_content = [], ''
    for msg in messages:
        if msg['role'] == 'system':
            sys_content = msg['content'] + '\n\n'
        elif msg['role'] == 'user' and sys_content:
            adapted.append({'role': 'user', 'content': sys_content + msg['content']})
            sys_content = ''
        else:
            adapted.append(msg)
    return adapted

def call_llm(client, messages, max_tokens=500, temperature=0.7):
    """Call LLM with automatic fallback across multiple free models."""
    for model in FALLBACK_MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=adapt_messages(messages, model),
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            if '429' in str(e) or '400' in str(e) or 'rate' in str(e).lower():
                time.sleep(2)
                continue
            else:
                raise e
    return '⚠️ Analysis temporarily unavailable (all models rate-limited). Please try again.'


# ──────────────────────────────────────────────
# FEATURE COMPUTATION FUNCTIONS
# ──────────────────────────────────────────────
def get_form(team, date, df, n=10):
    """Win percentage in the team's last n matches before date."""
    past = df[
        ((df['home_team'] == team) | (df['away_team'] == team)) & (df['date'] < date)
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
    """Average goals scored in the team's last n matches."""
    past = df[
        ((df['home_team'] == team) | (df['away_team'] == team)) & (df['date'] < date)
    ].tail(n)
    if len(past) == 0:
        return 1.0
    total = sum(
        r['home_score'] if r['home_team'] == team else r['away_score']
        for _, r in past.iterrows()
    )
    return total / len(past)

def get_h2h_stats(home, away, df):
    """Full head-to-head match history between two teams."""
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
    """Most recent FIFA ranking before date."""
    tr = rankings_df[
        (rankings_df['country_full'] == team) & (rankings_df['rank_date'] <= date)
    ]
    if len(tr) == 0:
        return 100
    return int(tr.sort_values('rank_date').iloc[-1]['rank'])

def build_features(home, away, results_df, rankings_df, is_neutral=False):
    """Build the 13-feature vector for a match prediction."""
    today      = pd.Timestamp.now()
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
        hw_rate, dr_rate, aw_rate = hw/total, dw/total, aw/total
    else:
        total, hw_rate, dr_rate, aw_rate = 0, 0.33, 0.33, 0.33

    return np.array([[
        home_rank, away_rank, home_rank - away_rank,
        home_form, away_form, home_form - away_form,
        home_goals, away_goals,
        hw_rate, dr_rate, aw_rate,
        total, int(is_neutral)
    ]])

def local_img_b64(path, height=120):
    """Embed a local image as base64 HTML."""
    try:
        with open(path, 'rb') as f:
            data = base64.b64encode(f.read()).decode()
        ext = Path(path).suffix.lstrip('.')
        return f'<img src="data:image/{ext};base64,{data}" style="height:{height}px; border-radius:8px;" />'
    except Exception:
        return '⚽'

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
        <div style='text-align:center; padding: 20px 0 10px 0;'>
            <div style='display:flex; justify-content:center; margin-bottom:10px;'>
                {logo_html('WC2026', 80)}
            </div>
            <div style='font-family:Oswald,sans-serif; font-size:1.3rem;
                        color:#e8c040; letter-spacing:3px;'>
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
        "Navigation",
        ["📊 Dashboard", "🔮 Predictor", "🏆 WC26 Predictor", "🤖 AI Analyst" ],
        label_visibility="collapsed"
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    if os.getenv("OPENROUTER_API_KEY"):
        st.markdown(
            "<div style='text-align:center; color:#4a9eff; font-size:0.85rem;'>🤖 AI features enabled</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='text-align:center; color:#888; font-size:0.85rem;'>⚠️ Add OPENROUTER_API_KEY to .env</div>",
            unsafe_allow_html=True
        )



# ──────────────────────────────────────────────
# LOAD DATA & MODEL
# ──────────────────────────────────────────────
try:
    results, rankings = load_data()
    data_ok = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    data_ok = False

try:
    model = load_model()
    model_ok = True
except Exception:
    model_ok = False

wc_matches = results[results['tournament'] == 'FIFA World Cup'].copy() if data_ok else pd.DataFrame()
all_teams  = sorted(set(results['home_team'].unique()) | set(results['away_team'].unique())) if data_ok else []
llm_client = get_openrouter_client(os.getenv("OPENROUTER_API_KEY", ""))


# ══════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ══════════════════════════════════════════════
if menu == "📊 Dashboard":
    st.markdown("# 📊 Historical Dashboard")
    st.markdown("*Exploratory analysis of FIFA World Cup history*")

    if not data_ok:
        st.error("Could not load data. Make sure `db/results.csv` and `db/ranking.csv` exist.")
        st.stop()

    # KPIs
    st.markdown("### Global Stats")
    col1, col2, col3, col4 = st.columns(4)
    wc_years = sorted(wc_matches['date'].dt.year.unique()) if len(wc_matches) > 0 else []
    col1.metric("International Matches", f"{len(results):,}")
    col2.metric("World Cup Matches", f"{len(wc_matches):,}")
    col3.metric("National Teams", f"{len(all_teams):,}")
    col4.metric("World Cup Editions", len(wc_years))

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab7 = st.tabs([
        "🏆 Wins by Nation",
        "🎯 Goal Scorers (Top Teams)",
        "📈 Historical Trends",
        "⚔️ Head to Head",
        # "🌍 Confederation Power",
        # "⚡ Biggest Upsets",
        "🏟️ Tournament Map"
    ])

    with tab1:
        st.subheader("Top 15 — All-Time World Cup Wins")
        wins_home  = wc_matches[wc_matches['home_score'] > wc_matches['away_score']].groupby('home_team').size()
        wins_away  = wc_matches[wc_matches['away_score'] > wc_matches['home_score']].groupby('away_team').size()
        total_wins = wins_home.add(wins_away, fill_value=0).sort_values(ascending=False).head(15)

        fig = px.bar(
            x=total_wins.values, y=total_wins.index,
            orientation='h', color=total_wins.values,
            color_continuous_scale=[[0, '#1a2744'], [0.5, '#e8c040'], [1, '#f0a500']],
            labels={'x': 'Wins', 'y': ''}
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e0e0e0', showlegend=False,
            coloraxis_showscale=False, height=450,
            yaxis=dict(autorange='reversed')
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, width='stretch')

        # ── NEW TAB 2: Top Scorer Teams ──
    with tab2:
        st.subheader("Most Goals Scored — Top 15 Nations")

        goals_home = wc_matches.groupby('home_team')['home_score'].sum()
        goals_away = wc_matches.groupby('away_team')['away_score'].sum()
        total_goals = goals_home.add(goals_away, fill_value=0).sort_values(ascending=False).head(15)

        fig = px.bar(
            x=total_goals.values, y=total_goals.index,
            orientation='h',
            color=total_goals.values,
            color_continuous_scale=[[0, '#1a2744'], [0.5, '#5cdb95'], [1, '#e8c040']],
            labels={'x': 'Total Goals', 'y': ''}
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e0e0e0', showlegend=False,
            coloraxis_showscale=False, height=500,
            yaxis=dict(autorange='reversed')
        )
        st.plotly_chart(fig, width='stretch')

    with tab3:
        st.subheader("Goals per World Cup Edition")
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
            name='Total Goals', marker_color='#1a2744', yaxis='y'
        ))
        fig.add_trace(go.Scatter(
            x=by_year['year'], y=by_year['avg_goals'],
            name='Average per Match', line=dict(color='#e8c040', width=3),
            yaxis='y2', mode='lines+markers'
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e0e0e0', legend=dict(bgcolor='rgba(0,0,0,0)'),
            yaxis=dict(title='Total Goals', gridcolor='rgba(255,255,255,0.06)'),
            yaxis2=dict(title='Avg per Match', overlaying='y', side='right', gridcolor='rgba(255,255,255,0.06)'),
            height=420, hovermode='x unified'
        )
        st.plotly_chart(fig, width='stretch')



    with tab4:
        st.subheader("Head-to-Head History")
        col_h, col_a = st.columns(2)
        with col_h:
            eq1 = st.selectbox("Team 1", all_teams,
                               index=all_teams.index("Brazil") if "Brazil" in all_teams else 0, key="h2h1")
        with col_a:
            eq2 = st.selectbox("Team 2", all_teams,
                               index=all_teams.index("Argentina") if "Argentina" in all_teams else 1, key="h2h2")

        if eq1 != eq2:
            h2h_df = get_h2h_stats(eq1, eq2, results)
            if h2h_df is not None and len(h2h_df) > 0:
                w1    = sum(h2h_df['winner'] == eq1)
                w2    = sum(h2h_df['winner'] == eq2)
                draws = sum(h2h_df['winner'] == 'Draw')

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Matches", len(h2h_df))
                c2.metric(f"{eq1} Wins", w1)
                c3.metric("Draws", draws)
                c4.metric(f"{eq2} Wins", w2)

                fig_pie = go.Figure(go.Pie(
                    labels=[eq1, 'Draw', eq2], values=[w1, draws, w2],
                    hole=0.55, marker_colors=['#e8c040', '#555577', '#4a9eff'],
                ))
                fig_pie.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0',
                    showlegend=True, height=320
                )
                st.plotly_chart(fig_pie, width='stretch')

                st.markdown("**Last 5 meetings:**")
                recent = h2h_df.sort_values('date', ascending=False).head(5)[
                    ['date', 'home_team', 'home_score', 'away_score', 'away_team', 'tournament']
                ].reset_index(drop=True)
                recent['date'] = recent['date'].dt.strftime('%d %b %Y')
                st.dataframe(recent, hide_index=True, width='stretch')
            else:
                st.info(f"No match data found between {eq1} and {eq2}.")



                    # ── NEW TAB 4: Confederation Power ──
    # with tab4:
    #     st.subheader("World Cup Wins by Confederation")
    #     st.caption("Which continent dominates the World Cup?")

    #     # Mapping of countries to confederations
    #     CONMEBOL = ['Brazil','Argentina','Uruguay','Colombia','Chile','Peru','Paraguay','Ecuador','Bolivia','Venezuela']
    #     UEFA     = ['Germany','Italy','France','Spain','England','Netherlands','Portugal','Belgium','Croatia','Sweden',
    #                 'Denmark','Switzerland','Austria','Poland','Russia','Czech Republic','Czechia','Hungary','Romania',
    #                 'Norway','Scotland','Wales','Republic of Ireland','Northern Ireland','Yugoslavia','Soviet Union',
    #                 'Bulgaria','Türkiye','Turkey','Ukraine','Greece','Serbia','Slovakia','Bosnia-Herzegovina','Iceland']
    #     CONCACAF = ['Mexico','USA','Canada','Costa Rica','Honduras','Jamaica','Trinidad and Tobago','Cuba','Haiti','Panama','El Salvador','Curaçao']
    #     CAF      = ['Cameroon','Nigeria','Senegal','Ghana','Algeria','Tunisia','Morocco','Egypt','South Africa',"Côte d'Ivoire",'Congo DR','Cabo Verde','Angola','Zaire']
    #     AFC      = ['South Korea','Korea Republic','Japan','Australia','Saudi Arabia','Iran','IR Iran','Iraq','Qatar','Jordan','China PR','Kuwait','UAE','Uzbekistan','North Korea']
    #     OFC      = ['New Zealand','Australia']

    #     def get_confed(team):
    #         if team in CONMEBOL: return 'CONMEBOL'
    #         if team in UEFA:     return 'UEFA'
    #         if team in CONCACAF: return 'CONCACAF'
    #         if team in CAF:      return 'CAF'
    #         if team in AFC:      return 'AFC'
    #         if team in OFC:      return 'OFC'
    #         return 'Other'

    #     wc_w = wc_matches.copy()
    #     wc_w['winner'] = wc_w.apply(
    #         lambda r: r['home_team'] if r['home_score'] > r['away_score']
    #                   else r['away_team'] if r['away_score'] > r['home_score']
    #                   else 'Draw', axis=1
    #     )
    #     wc_w = wc_w[wc_w['winner'] != 'Draw']
    #     wc_w['confederation'] = wc_w['winner'].apply(get_confed)
    #     confed_wins = wc_w['confederation'].value_counts()

    #     fig = px.pie(
    #         values=confed_wins.values,
    #         names=confed_wins.index,
    #         color_discrete_sequence=['#e8c040', '#4a9eff', '#ff6b6b', '#b59cff', '#5cdb95', '#666']
    #     )
    #     fig.update_traces(textfont_size=14, textfont_color='white', textposition='inside')
    #     fig.update_layout(
    #         paper_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0', height=420,
    #         legend=dict(font=dict(size=12))
    #     )
    #     st.plotly_chart(fig, width='stretch')

    # ── NEW TAB 5: Biggest Upsets ──
    # with tab5:
    #     st.subheader("Top 15 Biggest Goal-Difference Wins in World Cup History")
    #     st.caption("The most one-sided World Cup matches ever recorded")

    #     wc_upsets = wc_matches.copy()
    #     wc_upsets['goal_diff'] = abs(wc_upsets['home_score'] - wc_upsets['away_score'])
    #     wc_upsets['date_str'] = wc_upsets['date'].dt.strftime('%Y')
    #     wc_upsets['matchup']  = (
    #         wc_upsets['home_team'] + ' ' + wc_upsets['home_score'].astype(int).astype(str) +
    #         ' - ' + wc_upsets['away_score'].astype(int).astype(str) + ' ' + wc_upsets['away_team']
    #     )
    #     top_upsets = wc_upsets.nlargest(15, 'goal_diff')[['date_str', 'matchup', 'goal_diff', 'tournament']]
    #     top_upsets.columns = ['Year', 'Match', 'Goal Difference', 'Tournament']
    #     st.dataframe(top_upsets.reset_index(drop=True), width='stretch', hide_index=True)

    

    # ── NEW TAB 7: Tournament Map ──
    with tab7:
        st.subheader("World Cup Host Nations Across History")

# Map country names to ISO-3 codes
    COUNTRY_ISO3 = {
        'Brazil': 'BRA', 'Germany': 'DEU', 'Italy': 'ITA', 'Argentina': 'ARG',
        'France': 'FRA', 'England': 'ENG', 'Spain': 'ESP', 'Netherlands': 'NLD',
        'Uruguay': 'URY', 'Hungary': 'HUN', 'Sweden': 'SWE', 'Czech Republic': 'CZE',
        'Poland': 'POL', 'Russia': 'RUS', 'Portugal': 'PRT', 'Belgium': 'BEL',
        'Croatia': 'HRV', 'Denmark': 'DNK', 'Switzerland': 'CHE', 'Austria': 'AUT',
        'Mexico': 'MEX', 'USA': 'USA', 'Chile': 'CHL', 'Romania': 'ROU',
        'Bulgaria': 'BGR', 'Soviet Union': 'RUS', 'Yugoslavia': 'SRB',
        'West Germany': 'DEU', 'South Korea': 'KOR', 'Japan': 'JPN',
        'Australia': 'AUS', 'Turkey': 'TUR', 'Senegal': 'SEN', 'Morocco': 'MAR',
        'South Africa': 'ZAF', 'Qatar': 'QAT'
    }

    host_counts = wc_matches.groupby('country').size().reset_index(name='matches_hosted')
    host_counts['iso3'] = host_counts['country'].map(COUNTRY_ISO3)
    host_counts = host_counts.dropna(subset=['iso3'])

    fig = px.choropleth(
    host_counts,
    locations='iso3',
    locationmode='ISO-3',
    color='matches_hosted',
    hover_name='country',        # ← muestra el nombre del país
    hover_data={'iso3': False},  # ← oculta la columna iso3
    color_continuous_scale=[[0, '#1a2744'], [0.5, '#e8c040'], [1, '#ff6b6b']],
    labels={'matches_hosted': 'Matches Hosted'}
)
    fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#e0e0e0', height=500,
                geo=dict(
                    bgcolor='rgba(0,0,0,0)',
                    landcolor='#16213e',
                    showframe=False,
                    showcoastlines=False,
                    projection_type='natural earth'
                )
            )
    st.plotly_chart(fig, width='stretch')


# ══════════════════════════════════════════════
# PAGE 2: ML PREDICTOR
# ══════════════════════════════════════════════
elif menu == "🔮 Predictor":
    st.markdown("# 🔮 Match Predictor")
    st.markdown("*XGBoost prediction using 13 contextual features*")

    if not data_ok:
        st.error("Data not available.")
        st.stop()

    col_l, col_r = st.columns(2)
    with col_l:
        home_team  = st.selectbox("🏠 Home Team", all_teams,
                                  index=all_teams.index("Brazil") if "Brazil" in all_teams else 0)
        is_neutral = st.checkbox("⚖️ Neutral venue")
    with col_r:
        away_team = st.selectbox("✈️ Away Team", all_teams,
                                 index=all_teams.index("Argentina") if "Argentina" in all_teams else 1)

    st.markdown("---")

    if st.button("⚡ PREDICT RESULT", type="primary"):
        if home_team == away_team:
            st.error("Teams must be different.")
        elif not model_ok:
            st.warning("⚠️ Model not found. Run the notebook first to train and export it.")
        else:
            with st.spinner("Calculating..."):
                features   = build_features(home_team, away_team, results, rankings, is_neutral)
                pred       = model.predict(features)[0]
                proba      = model.predict_proba(features)[0]
                proba_dict = {'home_win': proba[2], 'draw': proba[1], 'away_win': proba[0]}

            if pred == 2:
                result_text = f"🏆 {home_team.upper()} WINS"
                conf = proba_dict['home_win']
            elif pred == 0:
                result_text = f"🏆 {away_team.upper()} WINS"
                conf = proba_dict['away_win']
            else:
                result_text = "⚖️ DRAW LIKELY"
                conf = proba_dict['draw']

            # Team cards
            c1, c2, c3 = st.columns([2, 1, 2])
            with c1:
                card_class = "pred-card pred-winner" if pred == 2 else "pred-card"
                st.markdown(f"""
                    <div class="{card_class}">
                        <div style='font-size:3rem'>🏠</div>
                        <div class="team-name">{home_team}</div>
                        <div style='color:#888; margin-top:8px'>Home</div>
                        <div style='color:#e8c040; font-size:1.8rem; font-family:Oswald; margin-top:12px'>
                            {proba_dict['home_win']:.0%}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                    <div style='text-align:center; padding-top:40px'>
                        <div class="vs-text">VS</div>
                        <div style='color:#888; font-size:0.9rem; margin-top:20px'>Draw</div>
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
                        <div class="team-name">{away_team}</div>
                        <div style='color:#888; margin-top:8px'>Away</div>
                        <div style='color:#e8c040; font-size:1.8rem; font-family:Oswald; margin-top:12px'>
                            {proba_dict['away_win']:.0%}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
                <div style='text-align:center; margin:28px 0'>
                    <span class='result-badge'>{result_text}</span>
                    <div style='color:#888; margin-top:10px; font-size:0.9rem'>
                        Model confidence: {conf:.1%}
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Probability chart
            fig = go.Figure(go.Bar(
                x=[f'🏠 {home_team}', '⚖️ Draw', f'✈️ {away_team}'],
                y=[proba_dict['home_win'], proba_dict['draw'], proba_dict['away_win']],
                marker_color=['#e8c040' if pred == 2 else '#334466',
                              '#e8c040' if pred == 1 else '#334466',
                              '#e8c040' if pred == 0 else '#334466'],
                text=[f"{v:.1%}" for v in [proba_dict['home_win'], proba_dict['draw'], proba_dict['away_win']]],
                textposition='outside', textfont_color='#e0e0e0'
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#e0e0e0',
                yaxis=dict(tickformat='.0%', gridcolor='rgba(255,255,255,0.06)', range=[0, 1]),
                showlegend=False, height=300, title="Model Probabilities"
            )
            st.plotly_chart(fig, width='stretch')

            # Quick H2H context
            h2h_df = get_h2h_stats(home_team, away_team, results)
            if h2h_df is not None and len(h2h_df) > 0:
                st.markdown("### 📋 Historical Context")
                w1 = sum(h2h_df['winner'] == home_team)
                w2 = sum(h2h_df['winner'] == away_team)
                dr = sum(h2h_df['winner'] == 'Draw')
                st.info(f"**H2H Record:** {home_team} {w1} — {dr} draws — {w2} {away_team} ({len(h2h_df)} total matches)")


# ══════════════════════════════════════════════
# PAGE 3: WC 2026 GAME
# ══════════════════════════════════════════════

elif menu == "🏆 WC26 Predictor":
    render_wc2026_game(
        model=model if model_ok else None,
        results_df=results if data_ok else None,
        rankings_df=rankings if data_ok else None,
        build_features_fn=build_features if data_ok else None
    )

# ══════════════════════════════════════════════
# PAGE 4: AI ANALYST
# ══════════════════════════════════════════════
elif menu == "🤖 AI Analyst":
    st.markdown("# 🤖 AI Sports Analyst")
    st.markdown("*Analysis generated by LLM (via OpenRouter) grounded in real historical data*")

    if not data_ok:
        st.error("Data not available.")
        st.stop()

    col_l, col_r = st.columns(2)
    with col_l:
        home_ai = st.selectbox("Home Team", all_teams,
                               index=all_teams.index("Germany") if "Germany" in all_teams else 0)
    with col_r:
        away_ai = st.selectbox("Away Team", all_teams,
                               index=all_teams.index("France") if "France" in all_teams else 1)

    # Language selector kept intentionally — useful bilingual feature
    language = st.radio("Analysis language", ["English", "Español"], horizontal=True)
    depth    = st.select_slider("Analysis depth", ["Brief", "Standard", "Detailed"], value="Standard")

    if st.button("🎙️ GENERATE ANALYSIS", type="primary"):
        if home_ai == away_ai:
            st.error("Teams must be different.")
        elif not llm_client:
            st.error("Configure your OpenRouter API Key in the sidebar to use this feature.")
        else:
            with st.spinner("AI analyst is preparing the report..."):
                today     = pd.Timestamp.now()
                h2h_df    = get_h2h_stats(home_ai, away_ai, results)
                home_form = get_form(home_ai, today, results)
                away_form = get_form(away_ai, today, results)
                home_rank = get_ranking_at_date(home_ai, today, rankings)
                away_rank = get_ranking_at_date(away_ai, today, rankings)

                h2h_summary = "No H2H data available"
                if h2h_df is not None and len(h2h_df) > 0:
                    w1 = sum(h2h_df['winner'] == home_ai)
                    w2 = sum(h2h_df['winner'] == away_ai)
                    dr = sum(h2h_df['winner'] == 'Draw')
                    h2h_summary = f"{home_ai}: {w1} wins | Draws: {dr} | {away_ai}: {w2} wins ({len(h2h_df)} matches)"

                tokens_map  = {"Brief": 200, "Standard": 400, "Detailed": 700}
                lang_instr  = "in Spanish" if language == "Español" else "in English"
                depth_instr = "2 concise paragraphs" if depth == "Brief" else "3 paragraphs" if depth == "Standard" else "4-5 detailed paragraphs"

                prompt = f"""You are an expert international football analyst with access to real statistical data.

MATCH: {home_ai} (home) vs {away_ai} (away)

STATISTICS:
- FIFA Ranking: {home_ai} #{home_rank} | {away_ai} #{away_rank}
- Recent form (win % last 10 matches): {home_ai} {home_form:.0%} | {away_ai} {away_form:.0%}
- Full H2H record: {h2h_summary}

INSTRUCTIONS:
- Write a professional sports analysis {lang_instr}
- Base your analysis ONLY on the data provided above
- Include: current form context, H2H psychological factor, and a conclusion
- Length: {depth_instr}
- Tone: journalistic, professional, passionate"""

                analysis = call_llm(
                    llm_client,
                    [{"role": "user", "content": prompt}],
                    max_tokens=tokens_map[depth],
                    temperature=0.7
                )

            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #1a1a2e, #16213e);
                            border: 1px solid #e8c04030; border-radius: 16px;
                            padding: 28px; margin-top: 20px;'>
                    <div style='color:#e8c040; font-family:Oswald; font-size:1rem;
                                letter-spacing:2px; margin-bottom:16px;'>
                        🎙️ ANALYSIS — {home_ai.upper()} vs {away_ai.upper()}
                    </div>
                    <div style='color:#e0e0e0; line-height:1.8; white-space: pre-wrap;'>{analysis}</div>
                </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE 4: CHAT WITH DATA
# ══════════════════════════════════════════════
elif menu == "💬 Chat with Data":
    st.markdown("# 💬 Chat with Historical Data")
    st.markdown("*Ask anything about World Cup history — the LLM queries real data to answer*")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat history
    for msg in st.session_state.messages:
        css_class = "chat-msg-user" if msg["role"] == "user" else "chat-msg-ai"
        icon      = "👤" if msg["role"] == "user" else "🤖"
        st.markdown(f'<div class="{css_class}">{icon} {msg["content"]}</div>', unsafe_allow_html=True)

    question = st.chat_input("e.g. How many times did Brazil win the World Cup? Best record against Germany?")

    if question:
        if not data_ok:
            st.error("Data not available.")
        elif not llm_client:
            st.error("Configure your OpenRouter API Key in the sidebar.")
        else:
            st.session_state.messages.append({"role": "user", "content": question})
            st.markdown(f'<div class="chat-msg-user">👤 {question}</div>', unsafe_allow_html=True)

            with st.spinner("Querying data..."):
                wc = results[results['tournament'] == 'FIFA World Cup'].copy()
                wins_home  = wc[wc['home_score'] > wc['away_score']].groupby('home_team').size()
                wins_away  = wc[wc['away_score'] > wc['home_score']].groupby('away_team').size()
                total_wins = wins_home.add(wins_away, fill_value=0).sort_values(ascending=False).head(10)

                context = f"""FIFA WORLD CUP HISTORICAL DATA:
- Total matches: {len(wc):,} | Period: {wc['date'].min().year}–{wc['date'].max().year}
- Total international matches in dataset: {len(results):,}
- National teams: {len(all_teams):,}

TOP 10 NATIONS BY WORLD CUP WINS:
{total_wins.to_string()}

RESULT DISTRIBUTION (World Cup):
Home win: {(wc['result_label']=='Home Win').mean():.1%} | Draw: {(wc['result_label']=='Draw').mean():.1%} | Away win: {(wc['result_label']=='Away Win').mean():.1%}"""

                # Add team-specific stats if a team is mentioned
                for team in all_teams:
                    if team.lower() in question.lower():
                        tm = wc[(wc['home_team'] == team) | (wc['away_team'] == team)]
                        if len(tm) > 0:
                            tw = len(tm[
                                ((tm['home_team'] == team) & (tm['home_score'] > tm['away_score'])) |
                                ((tm['away_team'] == team) & (tm['away_score'] > tm['home_score']))
                            ])
                            goals = int(
                                tm[tm['home_team'] == team]['home_score'].sum() +
                                tm[tm['away_team'] == team]['away_score'].sum()
                            )
                            context += f"\n\n{team.upper()} IN WORLD CUPS:\n"
                            context += f"- Matches: {len(tm)} | Wins: {tw} | Goals scored: {goals}\n"

                messages_for_api = [
                    {
                        "role": "system",
                        "content": f"""You are a FIFA World Cup statistics expert. You MUST always give a complete, informative answer of at least 2-3 sentences. Never respond with just one word or a filler phrase. Answer in the same language as the user. Base your answers on the data below, and if the exact data isn't available, say so clearly and provide what context you can.{context}"""
                    }
                ] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-6:]]

                answer = call_llm(llm_client, messages_for_api, max_tokens=700, temperature=0.5)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.markdown(f'<div class="chat-msg-ai">🤖 {answer}</div>', unsafe_allow_html=True)
            st.rerun()

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()
