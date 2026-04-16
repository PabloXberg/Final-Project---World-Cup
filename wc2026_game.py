"""
🏆 FIFA World Cup 2026 Predictor (v3)
Single mode: AI predicts groups → builds bracket → predicts entire knockout
Displays full bracket tree in one view with team logos.
"""
import streamlit as st
import pandas as pd
import os
import base64
from pathlib import Path

LOGOS_DIR = Path('assets/logos')

# ──────────────────────────────────────────────
# TEAM → LOGO FILENAME MAPPING
# Built from your screenshot of assets/logos/
# ──────────────────────────────────────────────
TEAM_LOGOS = {
    'Algeria':            'algeria-national-team-footballlogos-org.png',
    'Argentina':          'argentina-national-team-footballlogos-org.png',
    'Australia':          'australia-national-team-footylogos.png',
    'Austria':            'austria-national-team-footballlogos-org.png',
    'Belgium':            'belgium-national-team-footballlogos-org.png',
    'Bosnia-Herzegovina': 'bosnia-and-herzegovina-footballlogos-org.png',
    'Brazil':             'brazil-national-team-footballlogos-org.png',
    'Cabo Verde':         'cabo-verde-footballlogos-org.png',
    'Canada':             'canada-national-team-footballlogos-org.png',
    'Colombia':           'colombia-national-team-footballlogos-org.png',
    'Congo DR':           'dr-congo-footballlogos-org.png',
    'Croatia':            'croatia-national-team-footballlogos-org.png',
    'Curaçao':            'curacao-national-team-footballlogos-org.png',
    'Czechia':            'czechia-national-team-footballlogos-org.png',
    "Côte d'Ivoire":      'cote-d-ivoire-national-team-footballlogos-org.png',
    'Ecuador':            'ecuador-national-team-footballlogos-org.png',
    'Egypt':              'egypt-national-team-footballlogos-org.png',
    'England':            'england-national-team-footballlogos-org.png',
    'France':             'france-national-team-footballlogos-org.png',
    'Germany':            'germany-national-team-footballlogos-org.png',
    'Ghana':              'ghana-footballlogos-org.png',
    'Haiti':              'haiti-national-team-footylogos.png',
    'IR Iran':            'iran-national-team-footballlogos-org.png',
    'Iraq':               'iraq-footballlogos-org.png',
    'Japan':              'japan-national-team-footballlogos-org.png',
    'Jordan':             'jordan-footballlogos-org.png',
    'Korea Republic':     'south-korea-national-team-footballlogos-org.png',
    'Mexico':             'mexico-national-team-footballlogos-org.png',
    'Morocco':            'morocco-national-team-footballlogos-org.png',
    'Netherlands':        'netherlands-dutch-national-team-footballlogos-org.png',
    'New Zealand':        'new-zealand-national-team-footballlogos-org.png',
    'Norway':             'norway-national-team-footballlogos-org.png',
    'Panama':             'panama-national-team-footballlogos-org.png',
    'Paraguay':           'paraguay-national-team-footballlogos-org.png',
    'Portugal':           'portugal-national-team-footballlogos-org.png',
    'Qatar':              'qatar-national-team-footballlogos-org.png',
    'Saudi Arabia':       'saudi-arabia-national-team-footballlogos-org.png',
    'Scotland':           'scotland-national-team-footballlogos-org.png',
    'Senegal':            'senegal-national-team-footballlogos-org.png',
    'South Africa':       'south-africa-national-team-footballlogos-org.png',
    'Spain':              'spain-national-team-footballlogos-org.png',
    'Sweden':             'sweden-national-team-footballlogos-org.png',
    'Switzerland':        'swiss-national-team-footballlogos-org.png',
    'Tunisia':            'tunisia-national-team-footballlogos-org.png',
    'Türkiye':            'turkey-national-team-footballlogos-org.png',
    'USA':                'usa-national-team-footballlogos-org.png',
    'Uruguay':            'uruguay-national-team-footballlogos-org.png',
    'Uzbekistan':         'uzbekistan-national-team-footballlogos-org.png',
    'WC2026': 'wc2026_background.png'
}


WC_LOGO = 'WC26_Logo.avif'


@st.cache_data
def img_to_base64(path):
    """Convert image file to base64 data URI for inline embedding."""
    try:
        with open(path, 'rb') as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{data}"
    except Exception:
        return None


def logo_html(team_name, size=32):
    """Return HTML <img> tag for a team logo (inline, base64-encoded)."""
    filename = TEAM_LOGOS.get(team_name)
    if not filename:
        return f'<span style="color:#666; font-size:0.8rem;">[{team_name}]</span>'
    path = LOGOS_DIR / filename
    if not path.exists():
        return f'<span style="color:#666; font-size:0.8rem;">[{team_name}]</span>'
    data_uri = img_to_base64(str(path))
    if not data_uri:
        return ""
    return f'<img src="{data_uri}" style="height:{size}px; vertical-align:middle; margin-right:6px;" alt="{team_name}">'


def wc_logo_html(size=80):
    """WC 2026 logo at the top of the page."""
    path = LOGOS_DIR / WC_LOGO
    if not path.exists():
        return ""
    data_uri = img_to_base64(str(path))
    return f'<img src="{data_uri}" style="height:{size}px;" alt="WC 2026">'


# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────
@st.cache_data
def load_fixture():
    df = pd.read_csv('db/fifa-world-cup-2026-UTC.csv')
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M', errors='coerce')
    groups = df[df['Round Number'].isin(['1', '2', '3'])].copy()
    groups = groups[~groups['Home Team'].str.match(r'^\d')]
    return df, groups


# ──────────────────────────────────────────────
# BRACKET LOGIC
# ──────────────────────────────────────────────
def resolve_match(home_p, away_p, qualified):
    """Convert placeholder slots like '1A' / '3CDFGH' to real team names."""
    def resolve(p):
        if p in qualified:
            return qualified[p]
        if p.startswith('3') and len(p) > 1:
            for letter in p[1:]:
                if f'3{letter}' in qualified:
                    return qualified[f'3{letter}']
        return p
    return resolve(home_p), resolve(away_p)


# ──────────────────────────────────────────────
# BRACKET RENDERING
# ──────────────────────────────────────────────
def match_box(home, away, winner, color='#e8c040'):
    """Render a single match box for the bracket tree."""
    home_color = '#e8c040' if winner == home else '#888'
    away_color = '#e8c040' if winner == away else '#888'
    home_weight = '700' if winner == home else '400'
    away_weight = '700' if winner == away else '400'

    return f"""
        <div style='background:#1a1a2e; border:1px solid {color}30; border-left:3px solid {color};
                    border-radius:4px; padding:6px 10px; margin:6px 0; min-width:180px;'>
            <div style='display:flex; align-items:center; color:{home_color}; font-weight:{home_weight}; font-size:0.85rem; padding:2px 0;'>
                {logo_html(home, 22)}{home}
            </div>
            <div style='display:flex; align-items:center; color:{away_color}; font-weight:{away_weight}; font-size:0.85rem; padding:2px 0;'>
                {logo_html(away, 22)}{away}
            </div>
        </div>
    """


# ──────────────────────────────────────────────
# MAIN PAGE
# ──────────────────────────────────────────────
def render_wc2026_game(model=None, results_df=None, rankings_df=None, build_features_fn=None):
    """Render the World Cup 2026 Predictor page (AI-only mode)."""

    # Header with WC 2026 logo
    st.markdown(f"""
        <div style='display:flex; align-items:center; gap:24px; margin-bottom:16px;'>
            {wc_logo_html(100)}
            <div>
                <div style='font-family:Oswald,sans-serif; font-size:2.2rem; color:#fff;
                            letter-spacing:2px; line-height:1;'>
                    WC 2026 PREDICTOR
                </div>
                <div style='color:#e8c040; font-family:Oswald,sans-serif; letter-spacing:3px;
                            font-size:0.9rem; margin-top:6px;'>
                    🇺🇸 USA · 🇨🇦 CANADA · 🇲🇽 MEXICO
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Load data
    try:
        full_fixture, group_matches = load_fixture()
    except Exception as e:
        st.error(f"Could not load fixture. Error: {e}")
        return

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Matches", "104")
    col2.metric("Teams", "48")
    col3.metric("Groups", "12")
    col4.metric("Stadiums", "16")

    st.markdown("---")
    st.markdown("### 🤖 ML model simulates the entire tournament")
    st.caption("Click below — the model predicts every group match, calculates standings, "
               "and runs the full knockout bracket all the way to the champion.")

    if model is None or results_df is None or rankings_df is None or build_features_fn is None:
        st.warning("⚠️ ML model not available. Run the notebook first to train and export `modelo_fifa.pkl`.")
        return

    if st.button("🚀 SIMULATE FULL TOURNAMENT", type="primary"):
        with st.spinner("Simulating 104 matches..."):

            # ── STEP 1: Group stage ──
            group_results = {g: {} for g in sorted(group_matches['Group'].unique())}
            for _, m in group_matches.iterrows():
                for t in [m['Home Team'], m['Away Team']]:
                    group_results[m['Group']].setdefault(t, 0)

            for _, m in group_matches.iterrows():
                home, away = m['Home Team'], m['Away Team']
                try:
                    feats = build_features_fn(home, away, results_df, rankings_df)
                    pred  = model.predict(feats)[0]
                    if pred == 2:
                        group_results[m['Group']][home] += 3
                    elif pred == 0:
                        group_results[m['Group']][away] += 3
                    else:
                        group_results[m['Group']][home] += 1
                        group_results[m['Group']][away] += 1
                except Exception:
                    pass

            # ── STEP 2: Final standings + qualified teams ──
            ai_qualified = {}
            final_standings = {}
            for g, points in group_results.items():
                ranked = sorted(points.items(), key=lambda x: x[1], reverse=True)
                final_standings[g] = ranked
                letter = g.replace('Group ', '')
                if len(ranked) >= 1: ai_qualified[f'1{letter}'] = ranked[0][0]
                if len(ranked) >= 2: ai_qualified[f'2{letter}'] = ranked[1][0]
                if len(ranked) >= 3: ai_qualified[f'3{letter}'] = ranked[2][0]

            # ── STEP 3: Knockouts ──
            def predict_match(h, a):
                if h not in TEAM_LOGOS or a not in TEAM_LOGOS:
                    return h
                try:
                    f = build_features_fn(h, a, results_df, rankings_df)
                    p = model.predict(f)[0]
                    if p == 0: return a
                    return h
                except Exception:
                    return h

            # Round of 32
            ro32 = full_fixture[full_fixture['Round Number'] == 'Round of 32'].sort_values('Match Number')
            ro32_results = []
            ro16_teams = []
            for _, m in ro32.iterrows():
                h, a = resolve_match(m['Home Team'], m['Away Team'], ai_qualified)
                if h in TEAM_LOGOS and a in TEAM_LOGOS:
                    w = predict_match(h, a)
                    ro32_results.append((h, a, w))
                    ro16_teams.append(w)

            def play_round(teams):
                pairs = []
                winners = []
                for i in range(0, len(teams), 2):
                    if i + 1 < len(teams):
                        h, a = teams[i], teams[i+1]
                        w = predict_match(h, a)
                        pairs.append((h, a, w))
                        winners.append(w)
                return pairs, winners

            ro16_pairs, qf_teams      = play_round(ro16_teams)
            qf_pairs,   sf_teams      = play_round(qf_teams)
            sf_pairs,   final_teams   = play_round(sf_teams)
            final_pair, champion_list = play_round(final_teams)
            champion = champion_list[0] if champion_list else None

            st.session_state.wc_sim = {
                'standings': final_standings,
                'ro32': ro32_results,
                'ro16': ro16_pairs,
                'qf': qf_pairs,
                'sf': sf_pairs,
                'final': final_pair,
                'champion': champion
            }

    # ──────────────────────────────────────────────
    # DISPLAY RESULTS
    # ──────────────────────────────────────────────
    if 'wc_sim' not in st.session_state:
        return

    sim = st.session_state.wc_sim

    # Champion banner
    if sim['champion']:
        st.markdown(f"""
            <div style='background:linear-gradient(90deg, #e8c040, #f0a500);
                        padding:32px; border-radius:16px; text-align:center;
                        margin:24px 0; box-shadow:0 8px 32px #e8c04040;'>
                <div style='font-family:Oswald,sans-serif; font-size:1rem;
                            color:#0a0a0f; letter-spacing:4px;'>
                    🏆 PREDICTED WORLD CHAMPION 🏆
                </div>
                <div style='display:flex; align-items:center; justify-content:center;
                            gap:20px; margin-top:16px;'>
                    {logo_html(sim['champion'], 80)}
                    <div style='font-family:Oswald,sans-serif; font-size:3rem;
                                color:#0a0a0f; letter-spacing:4px;'>
                        {sim['champion'].upper()}
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

  # ── Group standings ──
    st.markdown("### 📊 Final Group Standings")
    cols = st.columns(3)
    for idx, (g, ranked) in enumerate(sorted(sim['standings'].items())):
        with cols[idx % 3]:
            medals = ['🥇', '🥈', '🥉', '4️⃣']
            # Group header
            st.markdown(f"""
                <div style='background:#1a1a2e; border:1px solid #e8c04030;
                            border-radius:8px; padding:10px 14px; margin-bottom:4px;'>
                    <span style='color:#e8c040; font-family:Oswald,sans-serif;
                                 letter-spacing:2px; font-size:1rem;'>{g}</span>
                </div>
            """, unsafe_allow_html=True)
            # Each team as its own markdown block (avoids nested tables)
            for i, (team, pts) in enumerate(ranked[:4]):
                medal = medals[i] if i < 4 else ''
                color = '#e8c040' if i < 2 else '#888'  # top 2 advance = gold
                st.markdown(f"""
                    <div style='background:#16213e; border-left:3px solid {"#e8c040" if i < 2 else "#333"};
                                padding:7px 12px; margin:2px 0; border-radius:0 4px 4px 0;
                                display:flex; align-items:center; justify-content:space-between;'>
                        <span style='display:flex; align-items:center; gap:8px;'>
                            <span>{medal}</span>
                            {logo_html(team, 24)}
                            <span style='color:{color}; font-size:0.9rem;'>{team}</span>
                        </span>
                        <span style='color:#e8c040; font-family:Oswald,sans-serif;
                                     font-size:1rem;'>{pts} pts</span>
                    </div>
                """, unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)

    # ── Knockout bracket — full tree view ──
    st.markdown("---")
    st.markdown("### 🌳 Knockout Bracket")
    st.caption("Full tournament tree from Round of 32 to the Final")

    # Display all rounds in 5 columns side by side (the tree view you wanted)
    cols = st.columns(5)
    rounds = [
        ('ROUND OF 32',     sim['ro32'],  '#e8c040'),
        ('ROUND OF 16',     sim['ro16'],  '#4a9eff'),
        ('QUARTER FINALS',  sim['qf'],    '#ff6b6b'),
        ('SEMI FINALS',     sim['sf'],    '#b59cff'),
        ('FINAL',           sim['final'], '#f0a500'),
    ]

    for col, (title, matches, color) in zip(cols, rounds):
        with col:
            st.markdown(f"""
                <div style='text-align:center; color:{color}; font-family:Oswald,sans-serif;
                            letter-spacing:2px; font-size:0.85rem; margin-bottom:8px;
                            border-bottom:2px solid {color}40; padding-bottom:6px;'>
                    {title}
                </div>
            """, unsafe_allow_html=True)
            for h, a, w in matches:
                st.markdown(match_box(h, a, w, color), unsafe_allow_html=True)

    # ── Champion summary ──
    if sim['champion']:
        st.markdown("---")
        st.markdown("### 🏅 Tournament Summary")

        runner_up = None
        if sim['final']:
            h, a, w = sim['final'][0]
            runner_up = a if w == h else h

        semifinalists = []
        for h, a, w in sim['sf']:
            loser = a if w == h else h
            semifinalists.append(loser)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f"""
                <div style='background:linear-gradient(135deg,#e8c040,#f0a500); padding:20px;
                            border-radius:12px; text-align:center;'>
                    <div style='font-size:2.5rem;'>🥇</div>
                    {logo_html(sim['champion'], 50)}
                    <div style='color:#0a0a0f; font-family:Oswald,sans-serif;
                                font-size:1.3rem; letter-spacing:1px; margin-top:8px;'>
                        {sim['champion']}
                    </div>
                    <div style='color:#0a0a0f80; font-size:0.8rem; margin-top:4px;'>CHAMPION</div>
                </div>
            """, unsafe_allow_html=True)
        with col_b:
            if runner_up:
                st.markdown(f"""
                    <div style='background:linear-gradient(135deg,#bbb,#888); padding:20px;
                                border-radius:12px; text-align:center;'>
                        <div style='font-size:2.5rem;'>🥈</div>
                        {logo_html(runner_up, 50)}
                        <div style='color:#0a0a0f; font-family:Oswald,sans-serif;
                                    font-size:1.3rem; letter-spacing:1px; margin-top:8px;'>
                            {runner_up}
                        </div>
                        <div style='color:#0a0a0f80; font-size:0.8rem; margin-top:4px;'>RUNNER-UP</div>
                    </div>
                """, unsafe_allow_html=True)
        with col_c:
            sf_text = " · ".join(semifinalists) if semifinalists else "—"
            sf_logos = "".join([logo_html(t, 32) for t in semifinalists])
            st.markdown(f"""
                <div style='background:linear-gradient(135deg,#cd7f32,#8b4513); padding:20px;
                            border-radius:12px; text-align:center;'>
                    <div style='font-size:2.5rem;'>🥉</div>
                    <div style='margin-top:8px;'>{sf_logos}</div>
                    <div style='color:#fff; font-family:Oswald,sans-serif;
                                font-size:0.95rem; letter-spacing:1px; margin-top:8px;'>
                        {sf_text}
                    </div>
                    <div style='color:#ffffff80; font-size:0.8rem; margin-top:4px;'>SEMIFINALISTS</div>
                </div>
            """, unsafe_allow_html=True)
