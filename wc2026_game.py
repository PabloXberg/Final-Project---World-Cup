"""
🏆 FIFA World Cup 2026 Prediction Game
Add this page to app.py by importing and calling render_wc2026_game()
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime


def load_fixture():
    """Load and prepare the WC 2026 fixture."""
    df = pd.read_csv('db/fifa-world-cup-2026-UTC.csv')
    # Only group stage matches (real team names, not placeholders like "1A")
    groups = df[df['Round Number'].isin(['1', '2', '3'])].copy()
    groups = groups[~groups['Home Team'].str.match(r'^\d')]  # remove placeholder teams
    return df, groups


def load_players():
    """Load player stats for WC 2026 participating nations."""
    try:
        players = pd.read_csv('db/players_data_light-2025_2026.csv')
        players['nation_code'] = players['Nation'].str.split().str[-1]
        return players
    except Exception:
        return None


# Nation code mapping: WC fixture name → player dataset code
WC_TO_CODE = {
    'Argentina': 'ARG', 'Australia': 'AUS', 'Austria': 'AUT', 'Algeria': 'ALG',
    'Belgium': 'BEL', 'Brazil': 'BRA', 'Canada': 'CAN', 'Colombia': 'COL',
    'Congo DR': 'COD', 'Croatia': 'CRO', 'Czechia': 'CZE', 'Ecuador': 'ECU',
    'Egypt': 'EGY', 'England': 'ENG', 'France': 'FRA', 'Germany': 'GER',
    'Ghana': 'GHA', 'Haiti': 'HAI', 'IR Iran': 'IRN', 'Iraq': 'IRQ',
    'Japan': 'JPN', 'Jordan': 'JOR', 'Korea Republic': 'KOR', 'Mexico': 'MEX',
    'Morocco': 'MAR', 'Netherlands': 'NED', 'New Zealand': 'NZL', 'Norway': 'NOR',
    'Panama': 'PAN', 'Paraguay': 'PAR', 'Portugal': 'POR', 'Qatar': 'QAT',
    'Saudi Arabia': 'KSA', 'Scotland': 'SCO', 'Senegal': 'SEN',
    'South Africa': 'RSA', 'Spain': 'ESP', 'Sweden': 'SWE',
    'Switzerland': 'SUI', 'Tunisia': 'TUN', 'Türkiye': 'TUR', 'USA': 'USA',
    'Uruguay': 'URU', 'Uzbekistan': 'UZB', "Côte d'Ivoire": 'CIV',
    'Bosnia-Herzegovina': 'BIH', 'Cabo Verde': 'CPV', 'Curaçao': 'CUW',
}


def get_team_stars(team_name, players_df):
    """Get top 3 players for a team by goals+assists."""
    if players_df is None:
        return []
    code = WC_TO_CODE.get(team_name)
    if not code:
        return []
    team_players = players_df[players_df['nation_code'] == code].copy()
    if len(team_players) == 0:
        return []
    team_players['score'] = team_players['Gls'].fillna(0) + team_players['Ast'].fillna(0)
    top = team_players.nlargest(3, 'score')[['Player', 'Pos', 'Squad', 'Gls', 'Ast', 'Min']].reset_index(drop=True)
    return top.to_dict('records')


def render_wc2026_game(model=None, results_df=None, rankings_df=None, build_features_fn=None):
    """Render the World Cup 2026 Prediction Game page."""

    st.markdown("# 🏆 World Cup 2026 — Prediction Game")
    st.markdown("*Predict every group stage match and download your bracket!*")

    # Load data
    try:
        full_fixture, group_matches = load_fixture()
    except Exception as e:
        st.error(f"Could not load fixture. Make sure `db/fifa-world-cup-2026-UTC.csv` exists. Error: {e}")
        return

    players_df = load_players()

    # ── Tournament Info ────────────────────────────
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Matches", 104)
    col2.metric("Teams", 48)
    col3.metric("Groups", 12)
    col4.metric("Host Countries", "🇺🇸🇨🇦🇲🇽")

    st.markdown("---")

    # ── Mode Selection ─────────────────────────────
    mode = st.radio(
        "Choose your mode:",
        ["🎮 Make My Predictions", "🤖 AI Predictions (use ML model)", "📊 Squad Explorer"],
        horizontal=True
    )

    # ══════════════════════════════════════════════
    # MODE 1: USER PREDICTIONS
    # ══════════════════════════════════════════════
    if mode == "🎮 Make My Predictions":
        st.markdown("### Select a winner for each group stage match")
        st.markdown("*Choose who you think will win — or select Draw*")

        # Get unique groups
        groups_list = sorted(group_matches['Group'].unique())

        # Initialize predictions in session state
        if 'wc_predictions' not in st.session_state:
            st.session_state.wc_predictions = {}

        # Create tabs for each group
        tabs = st.tabs([f"🏷️ {g}" for g in groups_list])

        for tab, group_name in zip(tabs, groups_list):
            with tab:
                matches = group_matches[group_matches['Group'] == group_name].sort_values('Match Number')

                for _, match in matches.iterrows():
                    match_id = int(match['Match Number'])
                    home = match['Home Team']
                    away = match['Away Team']
                    location = match['Location']
                    date_str = match['Date']

                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #1a1a2e, #16213e);
                                    border: 1px solid #e8c04020; border-radius: 12px;
                                    padding: 16px 20px; margin: 8px 0;'>
                            <div style='color:#888; font-size:0.8rem; margin-bottom:8px;'>
                                Match {match_id} · {date_str} · {location}
                            </div>
                            <div style='font-family:Oswald,sans-serif; font-size:1.3rem; color:#fff;'>
                                {home}  <span style='color:#e8c040;'>vs</span>  {away}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    # Show star players if available
                    home_stars = get_team_stars(home, players_df)
                    away_stars = get_team_stars(away, players_df)
                    if home_stars or away_stars:
                        c1, c2 = st.columns(2)
                        with c1:
                            if home_stars:
                                stars_text = ", ".join([f"**{p['Player']}** ({p['Gls']:.0f}G {p['Ast']:.0f}A)" for p in home_stars[:2]])
                                st.caption(f"⭐ {home}: {stars_text}")
                        with c2:
                            if away_stars:
                                stars_text = ", ".join([f"**{p['Player']}** ({p['Gls']:.0f}G {p['Ast']:.0f}A)" for p in away_stars[:2]])
                                st.caption(f"⭐ {away}: {stars_text}")

                    prediction = st.radio(
                        f"Your prediction for Match {match_id}:",
                        [f"🏠 {home}", "⚖️ Draw", f"✈️ {away}"],
                        key=f"pred_{match_id}",
                        horizontal=True,
                        label_visibility="collapsed"
                    )
                    st.session_state.wc_predictions[match_id] = {
                        'home': home,
                        'away': away,
                        'prediction': prediction,
                        'group': group_name
                    }
                    st.markdown("")  # spacing

        # ── Summary & Export ───────────────────────
        st.markdown("---")
        st.markdown("### 📋 Your Predictions Summary")

        total_matches = len(group_matches)
        predicted = len(st.session_state.wc_predictions)
        st.progress(predicted / total_matches if total_matches > 0 else 0)
        st.caption(f"{predicted} / {total_matches} matches predicted")

        if predicted > 0:
            # Build summary
            summary_data = []
            for mid, pred in sorted(st.session_state.wc_predictions.items()):
                winner = pred['prediction'].split(' ', 1)[-1] if '⚖️' not in pred['prediction'] else 'Draw'
                summary_data.append({
                    'Match': mid,
                    'Group': pred['group'],
                    'Home': pred['home'],
                    'Away': pred['away'],
                    'Your Pick': winner
                })

            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, width='stretch', hide_index=True)

            # Export options
            st.markdown("### 💾 Save Your Predictions")
            col_a, col_b = st.columns(2)

            with col_a:
                # JSON download
                export_data = {
                    'user': st.text_input("Your name (for the bracket):", value="Football Fan"),
                    'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'tournament': 'FIFA World Cup 2026',
                    'predictions': summary_data
                }
                json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
                st.download_button(
                    "⬇️ Download as JSON",
                    data=json_str,
                    file_name=f"wc2026_predictions_{export_data['user'].replace(' ', '_')}.json",
                    mime="application/json"
                )

            with col_b:
                # CSV download
                csv_str = df_summary.to_csv(index=False)
                st.download_button(
                    "⬇️ Download as CSV",
                    data=csv_str,
                    file_name=f"wc2026_predictions.csv",
                    mime="text/csv"
                )

    # ══════════════════════════════════════════════
    # MODE 2: AI MODEL PREDICTIONS
    # ══════════════════════════════════════════════
    elif mode == "🤖 AI Predictions (use ML model)":
        st.markdown("### ML Model predicts every group stage match")

        if model is None or results_df is None or rankings_df is None or build_features_fn is None:
            st.warning("⚠️ ML model not available. Make sure the notebook was run and `modelo_fifa.pkl` exists.")
            return

        if st.button("🚀 RUN ALL PREDICTIONS", type="primary"):
            with st.spinner("Running predictions for all 48 group matches..."):
                ai_predictions = []

                for _, match in group_matches.iterrows():
                    home = match['Home Team']
                    away = match['Away Team']

                    try:
                        features = build_features_fn(home, away, results_df, rankings_df)
                        pred  = model.predict(features)[0]
                        proba = model.predict_proba(features)[0]

                        if pred == 2:
                            winner = home
                            conf = proba[2]
                        elif pred == 0:
                            winner = away
                            conf = proba[0]
                        else:
                            winner = 'Draw'
                            conf = proba[1]

                        ai_predictions.append({
                            'Match': int(match['Match Number']),
                            'Group': match['Group'],
                            'Home': home,
                            'Away': away,
                            'Prediction': winner,
                            'Confidence': f"{conf:.0%}",
                            'Home %': f"{proba[2]:.0%}",
                            'Draw %': f"{proba[1]:.0%}",
                            'Away %': f"{proba[0]:.0%}",
                        })
                    except Exception as e:
                        ai_predictions.append({
                            'Match': int(match['Match Number']),
                            'Group': match['Group'],
                            'Home': home,
                            'Away': away,
                            'Prediction': f'Error: {str(e)[:30]}',
                            'Confidence': '-',
                            'Home %': '-',
                            'Draw %': '-',
                            'Away %': '-',
                        })

            df_ai = pd.DataFrame(ai_predictions)
            st.session_state.ai_predictions = df_ai

        # Show results if available
        if 'ai_predictions' in st.session_state:
            df_ai = st.session_state.ai_predictions

            # Stats
            c1, c2, c3 = st.columns(3)
            valid = df_ai[~df_ai['Prediction'].str.startswith('Error')]
            home_wins = (valid['Prediction'] == valid['Home']).sum()
            away_wins = (valid['Prediction'] == valid['Away']).sum()
            draws = (valid['Prediction'] == 'Draw').sum()
            c1.metric("🏠 Home Wins", home_wins)
            c2.metric("⚖️ Draws", draws)
            c3.metric("✈️ Away Wins", away_wins)

            # Group-by-group display
            for group in sorted(df_ai['Group'].unique()):
                st.markdown(f"#### {group}")
                group_df = df_ai[df_ai['Group'] == group][
                    ['Match', 'Home', 'Away', 'Prediction', 'Confidence', 'Home %', 'Draw %', 'Away %']
                ]
                st.dataframe(group_df, width='stretch', hide_index=True)

            # Download
            csv_ai = df_ai.to_csv(index=False)
            st.download_button(
                "⬇️ Download AI Predictions (CSV)",
                data=csv_ai,
                file_name="wc2026_ai_predictions.csv",
                mime="text/csv"
            )

    # ══════════════════════════════════════════════
    # MODE 3: SQUAD EXPLORER
    # ══════════════════════════════════════════════
    elif mode == "📊 Squad Explorer":
        st.markdown("### Explore top players from each World Cup team")
        st.markdown("*Stats from the top 5 European leagues (2025-26 season)*")

        if players_df is None:
            st.warning("Player data not found. Make sure `db/players_data_light-2025_2026.csv` exists.")
            return

        # Team selector (only WC teams)
        wc_team_names = sorted([t for t in WC_TO_CODE.keys()])
        selected_team = st.selectbox("Select a national team:", wc_team_names)

        code = WC_TO_CODE.get(selected_team)
        if code:
            team_players = players_df[players_df['nation_code'] == code].copy()

            if len(team_players) == 0:
                st.info(f"No player data available for {selected_team} in the top 5 European leagues.")
            else:
                st.metric(f"Players from {selected_team} in top leagues", len(team_players))

                # Sort options
                sort_by = st.selectbox("Sort by:", ["Goals", "Assists", "G+A", "Minutes", "Matches"])
                sort_col = {'Goals': 'Gls', 'Assists': 'Ast', 'G+A': 'G+A', 'Minutes': 'Min', 'Matches': 'MP'}[sort_by]

                display = team_players.nlargest(20, sort_col)[
                    ['Player', 'Pos', 'Squad', 'Comp', 'Age', 'MP', 'Starts', 'Min', 'Gls', 'Ast', 'G+A', 'CrdY', 'CrdR']
                ].reset_index(drop=True)
                display.columns = ['Player', 'Position', 'Club', 'League', 'Age', 'Matches', 'Starts', 'Minutes', 'Goals', 'Assists', 'G+A', 'Yellows', 'Reds']

                st.dataframe(display, width='stretch', hide_index=True)

                # Quick stats
                st.markdown("#### 📊 Squad Summary")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Goals", f"{team_players['Gls'].sum():.0f}")
                c2.metric("Total Assists", f"{team_players['Ast'].sum():.0f}")
                c3.metric("Avg Age", f"{team_players['Age'].mean():.1f}")
                c4.metric("Total Minutes", f"{team_players['Min'].sum():,.0f}")
