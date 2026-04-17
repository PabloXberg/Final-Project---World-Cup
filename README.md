# ⚽ FIFA World Cup AI Predictor

### End-to-End Data Science Project

A comprehensive Data Science project that predicts international football match outcomes using Machine Learning, Deep Learning, and Generative AI — deployed as an interactive Streamlit application with a full FIFA World Cup 2026 tournament simulator.

---

## 🎯 Project Goal

Build an end-to-end system that:
1. Analyzes 150+ years of international football history (1872–2024)
2. Engineers 13 contextual features per match from raw data
3. Trains and compares 4 ML/DL models to predict match outcomes
4. Integrates a real LLM (via OpenRouter) for data-grounded sports analysis
5. Simulates the entire FIFA World Cup 2026 bracket (48 teams, 104 matches)
6. Deploys everything through a professional Streamlit application

---

## 🗂️ Project Structure

```
FIFA-World-Cup-Predictor/
│
├── db/
│   ├── results.csv                      # International matches 1872–2024
│   ├── ranking.csv                      # FIFA world rankings 1992–2024
│   ├── fifa-world-cup-2026-UTC.csv      # WC 2026 fixture (104 matches)
│   └── features_engineered.csv          # Pre-computed feature matrix
│
├── assets/logos/                         # National team crests (48 teams + WC logo)
│
├── models/
│   ├── xgb_model.pkl                    # XGBoost (primary model)
│   ├── rf_model.pkl                     # Random Forest
│   ├── gb_model.pkl                     # Gradient Boosting
│   ├── scaler.pkl                       # StandardScaler for Neural Network
│   └── neural_network.keras             # MLP Deep Learning model
│
├── plots/                               # EDA and ML visualizations
│
├── main.ipynb                           # Complete notebook (EDA + ML + GenAI)
├── app.py                               # Streamlit application (main)
├── wc2026_game.py                       # WC 2026 predictor module
├── modelo_fifa.pkl                      # Primary model (used by app.py)
├── requirements.txt                     # Python dependencies
├── .env                                 # API keys (DO NOT push to GitHub)
├── .gitignore
└── README.md
```

---

## 📦 Datasets

Download from Kaggle and place in the `db/` folder:

| File | Source | Records |
|---|---|---|
| `results.csv` | [martj42/international-football-results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2022) | ~49,000 matches |
| `ranking.csv` | [cashncarry/fifaworldranking](https://www.kaggle.com/datasets/cashncarry/fifaworldranking) | ~67,000 rankings |
| `fifa-world-cup-2026-UTC.csv` | [fixturedownload.com](https://fixturedownload.com/results/fifa-world-cup-2026) | 104 matches |

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USER/FIFA-World-Cup-Predictor.git
cd FIFA-World-Cup-Predictor

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API key (free at openrouter.ai)
echo "OPENROUTER_API_KEY=sk-or-your-key-here" > .env
```

---

## 🚀 Usage

```bash
# Run the notebook (training + analysis)
jupyter notebook main.ipynb

# Run the Streamlit app
python -m streamlit run app.py
```

---

## 🧠 Feature Engineering (13 Features)

All features are computed dynamically per match using only data available *before* the match date (no data leakage):

| Feature | Description |
|---|---|
| `home_ranking` / `away_ranking` | FIFA ranking at match date |
| `ranking_diff` | Ranking gap (home − away) |
| `home_form` / `away_form` | Win % in last 10 matches |
| `form_diff` | Form difference |
| `home_goals_avg` / `away_goals_avg` | Avg goals scored (last 10) |
| `h2h_home_win_rate` / `h2h_draw_rate` / `h2h_away_win_rate` | Historical H2H rates |
| `h2h_total` | Total H2H meetings |
| `is_neutral` | Neutral venue flag |

---

## 📊 Models & Results

| Model | Test Accuracy | CV Accuracy (5-fold) |
|---|---|---|
| **XGBoost** ⭐ | ~62% | ~62% |
| Random Forest | ~61% | ~61% |
| Gradient Boosting | ~61% | ~61% |
| Neural Network (MLP) | ~49% | — |

> Football match prediction has a natural accuracy ceiling of ~65–70% due to the sport's inherent randomness. Our results are consistent with academic research on the topic.

**Most predictive features:** ranking difference, H2H win rates, and form difference.

---

## 🖥️ Streamlit Application (4 Pages)

| Page | Description |
|---|---|
| 📊 **Dashboard** | Interactive EDA with 5 tabs: wins by nation, goal scorers, historical trends, H2H explorer, and world map |
| 🔮 **Predictor** | Pick any two teams → XGBoost predicts the winner with probability cards and H2H context |
| 🏆 **WC26 Predictor** | Simulates the entire FIFA World Cup 2026: group standings + full knockout bracket with team crests |
| 🤖 **AI Analyst** | LLM-powered match analysis (English/Spanish) grounded in real historical statistics |

---

## 🤖 GenAI Component

- **LLM Integration:** OpenRouter free tier (Llama 3.3, Nemotron, Gemma)
- **Multi-model fallback:** Automatically tries 6 models if one is rate-limited
- **Gemma compatibility:** Adapts `system` role to `user` for models that don't support it
- **Data-grounded:** All LLM outputs are based on real statistics from the dataset

---

## 🛠️ Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · XGBoost · TensorFlow/Keras · OpenRouter (LLM) · Streamlit · Plotly · Joblib

---

## 👤 Author

Pablo — Final Project, Data Science Bootcamp 2025
