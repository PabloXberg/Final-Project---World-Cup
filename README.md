# ⚽ FIFA World Cup Predictor — End-to-End Data Science Project

Un proyecto completo de Data Science que predice resultados de la Copa Mundial FIFA usando Machine Learning, Deep Learning y GenAI.

## 🎯 Objetivo

Construir un sistema end-to-end que:
1. Analiza el historial completo de fútbol internacional (1872–2024)
2. Entrena múltiples modelos ML/DL para predecir resultados
3. Integra un LLM real (Llama 3 via Groq) para análisis deportivo inteligente
4. Expone todo en una aplicación interactiva con Streamlit

---

## 🗂️ Estructura del Proyecto

```
FIFA-World-Cup-Predictor/
│
├── db/
│   ├── results.csv          # Partidos internacionales 1872-2024
│   └── ranking.csv          # Rankings FIFA históricos
│
├── main.ipynb               # Notebook principal (EDA + ML + GenAI)
├── app.py                   # Aplicación Streamlit
├── requirements.txt         # Dependencias
├── .env                     # API Keys (NO subir a GitHub)
├── .gitignore
└── README.md
```

---

## 📦 Datasets Necesarios

Descargar de Kaggle y colocar en la carpeta `db/`:

| Archivo | Fuente Kaggle |
|---|---|
| `results.csv` | [martj42/international-football-results-from-1872-to-2022](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2022) |
| `ranking.csv` | [cashncarry/fifaworldranking](https://www.kaggle.com/datasets/cashncarry/fifaworldranking) |

---

## ⚙️ Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/FIFA-World-Cup-Predictor.git
cd FIFA-World-Cup-Predictor

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar API Key de Groq (gratis en groq.com)
# Crear archivo .env con:
# GROQ_API_KEY=tu_api_key_aqui
```

---

## 🚀 Uso

### Ejecutar el Notebook
```bash
jupyter notebook main.ipynb
```

### Ejecutar la App
```bash
streamlit run app.py
```

---

## 🧠 Features del Modelo

El modelo utiliza **13 features** calculadas dinámicamente para cada partido:

| Feature | Descripción |
|---|---|
| `home_ranking` | Ranking FIFA del equipo local |
| `away_ranking` | Ranking FIFA del equipo visitante |
| `ranking_diff` | Diferencia de rankings |
| `home_form` | % victorias en últimos 10 partidos |
| `away_form` | % victorias en últimos 10 partidos |
| `form_diff` | Diferencia de forma |
| `home_goals_avg` | Promedio goles marcados (últimos 10) |
| `away_goals_avg` | Promedio goles marcados (últimos 10) |
| `h2h_home_win_rate` | Tasa de victorias históricas H2H |
| `h2h_draw_rate` | Tasa de empates históricos H2H |
| `h2h_away_win_rate` | Tasa de victorias visitante H2H |
| `h2h_total` | Total de enfrentamientos históricos |
| `is_neutral` | ¿Se juega en campo neutral? |

---

## 📊 Modelos Implementados

- **XGBoost** — Gradient boosting optimizado
- **Random Forest** — Ensemble de árboles de decisión  
- **Red Neuronal (MLP)** — Deep Learning con Keras/TensorFlow

---

## 🤖 Componente GenAI

Integración con **Llama 3.3-70B** via Groq API (gratuita):
- Análisis contextual del partido basado en datos históricos
- Chatbot interactivo con contexto de datos reales
- Generación de crónicas deportivas fundamentadas

---

## 📈 Resultados

| Modelo | Accuracy | 
|---|---|
| XGBoost | ~65% |
| Random Forest | ~63% |
| Red Neuronal | ~61% |

*Nota: Predecir fútbol tiene un límite natural de precisión (~65-70%) debido a la naturaleza aleatoria del deporte.*

---

## 🛠️ Tecnologías

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![Groq](https://img.shields.io/badge/Groq-LLaMA3-green)

---

## 👤 Autor

Proyecto Final — Data Science Bootcamp 2024
