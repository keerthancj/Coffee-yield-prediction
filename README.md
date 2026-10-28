# CoffeeAI - Smart Coffee Yield Prediction & Climate Suitability System

An end-to-end Machine Learning web application designed to predict coffee yield, estimate total production, assess disease impact, and evaluate climate suitability across key coffee-producing districts.

---

## 🌟 Key Features

- **Machine Learning Yield Prediction**: Predicts yield (Metric Tonnes per hectare) and total production (MT & kg) using a trained Random Forest Regressor.
- **Climate Suitability Assessment**: Analyzes temperature, rainfall, and regional conditions to calculate suitability scores (Ideal, Suitable, Moderate, Unfavorable) for Arabica and Robusta.
- **Disease Impact Modeling**: Evaluates the potential reduction in yield caused by diseases like Coffee Leaf Rust, White Stem Borer, Berry Disease, and Root Disease.
- **Interactive Web Interface**: Clean, modern dashboard built with Flask, HTML5, and CSS.
- **Comprehensive Dataset**: Validated historical coffee production and climate records (2009–2024).

---

## 🛠️ Tech Stack

- **Backend**: Python 3, Flask
- **Machine Learning**: Scikit-Learn, Pandas, NumPy, Joblib
- **Frontend**: HTML5, CSS3, JavaScript
- **Model**: Random Forest Regressor Pipeline with `ColumnTransformer` (One-Hot Encoding + Passthrough)

---

## 📂 Project Structure

```text
├── app.py                                  # Flask application server
├── suitability.py                          # Climate suitability logic & rule engine
├── predict_coffee.py                       # CLI inference and prediction script
├── train_final_model.py                    # Final model training pipeline
├── train_model.py                          # Initial model experiments
├── evaluate_time_based_model.py            # Model evaluation script
├── coffee_ml_validated_dataset_2009_2024.csv # Validated coffee dataset
├── coffee_yield_final_model.pkl            # Pre-trained ML model artifact
├── templates/
│   └── index.html                          # Web UI template
├── static/
│   └── style.css                           # Styling and layout
├── requirements.txt                        # Project dependencies
└── .gitignore                              # Git exclusion rules
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/coffee-yield-prediction.git
cd coffee-yield-prediction
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000`.

---

## 📊 Dataset & Features

The model uses the following key features:
- **Quarter**: Q1, Q2, Q3, Q4
- **State & District**: Coffee growing zones (Karnataka, Kerala, Tamil Nadu, etc.)
- **Coffee Type**: Arabica / Robusta
- **Prediction Season**: Post Blossom / Post Monsoon
- **Area (ha)**: Cultivated area
- **Quarterly Rainfall (mm)**
- **Quarterly Temperatures (Tmax & Tmin in °C)**

