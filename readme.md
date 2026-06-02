# Customer Churn Prediction
**Codec Technologies Data Analytics Internship — Project 1**

## Churn Prediction Dashboard
  🖇️ https://custome-churn-prediction-bgdrtqvnfpu783fkvruszp.streamlit.app/

## Business Problem
Customer churn (cancellation of service) costs telecom and SaaS companies millions annually.
This project builds an ML pipeline to predict which customers are at risk of churning,
enabling the retention team to act proactively.

## Project Structure
```
customer_churn/
├── data/
│   └── churn.csv               ← Download from Kaggle (link below)
├── notebooks/
│   ├── eda.ipynb               ← Exploratory Data Analysis
│   └── model_training.ipynb   ← Preprocessing, Training, Evaluation, Tuning
├── app/
│   ├── predict.py              ← Streamlit web app
│   ├── best_model.pkl          ← Saved after running model_training.ipynb
│   ├── scaler.pkl              ← Saved after running model_training.ipynb
│   └── feature_names.pkl      ← Saved after running model_training.ipynb
├── report/
│   └── final_report.pdf       ← Your written report
└── requirements.txt
```

## Dataset
- **Telco Customer Churn** from Kaggle
- URL: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- File to download: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Rename to `churn.csv` and place in `data/`

## Setup & Run

### 1. Create virtual environment (Windows)
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run EDA notebook
```bash
jupyter notebook notebooks/eda.ipynb
```

### 4. Train the model
```bash
jupyter notebook notebooks/model_training.ipynb
```
Run all cells — this saves `best_model.pkl`, `scaler.pkl`, `feature_names.pkl` to `app/`

### 5. Launch Streamlit app
```bash
cd app
streamlit run predict.py
```

## Models Used
| Model               | Handles Imbalance     | Notes                        |
|---------------------|-----------------------|------------------------------|
| Logistic Regression | class_weight=balanced | Baseline, interpretable      |
| Random Forest       | class_weight=balanced | Ensemble, robust              |
| XGBoost (Tuned)     | scale_pos_weight      | Best performer, recommended  |

## Key Metrics (Target)
- ROC-AUC: ~0.86
- Recall (Churn class): ~80%
- Accuracy: ~82%

## Features Engineered
- `tenure_group` — 4 tenure buckets (0-12, 12-24, 24-48, 48-72 months)
- `avg_monthly_spend` — TotalCharges / (tenure + 1)
- `is_senior_monthly` — Senior citizen on month-to-month contract (high risk flag)
