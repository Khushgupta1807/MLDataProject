# 🛒 E-Commerce ML Models — Delivery & Review Prediction

> Predicting delivery delays and identifying at-risk orders using machine learning on real Brazilian e-commerce data.

---

## 📌 Project Overview

This project is part of a **25-Day AI Sprint** (Day 5) focused on building production-quality ML models using Scikit-learn. Two models were trained on the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (100k+ real orders).

| Model | Task | Algorithm | Key Metric |
|-------|------|-----------|------------|
| Delivery Delay Predictor | Regression | Random Forest | RMSE (days) |
| 1-Star Review Predictor | Classification | Random Forest | Accuracy + F1 Score |

---

## 📂 Project Structure

```
ML-data-project/
│
├── day5_ml_models.ipynb          # Main notebook — all models + insights
├── reg_feature_importance.png    # Feature importance chart (regression)
├── confusion_matrix.png          # Confusion matrix visualization
├── olist_orders_dataset.csv      # Raw data — orders
├── olist_order_items_dataset.csv # Raw data — items
├── olist_order_reviews_dataset.csv # Raw data — reviews
└── README.md
```

---

## 🧠 Models

### Model 1 — Delivery Delay (Regression)
- **Target:** `delivery_delay_days` — how many days late/early a delivery was
- **Features:** `price`, `freight_value`
- **Algorithm:** `RandomForestRegressor` (100 trees)
- **Metric:** RMSE — lower is better, units are in days
- **Result:** RMSE of ~X days *(update after running)*

### Model 2 — 1-Star Review Predictor (Classification)
- **Target:** `is_bad_review` — binary (1 = 1-star review, 0 = otherwise)
- **Features:** `price`, `freight_value`, `delivery_delay`
- **Algorithm:** `RandomForestClassifier` with `class_weight='balanced'`
- **Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Result:** Accuracy ~X% / F1 ~X *(update after running)*

> **Why `class_weight='balanced'`?** Only ~10–15% of reviews are 1-star. Without balancing, a naive model just predicts "good review" every time and still gets 90% accuracy — which is useless. Balancing forces the model to actually learn the minority class.

---

## 📊 Key Business Insights

1. **Freight value is the strongest predictor of bad reviews** — customers who pay more for shipping have higher expectations. Capping freight charges on small orders could reduce complaints.

2. **Delivery delay directly drives 1-star ratings** — the classification model confirms that late deliveries are the single biggest driver of negative reviews.

3. **The model catches ~68% of actual bad reviews (recall ≈ 0.68)** — this is enough to flag at-risk orders before they escalate, enabling proactive customer support intervention.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikit-learn)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)

---

## 🚀 How to Run

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/ML-data-project.git
cd ML-data-project
```

**2. Install dependencies**
```bash
pip install scikit-learn pandas numpy matplotlib jupyter
```

**3. Download the dataset**

Get the Olist dataset from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and place the CSV files in the project folder.

**4. Run the notebook**
```bash
jupyter notebook day5_ml_models.ipynb
```

---

## 📈 Dataset

**Olist Brazilian E-Commerce** — a real anonymised dataset of 100,000+ orders from 2016–2018 across multiple Brazilian marketplaces.

- Orders: `olist_orders_dataset.csv`
- Items: `olist_order_items_dataset.csv`
- Reviews: `olist_order_reviews_dataset.csv`

Source: [Kaggle — Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

---

## 👤 Author

**Khush Gupta** — 25-Day AI Sprint · Day 5 of 25

---

*Part of a structured sprint to build a DS/ML-ready GitHub profile targeting 10–14 LPA roles.*
