# 🧠 Smart Sales Forecasting System

**A Machine Learning project to predict retail sales with multi-platform deployment**
Developed as Graduation Project of the **Digital Egypt Pioneers Initiative (DEPI)**

---

## 📦 Project Overview

This system forecasts retail sales using machine learning techniques. It covers the complete ML lifecycle:

* 🧹 Data Wrangling
* 🧠 Model Training & Evaluation
* 🚀 Deployment (Desktop GUI, Web App, Streamlit)

---

## 📊 Features Used in Prediction

* 🕒 **Delivery Time**
* 📦 **Quantity**
* 🗂 **Category** (encoded)
* 🔖 **Sub-Category** (encoded)
* 🎯 **Discount (%)**
* 💰 **Profit (\$)**
* 🌟 **Product Popularity**
* 🛒 **Customer Order Count**

---

## 🖥 Deployment Instructions

### ▶ Run Desktop GUI (Tkinter)

```bash
python tkinter_app.py
```

### 🌐 Run Web App (Flask)

```bash
python flask_app.py
# Then open: http://127.0.0.1:5000/
```

### 📈 Run Streamlit App

```bash
streamlit run streamlit_app.py
```

---

## 📈 Model Performance

| Model             | R² Score | RMSE |
| ----------------- | -------- | ---- |
| Linear Regression | 0.14     | 1.47 |
| Ridge Regression  | 0.14     | 1.47 |
| Random Forest     | 0.93     | 0.39 |
| XGBoost Regressor | 0.94     | 0.36 |

🏆 **Best Model:** `XGBoost Regressor`
