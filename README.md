# 🚀 Cryptocurrency Price Prediction System Using BiLSTM

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![Flask](https://img.shields.io/badge/Flask-WebApp-green)
![License](https://img.shields.io/badge/License-Educational-lightgrey)

A **Machine Learning Web Application** that predicts cryptocurrency prices using a **Bidirectional Long Short-Term Memory (BiLSTM)** neural network and visualizes predictions through an interactive dashboard.

This project demonstrates how **deep learning can be applied to financial time-series forecasting**, specifically cryptocurrency markets.

---

# 📌 Project Overview

Cryptocurrency markets are highly volatile, making price prediction a challenging task. This project applies **deep learning techniques**, particularly **Bidirectional LSTM (BiLSTM)**, to learn patterns from historical price data and forecast future price movements.

The system includes:

* A **data pipeline** for fetching and preprocessing crypto data
* A **BiLSTM neural network** for time-series prediction
* A **Flask backend API**
* A **modern interactive dashboard**

---

# ✨ Key Features

✅ Multi-cryptocurrency prediction
✅ Interactive **price visualization**
✅ **Next-day price prediction**
✅ **Color-coded predictions (green/red)**
✅ **Live updating dashboard**
✅ Modular **machine learning pipeline**
✅ **Bidirectional LSTM architecture** for improved forecasting

---

# 🧠 Machine Learning Model

The prediction model uses **Bidirectional Long Short-Term Memory (BiLSTM)**.

### Why BiLSTM?

Traditional LSTM processes sequences **only forward**.

BiLSTM processes data **both forward and backward**, allowing the model to capture deeper temporal patterns.

```
Forward LSTM  → 
               → Combined Representation → Prediction
Backward LSTM →
```

This improves the model’s ability to learn **complex dependencies in time-series data**.

---

# 🏗 Project Architecture

```
                ┌──────────────────┐
                │  Crypto API/Data │
                └─────────┬────────┘
                          │
                          ▼
                ┌──────────────────┐
                │ Data Fetcher     │
                │ utils/data_fetcher.py
                └─────────┬────────┘
                          │
                          ▼
                ┌──────────────────┐
                │ Data Preprocessing │
                │ utils/preprocessing.py
                └─────────┬────────┘
                          │
                          ▼
                ┌──────────────────┐
                │ BiLSTM Model     │
                │ models/bilstm_model.py
                └─────────┬────────┘
                          │
                          ▼
                ┌──────────────────┐
                │ Flask Backend API│
                │ app/app.py       │
                └─────────┬────────┘
                          │
                          ▼
                ┌──────────────────┐
                │ Web Dashboard    │
                │ index.html       │
                └──────────────────┘
```

---

# 📂 Project Structure

```
Crypto-BiLSTM-prediction/
│
├── app/
│   ├── app.py
│   └── templates/
│       └── index.html
│
├── config/
│   └── config.py
│
├── models/
│   └── bilstm_model.py
│
├── utils/
│   ├── data_fetcher.py
│   └── preprocessing.py
│
├── training/
│   └── train_bilstm.py
│
├── data/
│   └── crypto_data.csv
│
├── requirements.txt
└── README.md
```

---

# 📊 Dashboard Preview

### Cryptocurrency Selection

Users can select different cryptocurrencies for prediction.

```
Bitcoin | Ethereum | Solana | Binance Coin | Cardano
```

---

### Prediction Display

The dashboard shows:

* Current price
* Next-day predicted price
* Trend indicator

Example:

```
Current Price: $45,320
Next Day Prediction: $45,870
```

Green → Price increase
Red → Price decrease

---

### Price Chart

Interactive chart displaying:

* historical prices
* predicted prices

Powered by **Chart.js**.

*(Add screenshot here)*

```
docs/dashboard.png
```

---

# ⚙️ Installation

## 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/crypto-bilstm-prediction.git
cd crypto-bilstm-prediction
```

---

## 2️⃣ Create a virtual environment

```bash
python -m venv venv
```

Activate it:

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

---

## 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## Step 1 — Train the model

```
python training/train_bilstm.py
```

---

## Step 2 — Run the Flask application

```
python app/app.py
```

---

## Step 3 — Open the dashboard

```
http://127.0.0.1:5000
```

---

# 🔌 API Endpoint

### Prediction API

```
POST /predict
```

Example request:

```json
{
  "crypto": "bitcoin"
}
```

Example response:

```json
{
  "current_price": 45000,
  "next_prediction": 45200,
  "forecast": [45200, 45500, 46000],
  "dates": ["2026-03-15", "2026-03-16", "2026-03-17"]
}
```

---

# 🧪 Technologies Used

### Programming

* Python

### Machine Learning

* TensorFlow
* Keras
* NumPy
* Pandas
* Scikit-learn

### Backend

* Flask

### Frontend

* HTML
* Bootstrap
* JavaScript
* Chart.js

---

# 🔮 Future Improvements

Possible enhancements include:

* Technical indicators (RSI, MACD)
* Sentiment analysis from social media
* Real-time streaming price updates
* Multi-day forecasting models
* Cloud deployment

---

# 🎓 Academic Context

This project was developed as part of a **Final Year Research Project**:

**Design and Implementation of a Cryptocurrency Price Prediction System Using Machine Learning**

The project demonstrates the practical application of **deep learning techniques in financial prediction systems**.

---

# 👨‍💻 Author

Odika Gabriel

Interests:

* Artificial Intelligence
* Machine Learning
* Financial Prediction Systems

---

# 📜 License

This project is intended for **educational and research purposes only**.
