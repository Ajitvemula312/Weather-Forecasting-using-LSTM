# 🌦️ Weather Forecasting using LSTM and Feature Fusion  
Time-series forecasting using a deep learning LSTM model with real meteorological data and engineered features.

## 🚀 Overview
This project builds an LSTM model that predicts next-day temperature using:

- Real climatological data (Daily Delhi Climate Dataset)
- Feature fusion (timestamp features + rolling averages)
- LSTM sequence modeling
- Train/predict scripts
- Complete outputs & visualizations

---

## 📊 Dataset
Download dataset from:

https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data

Place this file inside:

```
weather-lstm/data/DailyDelhiClimateTrain.csv
```

Columns include:
- meantemp  
- humidity  
- wind_speed  
- meanpressure  
- date  

---

## 🧩 Feature Engineering

- day, month, year, weekday
- 3-day rolling average
- 7-day rolling average
- Normalization (StandardScaler)
- Sequence generation (30-day window)

---

## 🧠 LSTM Model Architecture

```
Input → LSTM(64, return_sequences=True)
      → LSTM(32)
      → Dense(16, relu)
      → Dense(1)
```

Optimizer: Adam  
Loss: MSE  
EarlyStopping enabled  

---

## 📁 How to Run

### 1️⃣ Create environment
```bash
python3 -m venv venv
source venv/bin/activate    # macOS/Linux
# .\venv\Scripts\activate    # Windows
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Train model
```bash
python3 scripts/train.py
```

Outputs:
- `models/weather_lstm_real.h5`
- `models/scaler.joblib`
- `outputs/training_loss.png`

### 4️⃣ Predict
```bash
python3 scripts/predict.py
```

Output:
- Console prediction  
- `outputs/prediction_plot.png`

---

## 📈 Sample Results (Replace with yours)
Training Loss Curve  
Prediction vs Actual  

---

## 🛠 Future Improvements
- GRU / TCN models  
- 7-day horizon forecasting  
- Streamlit dashboard  
- FastAPI deployment  

---

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
