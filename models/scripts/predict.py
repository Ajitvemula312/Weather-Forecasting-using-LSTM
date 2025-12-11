import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
OUT_DIR = os.path.join(ROOT, "outputs")

MODEL_PATH = os.path.join(MODEL_DIR, "weather_lstm_real.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
CSV_PATH = os.path.join(DATA_DIR, "DailyDelhiClimateTrain.csv")

def load_data():
    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    df = df.rename(columns={"date": "ds", "meantemp": "temp"})
    
    df["day"] = df["ds"].dt.day
    df["month"] = df["ds"].dt.month
    df["year"] = df["ds"].dt.year
    df["dayofweek"] = df["ds"].dt.dayofweek
    df["temp_roll_3"] = df["temp"].rolling(3, min_periods=1).mean()
    df["temp_roll_7"] = df["temp"].rolling(7, min_periods=1).mean()

    df = df.dropna().reset_index(drop=True)

    features = df[["temp", "humidity", "wind_speed", "meanpressure",
                   "day", "month", "year", "dayofweek",
                   "temp_roll_3", "temp_roll_7"]]
    return df, features

def main():
    df, features = load_data()
    scaler = joblib.load(SCALER_PATH)
    model = load_model(MODEL_PATH)

    seq_len = 30
    X_scaled = scaler.transform(features)
    recent_seq = X_scaled[-seq_len:]
    X_input = np.expand_dims(recent_seq, axis=0)

    pred = model.predict(X_input)[0][0]
    print("Predicted next-day mean temperature:", pred)

    # Plot last 50 days + prediction
    plt.figure(figsize=(10,4))
    plt.plot(df["ds"].tail(50), df["temp"].tail(50), label="Actual Temp")
    plt.axhline(pred, color="red", linestyle="--", label="Predicted Next Day")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "prediction_plot.png"))

    print("Prediction plot saved.")

if __name__ == "__main__":
    main()
