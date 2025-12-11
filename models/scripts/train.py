import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import joblib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
OUT_DIR = os.path.join(ROOT, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "DailyDelhiClimateTrain.csv")

def load_real_weather():
    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    df = df.rename(columns={
        "date": "ds",
        "meantemp": "temp"
    })
    return df

def feature_engineering(df):
    df["day"] = df["ds"].dt.day
    df["month"] = df["ds"].dt.month
    df["year"] = df["ds"].dt.year
    df["dayofweek"] = df["ds"].dt.dayofweek

    # Rolling features
    df["temp_roll_3"] = df["temp"].rolling(3, min_periods=1).mean()
    df["temp_roll_7"] = df["temp"].rolling(7, min_periods=1).mean()

    df = df.dropna().reset_index(drop=True)
    return df

def make_sequences(X, y, seq_len=30):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

def build_lstm_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Dense(16, activation="relu"),
        layers.Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def main():
    df = load_real_weather()
    df = feature_engineering(df)

    target = df["temp"].values
    features = df[["temp", "humidity", "wind_speed", "meanpressure",
                   "day", "month", "year", "dayofweek",
                   "temp_roll_3", "temp_roll_7"]]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    seq_len = 30
    X_seq, y_seq = make_sequences(features_scaled, target, seq_len)

    # Train/test split (time-based)
    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    print("Train shape:", X_train.shape, y_train.shape)

    model = build_lstm_model(X_train.shape[1:])
    es = callbacks.EarlyStopping(patience=8, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[es]
    )

    model.save(os.path.join(MODEL_DIR, "weather_lstm_real.h5"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

    # Training plot
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, "training_loss.png"))

    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()
