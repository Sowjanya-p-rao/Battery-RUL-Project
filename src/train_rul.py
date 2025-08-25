"""
Training script for Remaining Useful Life (RUL) prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.models.lstm import build_lstm
import tensorflow as tf

DATA_FILE = "data/processed/cell01.csv"

def load_data(file):
    df = pd.read_csv(file)
    X = df[["voltage", "current", "temperature"]].values
    y = df["capacity"].values
    # Define SOH = C/C0, RUL = N_EOL – cycle
    C0 = y[0]
    SOH = y / C0
    RUL = len(y) - np.arange(len(y))
    return X, RUL

def train_model():
    X, y = load_data(DATA_FILE)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Reshape for LSTM [samples, timesteps, features]
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)

    model = build_lstm(input_shape=(1, X_train.shape[2]))
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    preds = model.predict(X_test).flatten()
    print("RMSE:", mean_squared_error(y_test, preds, squared=False))
    print("MAE:", mean_absolute_error(y_test, preds))
    print("R²:", r2_score(y_test, preds))

if __name__ == "__main__":
    train_model()
