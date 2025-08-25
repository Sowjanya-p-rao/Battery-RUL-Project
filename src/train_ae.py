"""
Training script for Autoencoder anomaly detection
Detects early degradation by reconstruction error
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from src.models.autoencoder import build_autoencoder
import tensorflow as tf

DATA_FILE = "data/processed/cell01.csv"

def load_data(file):
    df = pd.read_csv(file)
    X = df[["voltage", "current", "temperature"]].values
    return X

def train_autoencoder():
    X = load_data(DATA_FILE)
    X_train, X_test = train_test_split(X, test_size=0.2, shuffle=False)

    model = build_autoencoder(input_dim=X.shape[1])
    model.fit(X_train, X_train, epochs=20, batch_size=64, validation_split=0.2)

    # Compute reconstruction error
    recon = model.predict(X_test)
    errors = np.mean((X_test - recon) ** 2, axis=1)

    # Threshold anomaly detection
    threshold = np.percentile(errors, 95)
    anomalies = errors > threshold

    print(f"Anomaly Threshold: {threshold}")
    print(f"Detected anomalies: {np.sum(anomalies)}")

if __name__ == "__main__":
    train_autoencoder()
