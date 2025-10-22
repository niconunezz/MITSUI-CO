import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

window = 45
dtype = np.float32

df = pd.read_csv("MITSUI/hull-tactical-market-prediction/train.csv")
df = df.sort_values("date_id").reset_index(drop=True)
df = df.ffill().bfill()

y_cols = ["forward_returns", "market_forward_excess_returns"]
feature_cols = [c for c in df.columns if c not in ["risk_free_rate"] + y_cols + ["date_id"]]

X = df[feature_cols].values.astype(dtype)
y = df[y_cols].values.astype(dtype)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# escalar sin leakage
scaler_X = StandardScaler().fit(X_train)
scaler_y = StandardScaler().fit(y_train)

X_scaled = np.vstack([scaler_X.transform(X_train), scaler_X.transform(X_test)])
y_scaled = np.vstack([scaler_y.transform(y_train), scaler_y.transform(y_test)])

# target del día siguiente
y_shifted = np.roll(y_scaled, -1, axis=0)
y_shifted[-1] = y_shifted[-2]

def create_sequences(X, y, window):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i+window])
        ys.append(y[i+window])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_shifted, window)
train_size = int(len(X_seq) * 0.8)

np.savez_compressed("MITSUI/tensors/train/data", 
                    x=X_seq[:train_size].astype(dtype),
                    y=y_seq[:train_size].astype(dtype))
np.savez_compressed("MITSUI/tensors/test/data", 
                    x=X_seq[train_size:].astype(dtype),
                    y=y_seq[train_size:].astype(dtype))

joblib.dump(scaler_X, "MITSUI/tensors/scaler_X.pkl")
joblib.dump(scaler_y, "MITSUI/tensors/scaler_y.pkl")
