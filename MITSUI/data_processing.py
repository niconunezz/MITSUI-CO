import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


window = 45
dtype = np.float32

for data_type in ['train']:
    df = pd.read_csv(f"MITSUI/hull-tactical-market-prediction/{data_type}.csv").dropna().sort_values("date_id").reset_index(drop=True)

    y_cols = ['forward_returns', 'risk_free_rate' , 'market_forward_excess_returns'] if data_type=='train' else ['lagged_forward_returns',
                                                                                                                  'lagged_risk_free_rate',
                                                                                                                  'lagged_market_forward_excess_returns']
    feature_cols = [col for col in df.columns if col not in y_cols]

    x = df[feature_cols].values

    y_cols = ['forward_returns', 'market_forward_excess_returns']if data_type=='train' else ['lagged_forward_returns', 'lagged_market_forward_excess_returns']
    y = df[y_cols].values

    #! recordar desescalar
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(x)
    y_scaled = scaler_y.fit_transform(y)

    def create_sequences(X, y, window):
        Xs, ys = [], []
        for i in range(len(X) - window):
            Xs.append(X[i:i+window])
            ys.append(y[i+window])
        return np.array(Xs), np.array(ys)


    X_seq, y_seq = create_sequences(X_scaled, y_scaled, window)
    X_seq = X_seq.astype(dtype)
    y_seq = y_seq.astype(dtype)

    train_percentaje = int(len(X_seq)*0.8)
    X_train = X_seq[:train_percentaje]
    X_test = X_seq[train_percentaje:]

    Y_train = y_seq[:train_percentaje]
    y_test = y_seq[train_percentaje:]
    np.savez_compressed(f'MITSUI/tensors/train/data', x=X_train, y=Y_train)
    np.savez_compressed(f'MITSUI/tensors/test/data', x=X_test, y=y_test)

