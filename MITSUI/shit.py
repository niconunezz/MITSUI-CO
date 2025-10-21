import pandas as pd
import numpy as np

df = pd.read_csv("MITSUI/hull-tactical-market-prediction/train.csv")
print(len(df.columns))

data = np.load("MITSUI/tensors/test/data.npz")
print(data['y'].shape)