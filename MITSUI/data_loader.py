from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch

def get_batch_mult(batch_size, elements):
    i = 1
    while i*batch_size < elements:
        i += 1
    
    return (i - 1) * batch_size

def get_data_loader(type : str, batch_size):
    data = np.load(f"MITSUI/tensors/{type}/data.npz")
    x_data = torch.tensor(data['x'])
    y_data = torch.tensor(data['y'])
    dataset = TensorDataset(x_data, y_data)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_loader

def get_test_data():
    data = np.load(f"MITSUI/tensors/test/data.npz")
    x_data = torch.tensor(data['x'])
    y_data = torch.tensor(data['y'])

    return x_data, y_data
