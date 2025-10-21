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
    els = get_batch_mult(batch_size, len(data['x']))

    x_data = torch.tensor(data['x'][:els])
    y_data = torch.tensor(data['y'][:els])
    dataset = TensorDataset(x_data, y_data)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_loader
