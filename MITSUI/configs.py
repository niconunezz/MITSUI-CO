from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class MainConfig:
    batch_size = 64
    test_batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 10
    lr = 1e-4
    weight_decay = 1e-2
    loss_func = nn.MSELoss(reduction="mean")
    grad_clip = 0.5

@dataclass
class TransformerConfig:
    block_size = 45 # window
    n_embd = 384
    n_heads = 6
    n_blocks = 6
    n_features = 95

@dataclass
class KanConfig:
    layers_hidden = [384, 384, 384, 384, 384]

@dataclass
class GruConfig:
    hidden_size = 384
    bigru_layers = 1

@dataclass
class IlyaConfigs:
    main = MainConfig()
    encoder = TransformerConfig()
    kan = KanConfig()
    gru = GruConfig()

@dataclass
class Configs:
    main = MainConfig()
    ilya = IlyaConfigs()