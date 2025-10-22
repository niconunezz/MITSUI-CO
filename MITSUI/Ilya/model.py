import torch
import torch.nn.functional as F
from torch import nn
from Ilya.encoder import EncoderBlock
from Ilya.kan import KAN

class Ilya(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.init_projection = nn.Linear(config.ilya.encoder.n_features, config.ilya.encoder.n_embd)
        self.pos_emb = nn.Embedding(config.ilya.encoder.block_size, config.ilya.encoder.n_embd)
        self.encoders = nn.Sequential(*[EncoderBlock(config) for _ in range(config.ilya.encoder.n_blocks)])
        self.kans = KAN(config.ilya.kan.layers_hidden)

        self.gru = nn.GRU(config.ilya.kan.layers_hidden[-1], config.ilya.gru.hidden_size, config.ilya.gru.bigru_layers, batch_first=True)
        self.bigru = nn.GRU(config.ilya.gru.hidden_size, config.ilya.gru.hidden_size, config.ilya.gru.bigru_layers, batch_first=True, bidirectional=True)

        self.norm1 = nn.LayerNorm(config.ilya.gru.hidden_size * 2)
        self.dropout1 = nn.Dropout(0.3)

        self.att_mult = nn.Linear(config.ilya.encoder.block_size, config.ilya.encoder.block_size)
        self.norm2 = nn.LayerNorm(config.ilya.gru.hidden_size * 2)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.3)

        self.lm_head = nn.Linear(config.ilya.gru.hidden_size * 2 * config.ilya.encoder.block_size, 2)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

    def forward(self, x, targets=None):
        B, T, C = x.shape
        x = self.init_projection(x)
        pos = self.pos_emb(torch.arange(T, device=x.device))
        x = x + pos.unsqueeze(0)
        x = self.encoders(x)
        x = self.kans(x)

        x, _ = self.gru(x)
        x, _ = self.bigru(x)

        x = self.norm1(x)
        x = self.dropout1(x)

        att = F.softmax(self.att_mult(x.permute(0,2,1)), dim=-1)
        x = x * att.permute(0,2,1)

        x = self.norm2(x)
        x = self.dropout2(x)

        x = torch.flatten(x, start_dim=1)
        x = self.dropout3(x)
        logits = self.lm_head(x)
        return logits