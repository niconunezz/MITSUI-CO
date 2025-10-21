import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from model import Ilya
import matplotlib.pyplot as plt
from dataclasses import dataclass
from data_loader import get_data_loader
import time

@dataclass
class Config:
    batch_size = 64
    block_size = 45 # window
    n_embd = 384
    n_heads = 6
    n_blocks = 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layers_hidden = [n_embd, n_embd, n_embd, n_embd, n_embd]
    hidden_size = n_embd
    bigru_layers = 1
    n_features = 95
    epochs = 10

config = Config()

train_loader = get_data_loader('train', config.batch_size)
test_loader = get_data_loader('test', config.batch_size)

# torch.set_float32_matmul_precision('high')

m = Ilya(config).to(config.device)

opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.epochs * len(train_loader))

mse = nn.MSELoss(reduction="mean")
for epoch in range(config.epochs):
    m.train()
    for i, (x, y) in enumerate(train_loader):

        t0 = time.time()
        x = x.to(config.device)
        targets = y.to(config.device)

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            preds = m(x)
            loss = 0.5*(mse(preds[:,0], targets[:,0]) + mse(preds[:,1], targets[:,1]))
            
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 0.5)
        opt.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0)*1000 # in ms

        tokens_per_sec = (config.batch_size * config.block_size) / (dt/1000)
        if i % (len(train_loader)//4):
            corr = torch.corrcoef(torch.stack([
            preds[:,0].detach().flatten(), targets[:,0].flatten()]))[0,1].item()
            sign_acc = (torch.sign(preds[:,0]) == torch.sign(targets[:,0])).float().mean().item()

            print(f"epoch {epoch} | step {i} | loss: {loss.item()} | time {dt:.2f} ms | norm: {norm:.4f} |  corr {corr:.4f} | sign_acc {sign_acc:.4f}")
    
    m.eval()
    eval_losses = []
    sign_accs = []
    for i, (x, y) in enumerate(test_loader):
        
        with torch.no_grad():
            x = x.to(config.device)
            targets = y.to(config.device)
            
            preds = m(x)
            loss = 0.5*(mse(preds[:,0], targets[:,0]) + mse(preds[:,1], targets[:,1]))
            sign_acc = (torch.sign(preds[:,0]) == torch.sign(targets[:,0])).float().mean().item()
            
            eval_losses.append(loss)
            sign_accs.append(sign_acc)
    
    print(f"\nepoch {epoch} | mean loss {sum(eval_losses)/(i+1)} | mean sign_acc {sum(sign_accs)/(i+1)}\n")


