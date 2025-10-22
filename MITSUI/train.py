import torch
import torch.nn as nn
from Ilya.model import Ilya
from data_loader import get_data_loader, get_test_data
import time
from configs import Configs


config = Configs()
train_loader = get_data_loader('train', config.main.batch_size)
test_loader = get_data_loader('test', config.main.test_batch_size)


def corr_loss(preds, targets):
    vx = preds[:,0] - preds[:,0].mean()
    vy = targets[:,0] - targets[:,0].mean()
    corr1 = (vx * vy).sum() / (torch.sqrt((vx**2).sum()) * torch.sqrt((vy**2).sum()))

    mse = nn.MSELoss(reduction="mean")
    p1 =  0.6* mse(preds[:,0], targets[:,0]) + 0.4*(1 - (corr1))
    # p2 =  0.6* mse(preds[:,1], targets[:,1]) + 0.4*(1 - corr2)
    return p1 #+ p2

m = Ilya(config).to(config.main.device)
opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-4, steps_per_epoch=len(train_loader), epochs=config.main.epochs)


loss_func = corr_loss
corrs1 = []
corrs2 = []
for epoch in range(config.main.epochs):
    m.train()
    for i, (x, y) in enumerate(train_loader):

        t0 = time.time()
        x = x.to(config.main.device)
        targets = y.to(config.main.device)

        opt.zero_grad(set_to_none=True)
        preds = m(x)

        loss = loss_func(preds, targets)
            
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(m.parameters(), config.main.grad_clip)
        opt.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0)*1000 # in ms

        # tokens_per_sec = (config.batch_size * config.block_size) / (dt/1000)

        # if i % (len(train_loader)//4) == 0:
            
        #     sign_acc = (torch.sign(preds[:,0]) == torch.sign(targets[:,0])).float().mean().item()

        #     print(f"epoch {epoch} | step {i} | loss: {loss.item()} | time {dt:.2f} ms | norm: {norm:.4f} |  corr1 {corr1:.4f} | sign_acc {sign_acc:.4f}")
    
    m.eval()    
    x_test, y_test = get_test_data()

    with torch.no_grad():
        x = x_test.to(config.main.device)
        targets = y_test.to(config.main.device)
            
        preds = m(x)
        sign_acc = (torch.sign(preds[:,0]) == torch.sign(targets[:,0])).float().mean().item()
        corr1 = torch.corrcoef(torch.stack([
            preds[:,0].detach().flatten(), targets[:,0].flatten()]))[0,1].item()
        
        corr2 = torch.corrcoef(torch.stack([
            preds[:,1].detach().flatten(), targets[:,1].flatten()]))[0,1].item()
        
        corrs1.append(corr1)
        corrs2.append(corr2)
        loss = loss_func(preds, targets)        
        
        # buscamos corr > 0.2
        # buscamos s != 0.5
        print(f"\nepoch {epoch} | loss {loss} | corr1 {corr1} | corr2 {corr2} | sign_acc {sign_acc}\n")

print(f"corrs1 : {corrs1}, corrs2 : {corrs2}")