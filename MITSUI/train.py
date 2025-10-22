import torch
from Ilya.model import Ilya
from data_loader import get_data_loader, get_test_data
import time
from configs import Configs


config = Configs()
train_loader = get_data_loader('train', config.main.batch_size)
test_loader = get_data_loader('test', config.main.test_batch_size)


m = Ilya(config).to(config.main.device)
opt = torch.optim.AdamW(m.parameters(), lr=config.main.lr, weight_decay=config.main.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.main.epochs * len(train_loader))

loss_func = config.main.loss_func
for epoch in range(config.main.epochs):
    m.train()
    for i, (x, y) in enumerate(train_loader):

        t0 = time.time()
        x = x.to(config.main.device)
        targets = y.to(config.main.device)

        opt.zero_grad(set_to_none=True)
        preds = m(x)
        loss = 0.5*(loss_func(preds[:,0], targets[:,0]) + loss_func(preds[:,1], targets[:,1]))
            
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(m.parameters(), config.main.grad_clip)
        opt.step()
        # torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0)*1000 # in ms

        # tokens_per_sec = (config.batch_size * config.block_size) / (dt/1000)

        if i % (len(train_loader)//4):
            corr = torch.corrcoef(torch.stack([
            preds[:,0].detach().flatten(), targets[:,0].flatten()]))[0,1].item()
            sign_acc = (torch.sign(preds[:,0]) == torch.sign(targets[:,0])).float().mean().item()

            print(f"epoch {epoch} | step {i} | loss: {loss.item()} | time {dt:.2f} ms | norm: {norm:.4f} |  corr {corr:.4f} | sign_acc {sign_acc:.4f}")
    
    m.eval()    
    x_test, y_test = get_test_data()
    with torch.no_grad():
        x = x_test.to(config.device)
        targets = y_test.to(config.device)
            
        preds = m(x)
        loss = 0.5*(loss_func(preds[:,0], targets[:,0]) + loss_func(preds[:,1], targets[:,1]))
        sign_acc = (torch.sign(preds[:,0]) == torch.sign(targets[:,0])).float().mean().item()
        corr = torch.corrcoef(torch.stack([
            preds[:,0].detach().flatten(), targets[:,0].flatten()]))[0,1].item()
        
        # buscamos corr > 0.2
        # buscamos s != 0.5
        print(f"\nepoch {epoch} | loss {loss} | corr {corr} | sign_acc {sign_acc}\n")