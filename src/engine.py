import utils
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

def loss_fun(o1,o2,t1,t2):
    l1=nn.BCEWithLogitsLoss()(o1,t1)
    l2=nn.BCEWithLogitsLoss()(o2,t2)
    return l1+l2

def train_fn(data_loader,model,optimizer,device,scheduler):
    model.train()
    losses=utils.AverageMeter( )
    tk0=tqdm(data_loader,total=len(data_loader))

    for bi, d in enumerate(tk0):
        ids = d["ids"]
        token_type_ids = d["tok_type_ids"]
        mask = d["mask"]
        targets_start = d["targets_start"]
        targets_end=d["targets_end"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.float)

        targets_end = targets_end.to(device, dtype=torch.float)

        optimizer.zero_grad()
        o1,o2 = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

        loss = loss_fn(o1,o2, targets_start,targets_end)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(),ids.size(0))
        tk0.set_postfix(loss=losses.avg)