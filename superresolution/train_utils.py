import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, direc, sample_list, stack):
        self.direc = direc
        self.sample_list = sample_list
        self.stack = stack

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        sample = torch.load(self.direc + "/sample" + str(self.sample_list[index]) + ".pt")
        x = sample[0] #shape: [3, 3, 32, 32, 32]
        y = sample[1] #shape: [1, 3, 128, 128, 128]
        
        if self.stack:
            x = x.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])     
            y = y.reshape(-1, y.shape[-3], y.shape[-2], y.shape[-1])  
        return x.float(), y.float()

def train_epoch(train_loader, model, optimizer, loss_function, collect_relaxed_weights=False):
    train_mae = []
    relaxed_weights = []
    for xx, yy in train_loader:
        if collect_relaxed_weights:
            iter_weights = []
            iter_weights.append(model.module.embedding.relaxed_weights.data)
            for i in range(4):
                iter_weights.append(model.module.model[i].layer1.relaxed_weights.data)
                iter_weights.append(model.module.model[i].layer2.relaxed_weights.data)
            iter_weights = torch.stack(iter_weights)
            relaxed_weights.append(iter_weights)
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)
        preds = model(xx)
        loss = loss_function(preds, yy)  
        train_mae.append(loss.item()) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_mae = round(np.mean(train_mae),5)
    if collect_relaxed_weights:
        return train_mae, torch.stack(relaxed_weights, dim = 0)
    else:
        return train_mae

def eval_epoch(valid_loader, model, loss_function):
    valid_mae = []
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for xx, yy in valid_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            preds = model(xx)
            loss = loss_function(preds, yy)
            all_preds.append(preds.cpu().data.numpy())
            all_trues.append(yy.cpu().data.numpy())
            valid_mae.append(loss.item())
        all_preds = np.concatenate(all_preds, axis = 0)  
        all_trues = np.concatenate(all_trues, axis = 0)  
        valid_mae = round(np.mean(valid_mae), 5)
    return valid_mae, all_preds, all_trues