import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, inp, tgt, lr = 1e-2, n_epochs = 100):
    optimizer = torch.optim.Adam(model.parameters(), lr)
    best_loss = 1e6
    for i in range(n_epochs):
        output = model(inp)
        loss = (output-tgt).pow(2).mean()
        if loss < best_loss:
            best_output = output
            best_loss = loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % max((n_epochs//10), 50) == 0:
            print("Epoch {} | MSE: {:0.5f}".format(i+1, loss.cpu().data.numpy()))
        if loss.item() < 1e-7:
            break
    if output.device.type == 'cpu':
        return output.data.numpy()[0,0]
    else:
        return output.cpu().data.numpy()[0,0]
        
        
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

    
    
class DatasetFluid2D(torch.utils.data.Dataset):
    def __init__(self, 
                 direc,
                 sample_list, 
                 x_range = None, 
                 y_range = None
                ):
        self.sample_list = sample_list
        self.dataset = torch.from_numpy(np.load(direc))
        self.x_range = x_range
        self.y_range = y_range

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        sample = self.dataset[:2, index]
        # May only select a subregion based on x_range and y_range
        if self.x_range is not None:
            sample = sample[:,self.x_range]
        if self.y_range is not None:
            sample = sample[:,:,self.y_range]
        x = sample #shape: [2, H, W]
        y = sample #shape: [2, H, W]
        return x.float(), y.float()
    
class Dataset2DTime(torch.utils.data.Dataset):
    def __init__(self, direc, sample_list, length):
        self.sample_list = sample_list
        self.dataset = np.load(direc)
        self.dataset = torch.from_numpy(self.dataset)
        self.length = length

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        sample = self.dataset[:, index:index+self.length]
        x = sample[:, :self.length] 
        y = sample[:, :self.length] 
        return x.float(), y.float()    

def train_epoch_batch(train_loader, model, optimizer, loss_function):
    train_mse = []
    relaxed_weights = []
    for xx, yy in train_loader:
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)
        preds = model(xx)
        loss = loss_function(preds, yy)  
        train_mse.append(loss.item()) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_rmse = round(np.mean(train_mse),5)
    return train_rmse

def eval_epoch_batch(valid_loader, model, loss_function):
    valid_mse = []
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
            valid_mse.append(loss.item())
        all_preds = np.concatenate(all_preds, axis = 0)  
        all_trues = np.concatenate(all_trues, axis = 0)  
        valid_rmse = round(np.mean(valid_mse), 5)
    return valid_rmse, all_preds, all_trues


def train_model_batch(model, train_loader, valid_loader, num_epoch, learning_rate, decay_rate):
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, betas=(0.9, 0.999), weight_decay=4e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=decay_rate)
    loss_fun = torch.nn.MSELoss()
    train_rmse = []
    valid_rmse = []
    relaxed_weights_epochs = []
    min_rmse = 1e6

    for epoch in range(num_epoch):
        start_time = time.time()

        model.train()
        rmse = train_epoch_batch(train_loader, model, optimizer, loss_fun)
        train_rmse.append(rmse)

        model.eval()
        rmse, preds, trues = eval_epoch_batch(valid_loader, model, loss_fun)
        valid_rmse.append(rmse)

        scheduler.step()

        if valid_rmse[-1] < min_rmse:
            min_rmse = valid_rmse[-1] 
            best_model = model

        epoch_time = time.time() - start_time
        # Early Stopping
        if (len(train_rmse) > 30 and np.mean(valid_rmse[-5:]) >= np.mean(valid_rmse[-10:-5])):
                break      
        if epoch % 10 == 0:
            print("Epoch {} | T: {:0.2f} | Train MAE: {:0.5f} | Valid MAE: {:0.5f} | LR {:0.6f}".format(epoch+1, epoch_time/60, train_rmse[-1], valid_rmse[-1], get_lr(optimizer)))
    return best_model