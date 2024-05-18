import torch
from torch import nn
import random
import time
import numpy as np
import argparse
import os
from modules import NonEquivResNet, EquivResNet, Relaxed_EquivResNet
from train_utils import Dataset, train_epoch, eval_epoch
from utils import get_lr
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.device_count())


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



parser = argparse.ArgumentParser(description='Super Resolution')
parser.add_argument('--dataset_name', type=str, required=True, default="isotropic", help='Isotropic Flow or Channel Flow')
parser.add_argument('--kernel_size', type=int, required=False, default="3", help='convolution kernel size')
parser.add_argument('--batch_size', type=int, required=False, default="2", help='batch size')
parser.add_argument('--num_epoch', type=int, required=False, default="100", help='maximum number of epochs')
parser.add_argument('--learning_rate', type=float, required=False, default="0.001", help='learning rate')
parser.add_argument('--decay_rate', type=float, required=False, default="0.95", help='learning decay rate')
parser.add_argument('--num_filter_basis', type=int, required=False, default="1", help='number of basis for relaxed convolution')
parser.add_argument('--reflection', type=str2bool, required=False, default=False, help='whether include reflections')
parser.add_argument('--hidden_dim', type=int, required=False, default="1", help='hidden dimension')
parser.add_argument('--seed', type=int, required=False, default="0", help='random seed')
parser.add_argument('--in_frames', type=int, required=False, default=3, help='#input frames')
parser.add_argument('--out_frames', type=int, required=False, default=1, help='#output frames')
parser.add_argument('--model', type=str, required=False, default="EquivResNet", help='model name')
parser.add_argument('--equiv_last', type=str2bool, required=False, default=True, help='whether to use equivariant last layer')
parser.add_argument('--separable', type=str2bool, required=False, default=False, help='whether to use separable group convolution')

args = parser.parse_args()
in_frames = args.in_frames
out_frames = args.out_frames
dataset_name = args.dataset_name
seed = args.seed
batch_size = args.batch_size
num_filter_basis = args.num_filter_basis
kernel_size = args.kernel_size
hidden_dim = args.hidden_dim
learning_rate = args.learning_rate
decay_rate = args.decay_rate
num_epoch = args.num_epoch
reflection = args.reflection
model_name = args.model
equiv_last = args.equiv_last
separable = args.separable

if model_name == "NonEquivResNet":
    stack = True
else:
    stack = False
    
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if dataset_name == "isotropic":
    train_direc = "/pscratch/sd/r/rwang2/SymmetryBreaking/3dTurbulence/datasets/isotropic_flow_train"
    valid_direc = "/pscratch/sd/r/rwang2/SymmetryBreaking/3dTurbulence/datasets/isotropic_flow_train"
    test_direc = "/pscratch/sd/r/rwang2/SymmetryBreaking/3dTurbulence/datasets/isotropic_flow_train"
    train_indices = np.arange(0, 1200)
    valid_indices = np.arange(1200, 1500)
    test_indices = np.arange(1500, 1800)
elif dataset_name == "channel":
    train_direc = "/pscratch/sd/r/rwang2/SymmetryBreaking/3dTurbulence/datasets/channel_flow_train"
    valid_direc = "/pscratch/sd/r/rwang2/SymmetryBreaking/3dTurbulence/datasets/channel_flow_train"
    test_direc = "/pscratch/sd/r/rwang2/SymmetryBreaking/3dTurbulence/datasets/channel_flow_train"
    train_indices = np.arange(0, 1200)
    valid_indices = np.arange(1200, 1500)
    test_indices = np.arange(1500, 1800)
else:
    raise ValueError('Wrong dataset name')

train_set = Dataset(direc = train_direc, 
                    sample_list = train_indices,
                    stack = stack)

valid_set = Dataset(direc = train_direc, 
                    sample_list = valid_indices,
                    stack = stack)

test_set = Dataset(direc = train_direc, 
                   sample_list = test_indices,
                   stack = stack)

train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 4)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size = batch_size, shuffle = False, num_workers = 4)    
test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 4)    
        
save_name = "dataset{}_equlast{}_model{}_seed{}_lr{}_bz{}_dim{}_reflection{}_basis{}_separable{}".format(dataset_name, equiv_last, model_name, seed, learning_rate, batch_size, hidden_dim, reflection, num_filter_basis, separable)
save_path = "/pscratch/sd/r/rwang2/SymmetryBreaking/3dTurbulence/results/" + dataset_name + "/"

print(save_name)
if os.path.exists(save_path + save_name + ".pth"):
    model, last_epoch, learning_rate, all_relaxed_weights = torch.load(save_path + save_name + ".pth")
    print("Resume Training")
    print("last_epoch:", last_epoch, "learning_rate:", learning_rate)
    best_model = model
else:
    all_relaxed_weights = []
    last_epoch = 0
    print("New Model")
    if model_name == "NonEquivResNet":
        model = nn.DataParallel(NonEquivResNet(in_channels = in_frames*3, 
                                               out_channels = out_frames*3, 
                                               hidden_dim = hidden_dim, 
                                               kernel_size = kernel_size).to(device))

    elif model_name == "EquivResNet":
        model = nn.DataParallel(EquivResNet(in_channels = in_frames, 
                                            out_channels = out_frames, 
                                            hidden_dim = hidden_dim, 
                                            reflection = reflection,
                                            kernel_size = kernel_size,
                                            separable = separable,
                                            vec_inp = True, 
                                            equiv_lastlayer = equiv_last).to(device))

    elif model_name == "Relaxed_EquivResNet":
        model = nn.DataParallel(Relaxed_EquivResNet(in_channels = in_frames, 
                                                    out_channels = out_frames, 
                                                    hidden_dim = hidden_dim, 
                                                    reflection = reflection,
                                                    num_filter_basis = num_filter_basis,
                                                    kernel_size = kernel_size,
                                                    separable = separable,
                                                    vec_inp = True,
                                                    equiv_lastlayer = equiv_last).to(device))
        
    else:
        raise ValueError('Wrong model name')

print(model_name)
print("number of paramters:", sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6)


optimizer = torch.optim.Adam(model.parameters(), learning_rate, betas=(0.9, 0.999), weight_decay=4e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=decay_rate)
loss_fun = torch.nn.L1Loss()


train_mae = []
valid_mae = []
min_mae = 1e6

for epoch in range(last_epoch, num_epoch):
    start_time = time.time()
    
    model.train()
    if model_name == "Relaxed_EquivResNet":
        mae, relaxed_weights = train_epoch(train_loader, model, optimizer, loss_fun, collect_relaxed_weights = True)
        all_relaxed_weights.append(relaxed_weights.cpu().numpy())
    else:
        mae = train_epoch(train_loader, model, optimizer, loss_fun, collect_relaxed_weights = False)
    train_mae.append(mae)
    

    model.eval()
    mae, preds, trues = eval_epoch(valid_loader, model, loss_fun)
    valid_mae.append(mae)
    
    scheduler.step()

    if valid_mae[-1] < min_mae:
        min_mae = valid_mae[-1] 
        best_model = model
    torch.save([model, epoch, get_lr(optimizer), all_relaxed_weights],
               save_path + save_name + ".pth")


    epoch_time = time.time() - start_time
    
    # Early Stopping
    if (len(train_mae) > 30 and np.mean(valid_mae[-5:]) >= np.mean(valid_mae[-10:-5])):
            break       
    print("{} | Epoch {} | T: {:0.2f} | Train MAE: {:0.3f} | Valid MAE: {:0.3f} | LR {:0.6f}".format(model_name, epoch+1, epoch_time/60, train_mae[-1], valid_mae[-1], get_lr(optimizer)))
    

test_mae, test_preds, test_tgts = eval_epoch(test_loader, best_model, loss_fun)
torch.save({"best_eval_rmse": min_mae, 
            "test": [test_mae, test_preds, test_tgts],
            "model": best_model,
            "relaxed_weights": np.array(all_relaxed_weights)},
            save_path + "test_results_" + save_name + ".pt")