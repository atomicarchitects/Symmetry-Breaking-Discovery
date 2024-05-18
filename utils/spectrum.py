import torch
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def FourierFilter(inp, lower, upper):
    pass1 = (torch.abs(torch.fft.fftfreq(inp.shape[-1])) >= lower) & (torch.abs(torch.fft.fftfreq(inp.shape[-1])) <= upper)
    pass2 = (torch.abs(torch.fft.fftfreq(inp.shape[-2])) >= lower) & (torch.abs(torch.fft.fftfreq(inp.shape[-2])) <= upper)
    kernel = torch.outer(pass2, pass1).to(inp.device)
    fft_input = torch.fft.fft2(inp)
    return torch.fft.ifft2(fft_input * kernel, s=inp.shape[-2:]).real


def get_Emean(T, du_dns, fsval, height, d):
    Elist = []
    for t in range(T):
        f,E = signal.welch(du_dns[t,height,:], fs = fsval, scaling = 'spectrum') 
        Elist.append(E)
    E_mean = np.asarray(Elist).mean(axis=0)
    return f, E_mean


def spectrum1d(dns_video, target_yplus = 150, utau = 0.045, nu = 2.5e-4, retau=180, direction = 'spanwise',ytol = 1e-5):
    # sampled_video (dns_video) : dim 1x4xTimex256x128
    # extract information
    if direction == 'spanwise':
        fsval = 128/2/np.pi
    elif direction == 'streamwise':
        fsval = 160/2/np.pi
    T = dns_video.shape[2]
    dy = 2/256
    y = np.linspace(2/256/2,2,256)
    yplus = y * utau / nu
    ylabels  = [r'$\overline{E^+}_{uu}$', r'$\overline{E^+}_{vv}$', r'$\overline{E^+}_{ww}$']
    # for loop u v w
    # for d in range(3):
    d = 0
    # get mean
    u_dns = dns_video[0,d,:,:,:].mean(axis = 0).mean(axis = 1).reshape([1,-1,1])
    # fluctaion
    du_dns = dns_video[0,d,:,:,:] - u_dns
    # different normalized height
    height = int(target_yplus * nu / utau / dy)
    f, E_mean_dns = get_Emean(T, du_dns, fsval, height,d)
    return f, E_mean_dns
    