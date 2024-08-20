# %%
import numpy as np
import astra
import torch
import os 
from config import config
import matplotlib.pyplot as plt
import pandas as pd
import torch_radon
import h5py


def my_grad(X):
    fx = torch.cat((X[1:,:], X[-1,:].unsqueeze(0)), dim=0) - X
    fy = torch.cat((X[:,1:], X[:,-1].unsqueeze(1)), dim=1) - X
    return fx, fy

def my_div(Px, Py):
    fx = Px - torch.cat((Px[0,:].unsqueeze(0), Px[0:-1,:]), dim=0)
    fx[0,:] = Px[0,:]
    fx[-1,:] = -Px[-2,:]
   
    fy = Py - torch.cat((Py[:,0].unsqueeze(1), Py[:,0:-1]), dim=1)
    fy[:,0] = Py[:,0]
    fy[:,-1] = -Py[:,-2]

    return fx + fy

def tv(x0, A, g, alpha, L, Niter, f=None, print_flag=True):

    tau = 1/L
    sigma = 1/L
    theta = 1
    grad_scale = 1e+2
    if f is not None:
        m, n = f.shape
    Nal, Ns = g.shape
    p    = torch.zeros_like(g).to(config.device)
    qx   = x0
    qy   = x0
    u    = x0
    ubar = x0

    alpha = torch.Tensor([alpha]).to(config.device)
    zero_t = torch.Tensor([0]).to(config.device)

    error = torch.zeros(Niter)
    for k in range(Niter):

        p  = (p + sigma*(A.forward(ubar) - g))/(1+sigma)
        
        ubarx, ubary = my_grad(ubar)
        if alpha > 0:
            qx = alpha*(qx + grad_scale*sigma*ubarx)/torch.maximum(alpha, torch.abs(qx + grad_scale*sigma*ubarx)) 
            qy = alpha*(qy + grad_scale*sigma*ubary)/torch.maximum(alpha, torch.abs(qy + grad_scale*sigma*ubary))
            uiter = torch.maximum(zero_t, u - tau*(A.backward(p) - grad_scale*my_div(qx, qy)))
        else:
            uiter = torch.maximum(zero_t, u - tau*A.backward(p))
                                  
        ubar = uiter + theta*(uiter - u)
        u = ubar
        
        if f is not None:
            error[k] = torch.sum(torch.abs(ubar - f)**2)/torch.sum(torch.abs(f)**2)
        if print_flag and np.mod(k+1, 100) == 0:
            print('TV Iteration: ' + str(k+1) + '/' + str(Niter) + ', Error: ' + str(error[k].item()))
      
    return ubar


if __name__ == "__main__":
    device = "cuda"
    index = 1366
    images = (h5py.File('randshepp.mat')['data'][:]).transpose([-1,0,1])

    phantom = images[index,:,:]
    phantom = phantom/np.max(phantom)
    # phantom *= 0.006
    x = torch.Tensor(phantom).to(device)
    Nal = 80
    angles = np.linspace(-np.pi/3, np.pi/3, Nal, endpoint=False)
    NUM_ANGLES = len(angles)


    radon = torch_radon.Radon(128, angles, det_count=128, det_spacing=1, clip_to_circle=True)
    
    sinogram = radon.forward(x)
    noise = torch.randn(*sinogram.shape).to(device)
    sinogram += 0.03*torch.max(torch.abs(sinogram))*noise

    x0 = torch.zeros([128, 128]).to(device)
    L = 400
    Niter = 500
    alpha = 0.04
    rec = tv(x0, radon, sinogram, alpha, L, Niter, f=x, print_flag=True)


    plt.figure()
    plt.imshow(rec.cpu().numpy())
    plt.colorbar()
    plt.savefig("tv.png")