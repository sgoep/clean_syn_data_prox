# %%
import torch
import torch.nn as nn
import numpy as np
from config import config

class range_layer(nn.Module):
    def __init__(self):
        super(range_layer, self).__init__()

    def forward(self, x, A, B):
        # y = self.B(self.A(x))
        Z = torch.zeros_like(x)
        
        for j in range(x.shape[0]):
            Z[j,0,:,:] = B(A(x[j,0,:,:]))
        return Z
        # return y


class null_space_layer(nn.Module):
    def __init__(self):
        super(null_space_layer, self).__init__()        

    def forward(self, x, A, B):
        Z = torch.zeros_like(x)
        for j in range(x.shape[0]):
            Z[j,0,:,:] = x[j,0,:,:] - B(A(x[j,0,:,:]))
        return Z
        # y = x - self.B(self.A(x))
        # return y


class proximal_layer(nn.Module):
    def __init__(self, ell2_norm):
        super(proximal_layer, self).__init__()
        self.ell2_norm = ell2_norm
        
    def Phi(self, x):
        y = torch.zeros_like(x)
        norm = torch.linalg.norm(x)
        if norm < self.ell2_norm:
            y = x
        else:
            y = self.ell2_norm*x/torch.sqrt(norm**2)
            # y = 0.05*x/torch.max(torch.abs(x))
        return y
                
    def forward(self, x, A, B):
        Z = torch.zeros_like(x)
        for j in range(x.shape[0]):
            y = A(x[j,0,:,:])
            y = self.Phi(y)
            z = B(y)
            Z[j,0,:,:] = z
        return Z
