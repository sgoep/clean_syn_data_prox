import numpy as np
import astra
import torch
import os 
from config import config
import matplotlib.pyplot as plt
import pandas as pd

import torch_radon
from radon_operator import filter_sinogram#, ram_lak_filter
import h5py

device = "cuda"
index = 1366
images = (h5py.File('randshepp.mat')['data'][:]).transpose([-1,0,1])
phantom = images[index,:,:]
phantom = phantom/np.max(phantom)
x = torch.Tensor(phantom).to(device)
Nal = 180
angles = np.linspace(0, np.pi, Nal, endpoint=False)
NUM_ANGLES = len(angles)

radon = torch_radon.Radon(128, angles, det_count=128, det_spacing=1, clip_to_circle=True)

sinogram = radon.forward(x)
noise = torch.randn(*sinogram.shape).to(device)
sinogram += 0.03*torch.max(torch.abs(sinogram))*noise
sinogram = filter_sinogram(sinogram)
# sinogram = ram_lak_filter(sinogram)

fbp = radon.backward(sinogram)
plt.figure()
plt.imshow(fbp.cpu().numpy())
plt.colorbar()
plt.savefig("bp.png")