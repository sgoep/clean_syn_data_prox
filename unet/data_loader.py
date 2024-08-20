# %%
import numpy as np
import torch


class DataLoader(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, id, constraint, initrecon, which):
        'Initialization'
        self.id         = id
        self.constraint = constraint
        self.initrecon  = initrecon
        self.X          = np.load("data/phantom.npy")
        if self.initrecon:      
            self.Y = np.load(f"data/init_regul_{which}.npy")
        else:
            self.Y = np.load("data/fbp.npy")
      
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.id)
    
    def __getitem__(self, index):
        'Generates one sample of data'          
        
        X = self.X[index, :, :]
        Y = self.Y[index, :, :]
        
        X = X[None,:,:]
        Y = Y[None,:,:]
        
        return X, Y
    
# import matplotlib.pyplot as plt

# X = np.load(".data_htc2022_simulated/phantom.npy")
# Y = np.load(".data_htc2022_simulated/init_regul.npy")
# Y = np.load(".data_htc2022_simulated/fbp.npy")
# index = 1
# plt.figure()
# plt.imshow(X[index,:,:])
# plt.colorbar()
# plt.figure()
# plt.imshow(Y[index,:,:])
# plt.colorbar()
# %%