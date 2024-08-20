import os

import numpy as np
import pandas as pd
import torch
from pytorch_wavelets import DWTForward, DWTInverse
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from radon_operator import filter_sinogram

from unet.data_loader import DataLoader
from unet.unet import UNet
from visualization import visualization_with_zoom
import matplotlib.pyplot as plt
import torch_radon
from config import config


print("Start testing.")

def my_mse(f, g):
    return np.linalg.norm(f-g)/np.linalg.norm(g)

# index = 800
index = 1366
# index = 1

device = "cuda"
errors = {}
zoom = False
colorbar = True
# constraint = True
# initrecon  = True

error_fbp_mse = 0
error_ell1_mse = 0

error_fbp_psnr = 0
error_ell1_psnr = 0

error_fbp_ssim = 0
error_ell1_ssim = 0

Nal = 80
angles = np.linspace(-np.pi/3, np.pi/3, Nal, endpoint=False)
radon = torch_radon.Radon(128, angles, det_count=128, det_spacing=1, clip_to_circle=True)

for index in range(config.len_train, config.len_train+config.len_test):
    # Get FBP reconstruction
    D = DataLoader(index, False, False, config.which)
    X, Y = D[index]

    Y = Y.squeeze()

    error_fbp_mse += mse(X.flatten(), Y.flatten())/config.len_test
    error_fbp_psnr += psnr(X.flatten(), Y.flatten(), data_range=Y.max() - Y.min())/config.len_test
    error_fbp_ssim += ssim(X.flatten(), Y.flatten(), data_range=Y.max() - Y.min())/config.len_test

    # Get TV reconstruction
    D = DataLoader(index, False, True, config.which)
    X, Y = D[index]

    Y = Y.squeeze()
    
    error_ell1_mse += mse(X.flatten(), Y.flatten())/config.len_test
    error_ell1_psnr += psnr(X.flatten(), Y.flatten(), data_range=Y.max() - Y.min())/config.len_test
    error_ell1_ssim += ssim(X.flatten(), Y.flatten(), data_range=Y.max() - Y.min())/config.len_test

    # for constraint in [True, False]:
        # for initrecon in [True, False]:

            # model_name = f'data_prox_network_constraint_{str(constraint)}_init_regul_{str(initrecon)}'

errors["fbp"] = {
    "MSE": error_fbp_mse,
    # "MSE": my_mse(Y, X),
    "SSIM": error_fbp_ssim,
    "PSNR": error_fbp_psnr
}

errors[f"{config.which}"] = {
    "MSE": error_ell1_mse,
    # "MSE": my_mse(Y, X),
    "SSIM": error_ell1_ssim,
    "PSNR": error_ell1_psnr
}


for model_name in ["fbp_resnet", f"{config.which}_resnet", "fbp_nsn", f"{config.which}_nsn", f"{config.which}_dpnsn"]:
    print(f"Model: {model_name}")
    for index in range(config.len_train, config.len_train+config.len_test):

        if model_name == "fbp_resnet":
            constraint = False
            initrecon  = False
            null_space_network = False
            model = UNet(1, 1, constraint, null_space_network)
        elif model_name == "fbp_nsn":
            constraint = False
            initrecon  = False
            null_space_network = True
            model = UNet(1, 1, constraint, null_space_network)
        elif model_name == f"{config.which}_resnet":
            constraint = False
            initrecon  = True
            null_space_network = False
            model = UNet(1, 1, constraint, null_space_network)
        elif model_name == f"{config.which}_nsn":
            constraint = False
            initrecon  = True
            null_space_network = True
            model = UNet(1, 1, constraint, null_space_network)
        elif model_name == f"{config.which}_dpnsn":
            constraint = True
            initrecon  = True
            null_space_network = True
            ell2_norm = np.load('data/norm2.npy', allow_pickle=True)
            ell2_norm *= config.factor
            model = UNet(1, 1, constraint, null_space_network, norm2=ell2_norm)
        else:
            pass

        D = DataLoader(index, constraint, initrecon, config.which)
        X, Y = D[index]
        
        model.load_state_dict(torch.load('models/' + model_name, map_location=torch.device(device)))
        model.eval()
        model.to(device)

        Y = torch.Tensor(Y).to(device)
        # print(Y.is_cuda)
        Y = Y.unsqueeze(0)

        out, res = model(Y)
        out = out.cpu().detach().numpy().squeeze()
        out = out.astype('float64')
        res = res.cpu().detach().numpy().squeeze()
        res = res.astype('float64')
        
        if index == config.len_train:
            errors[model_name] = {
                "MSE": mse(X.flatten(), out.flatten())/config.len_test,
                # "MSE": my_mse(out, X), 
                "SSIM": ssim(X.flatten(), out.flatten(), data_range=out.max() - out.min())/config.len_test,
                "PSNR": psnr(X.flatten(), out.flatten(), data_range=out.max() - out.min())/config.len_test
            }
        else:
            errors[model_name]["MSE"] += mse(X.flatten(), out.flatten())/config.len_test,
            errors[model_name]["SSIM"] += ssim(X.flatten(), out.flatten(), data_range=out.max() - out.min())/config.len_test,
            errors[model_name]["PSNR"] += psnr(X.flatten(), out.flatten(), data_range=out.max() - out.min())/config.len_test
        
    
error_table = pd.DataFrame.from_dict(errors).transpose()
error_table.to_csv(f"results/{config.which}_error_table_average.csv")
print("Finished.")
