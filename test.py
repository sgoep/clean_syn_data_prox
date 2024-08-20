import os

import numpy as np
import pandas as pd
import torch
from pytorch_wavelets import DWTForward, DWTInverse
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from radon_operator import filter_sinogram#get_TorchRadonOperator, ram_lak_filter

from unet.data_loader import DataLoader
from unet.unet import UNet
from visualization import visualization_with_zoom
import matplotlib.pyplot as plt
from config import config


print("Start testing.")

def my_mse(f, g):
    return np.linalg.norm(f-g)/np.linalg.norm(g)

# index = 800
index = 1366
# index = 1

device = "cuda"
errors = {}
zoom = True
colorbar = False
# constraint = True
# initrecon  = True

# Get FBP reconstruction
D = DataLoader(index, False, False, config.which)
X, Y = D[index]
visualization_with_zoom(X.squeeze(), zoom, colorbar, f'results/ground_truth.pdf')

Y = Y.squeeze()
errors["fbp"] = {
    "MSE": mse(X.flatten(), Y.flatten()),
    # "MSE": my_mse(Y, X),
    "SSIM": ssim(X.flatten(), Y.flatten(), data_range=Y.max() - Y.min()),
    "PSNR": psnr(X.flatten(), Y.flatten(), data_range=Y.max() - Y.min())
}
visualization_with_zoom(Y, zoom, colorbar, f'results/fbp.pdf')

# Get TV reconstruction
D = DataLoader(index, False, True, config.which)
X, Y = D[index]

Y = Y.squeeze()
errors[f"{config.which}"] = {
    "MSE": mse(X.flatten(), Y.flatten()),
    # "MSE": my_mse(Y, X),
    "SSIM": ssim(X.flatten(), Y.flatten(), data_range=Y.max() - Y.min()),
    "PSNR": psnr(X.flatten(), Y.flatten(), data_range=Y.max() - Y.min())
}
visualization_with_zoom(Y, zoom, colorbar, f'results/{config.which}.pdf')
# for constraint in [True, False]:
    # for initrecon in [True, False]:

        # model_name = f'data_prox_network_constraint_{str(constraint)}_init_regul_{str(initrecon)}'

for model_name in ["fbp_resnet", f"{config.which}_resnet", "fbp_nsn", f"{config.which}_nsn", f"{config.which}_dpnsn"]:

    print(f"Model: {model_name}")
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
        # ell2_norm *= 100
        ell2_norm *= config.factor
        model = UNet(1, 1, constraint, null_space_network, norm2=ell2_norm)
    else:
        pass

    # if model_name in ["fbp_resnet", "ell1_resnet"]:
    #     null_space_network = False
    # else:
    #     null_space_network = True

    D = DataLoader(index, constraint, initrecon, config.which)
    X, Y = D[index]

    print(f"Loading.")
    
    model.load_state_dict(torch.load('models/' + model_name, map_location=torch.device(device)))
    model.eval()
    model.to(device)

    Y = torch.Tensor(Y).to(device)
    # print(Y.is_cuda)
    Y = Y.unsqueeze(0)

    print(f"Inference.")
    out, res = model(Y)
    out = out.cpu().detach().numpy().squeeze()
    out = out.astype('float64')
    res = res.cpu().detach().numpy().squeeze()
    res = res.astype('float64')

    plt.figure()
    plt.imshow(res)
    plt.colorbar()
    plt.savefig(f'results/res_{model_name}.pdf')

    # print(f'MSE:  {mse(X.flatten(), out.flatten())}')
    # print(f'MSE:  {my_mse(out, X)}')
    print(f'MSE:  {mse(out.flatten(), X.flatten())}')
    print(f'SSIM: {ssim(X.flatten(), out.flatten(), data_range=out.max() - out.min())}')
    print(f'PSNR: {psnr(X.flatten(), out.flatten(), data_range=out.max() - out.min())}')

    errors[model_name] = {
        "MSE": mse(X.flatten(), out.flatten()),
        # "MSE": my_mse(out, X), 
        "SSIM": ssim(X.flatten(), out.flatten(), data_range=out.max() - out.min()),
        "PSNR": psnr(X.flatten(), out.flatten(), data_range=out.max() - out.min())
    }
    
    print(f"Plotting.")    
    visualization_with_zoom(out, zoom, colorbar, f'results/{model_name}.pdf')

error_table = pd.DataFrame.from_dict(errors).transpose()
error_table.to_csv(f"results/{config.which}_error_table.csv")
print(f"Finished.")    

# # %% REAL DATA
# from my_shearlet import ShearletTransform

# print("Real Data.")
# _, sino, angles, ground_truth, limited_fbp = load_matrix_and_data(sample="01a")
# Aop = get_TorchRadonOperator(512, angles)
# A = lambda x: Aop.forward(x)
# B = lambda y: Aop.backward(y)

# # level = 4
# # P = DWTForward(J=level, mode='zero', wave='db1').cuda()
# # Q = DWTInverse(mode='zero', wave='db1').cuda()

# grount_truth = torch.Tensor(ground_truth).to(device)
# g = torch.Tensor(sino).to(device)

# L = 80000
# Niter = 1000
# N = 512
# # x0 = P(torch.zeros([1, 1, N, N], device=device))
# # solver = FISTA(alpha = 1, L = L, device = device)
# # crec = solver.recon(x0, A, B, P, Q, g, Niter=Niter, ground_truth=None)
# # rec = Q(crec)
# SH = ShearletTransform(N, N, [0.5]*4, cache=None)
# P = lambda x: SH.forward(x)
# Q = lambda y: SH.backward(y)
# x0 = P(torch.zeros([N, N], device=device))
# solver = FISTA_SHEARLET(alpha = 0.05*0.0006, L = 100000, device = device)
# crec = solver.recon(x0, A, B, P, Q, g, Niter = 150, ground_truth=f)
# rec = Q(crec)
# visualization_with_zoom(rec.squeeze().cpu().numpy(), zoom, colorbar, f'results/real_ell1.pdf')
# visualization_with_zoom(ground_truth, zoom, colorbar, f'results/real_ground_truth.pdf')
# visualization_with_zoom(limited_fbp, zoom, colorbar, f'results/real_fbp.pdf')


# for model_name in ["fbp_resnet", "ell1_resnet", "fbp_nsn", "ell1_nsn", "ell1_dpnsn"]:
#     X = torch.Tensor(ground_truth).to(device)    
#     print(f"Model: {model_name}")
#     if model_name in ["fbp_resnet", "fbp_nsn"]:
#         constraint = False
#         initrecon  = False
#         Y = torch.Tensor(limited_fbp).to(device).squeeze()[None,:,:]
#     elif model_name in ["ell1_resnet", "ell1_nsn"]:
#         constraint = False
#         initrecon  = True
#         Y = rec.squeeze()[None,:,:]
#     elif model_name == "ell1_dpnsn":
#         constraint = True
#         initrecon  = True
#         Y = rec.squeeze()[None,:,:]
#     else:
#         pass

#     if model_name in ["fbp_resnet", "ell1_resnet"]:
#         null_space_network = False
#     else:
#         null_space_network = True

#     print(f"Loading.")
#     model = UNet(1, 1, constraint, null_space_network)
#     model.load_state_dict(torch.load('models/' + model_name, map_location=torch.device(device)))
#     model.eval()
#     model.to(device)

#     Y = torch.Tensor(Y).to(device)
#     # print(Y.is_cuda)
#     Y = Y.unsqueeze(0)

#     print(f"Inference.")
#     out, res = model(Y)
#     out = out.cpu().detach().numpy().squeeze()
#     out = out.astype('float64')
#     res = res.cpu().detach().numpy().squeeze()
#     res = res.astype('float64')

#     X = X.cpu().numpy().astype('float64')    
#     print(f'MSE:  {mse(X.flatten(), out.flatten())}')
#     print(f'SSIM: {ssim(X.flatten(), out.flatten(), data_range=out.max() - out.min())}')
#     print(f'PSNR: {psnr(X.flatten(), out.flatten(), data_range=out.max() - out.min())}')

#     errors[model_name] = {
#         "MSE": mse(X.flatten(), out.flatten()),
#         "SSIM": ssim(X.flatten(), out.flatten(), data_range=out.max() - out.min()),
#         "PSNR": psnr(X.flatten(), out.flatten(), data_range=out.max() - out.min())
#     }
    
#     print(f"Plotting.")    
#     visualization_with_zoom(out, zoom, colorbar, f'results/real_{model_name}.pdf')

# error_table = pd.DataFrame.from_dict(errors).transpose()
# error_table.to_csv("results/real_error_table.csv")
# print(f"Finished.")