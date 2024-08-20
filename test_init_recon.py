# %%
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch_radon
from pytorch_wavelets import DWTForward, DWTInverse
from scipy import sparse
from skimage.transform import resize

from my_shearlet import ShearletTransform
from radon_operator import (RadonOperator, get_matrix, get_real_matrix,
                            get_TorchRadonOperator, ram_lak_filter)
from recon_algorithms import FISTA, FISTA_SHEARLET, TV

device = "cuda"


def load_matrix(name: str = "system_matrix_A.npz"):
    Amat = sparse.load_npz(name)
    Amat = Amat.tocoo()
    values = Amat.data
    indices = np.vstack((Amat.row, Amat.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = Amat.shape
    Amat = torch.sparse_coo_tensor(i, v, torch.Size(shape), dtype=torch.float, device="cuda")
    return Amat

def load_data(sample: str = "01a"):
    # Load example data
    
    ct_data = scipy.io.loadmat(f'data_htc2022/htc2022_test_data/htc2022_{sample}_limited.mat')
    ct_data = ct_data["CtDataLimited"][0][0]
    sino = ct_data["sinogram"]
    angles = ct_data["parameters"]["angles"][0, 0][0]
    return sino, angles

def load_matrix_and_data(sample: str):
    sino, angles = load_data(sample)
    # matrix = load_matrix(f"system_matrix_A_diff{sample}.npz")
    
    gt_data = scipy.io.loadmat(f'data_htc2022/htc2022_test_data/htc2022_{sample}_recon_fbp.mat')
    ground_truth = gt_data["reconFullFbp"]
    fbp_data = scipy.io.loadmat(f'data_htc2022/htc2022_test_data/htc2022_{sample}_recon_fbp_limited.mat')
    limited_fbp = fbp_data["reconLimitedFbp"]
    return [], sino, angles, ground_truth, limited_fbp

def _pytorch_to_np_coeff(Yl, Yh):
    coeff = [Yl.squeeze()]
    for l in Yh:
        coeff.append(tuple(l.squeeze()))
    return coeff

def _np_to_pytorch_coeff(c):
    Yl = c[0][None, None, :, :]
    Yh = []
    for l in c[1:]:
        m, n = l[0].shape
        t = torch.zeros([1, 1, 3, m, n])
        t[0, 0, 0, :, :] = l[0]
        t[0, 0, 1, :, :] = l[1]
        t[0, 0, 2, :, :] = l[2]
        Yh.append(t)
    return Yl, Yh



if __name__ == "__main__":
    torch.manual_seed(0)

    device = "cuda"

    real_or_syn = "syn"    
    fista_or_tv = "fista_shear"
    # fista_or_tv = "fista"
    # fista_or_tv = "tv"

    N = 512
    Ns = int(np.sqrt(2)*N)
    
    
    _, sino, angles, ground_truth, _ = load_matrix_and_data(sample="01a")
    Aop = get_TorchRadonOperator(N, angles)
    A = lambda x: Aop.forward(x)
    B = lambda y: Aop.backward(y)
    
    sino = torch.Tensor(sino).to(device)
    f = torch.Tensor(ground_truth).to(device)
    if real_or_syn == "syn":
        f = np.load("data_htc2022_simulated/images.npy")[1].astype('float64')
        # f = np.load("data_htc2022_simulated/images_without_blur.npy")[1].astype('float64')
        f /= np.max(f)
        # f *= 0.006
        f = resize(f, (N, N))
        f = torch.Tensor(f).to(device)
        # f = torch.zeros_like(f)
        # f[N//2-N//4:N//2+N//4, N//2-N//4:N//2+N//4] = 1
        sino = A(f)
        
        noise = torch.randn(*sino.shape).to(device)
        # sino += 0.05*torch.max(torch.abs(sino))*noise
        sino += 0.01*torch.max(torch.abs(sino))*noise
    
    plt.figure()
    plt.imshow(sino.cpu().numpy())
    plt.colorbar()
    plt.savefig("init_recon_sino.png")
    g = torch.Tensor(sino).to(device)
    
    plt.figure()
    plt.imshow(f.cpu().numpy())
    plt.colorbar()
    plt.savefig("init_recon_test_ground_truth.png")
    g = torch.Tensor(sino).to(device)
    
    fbp = B(ram_lak_filter(g))
    # fbp = B(Filter(g))

    plt.figure()
    plt.imshow(fbp.cpu().numpy())
    plt.colorbar()
    plt.savefig("init_recon_test_fbp.png")

    print("FBP calculated.")

    # sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor)
    # L = 80000
    # Niter = 5000

    # for alpha in [0, 0.001, 0.01, 0.1, 1]:
    # for alpha in [1]:
    # alpha = 0.01 for syn
    if fista_or_tv == "fista":
        level = 4
        wavelet = "db1"
        P = DWTForward(J=level, mode='zero', wave=wavelet).cuda()
        Q = DWTInverse(mode='zero', wave=wavelet).cuda()

        x0 = P(torch.zeros([1, 1, N, N], device=device))
        solver = FISTA(alpha = 6, L = 80000, device = device)
        crec = solver.recon(x0, A, B, P, Q, g, Niter = 500, ground_truth=f)
        rec = Q(crec)
    elif fista_or_tv == "tv":
        x0 = torch.zeros([1, 1, N, N], device=device)
        solver = TV(alpha = 0.1, L = 300, anisotropic=False, device = device)
        rec = solver.recon(x0, A, B, g, Niter=1000, ground_truth=f)
    elif fista_or_tv == "fista_shear":
        SH = ShearletTransform(N, N, [0.5]*4, cache=None)
        P = lambda x: SH.forward(x)
        Q = lambda y: SH.backward(y)
        x0 = P(torch.zeros([N, N], device=device))

        # solver = FISTA_SHEARLET(alpha = 0.05*0.0006, L = 100000, device = device)
        solver = FISTA_SHEARLET(alpha = 0.0005, L = 80000, device = device)
        crec = solver.recon(x0, A, B, P, Q, g, Niter = 500, ground_truth=f)
        rec = Q(crec)

  
    plt.figure()
    plt.imshow(rec.squeeze().cpu().numpy())
    plt.colorbar()
    plt.savefig("init_recon_test.png")
    print("Finished.")
