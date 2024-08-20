
import numpy as np
import torch
# from radon_operator import ram_lak_filter, get_TorchRadonOperator
from radon_operator import filter_sinogram
from my_shearlet import ShearletTransform
from torch_radon.solvers import cg
import h5py
from total_variation import tv
import torch_radon

def shrink(a, b):
    return (torch.abs(a) - b).clamp_min(0) * torch.sgn(a)


print("Data generation.")
device = "cuda"
# N = 128
# Nal = 80
# angles = np.linspace(-np.pi/3, np.pi/3, Nal, endpoint=False)
# radon = get_TorchRadonOperator(N, angles)
N = 128
Nal = 80
angles = np.linspace(-np.pi/3, np.pi/3, Nal, endpoint=False)
radon = torch_radon.Radon(128, angles, det_count=128, det_spacing=1, clip_to_circle=True)
n_scales = 3
Num = 1500
norm2 = 0

images = (h5py.File('randshepp.mat')['data'][:]).transpose([-1,0,1])

phantom_all = np.zeros([Num, N, N])
fbp_all = np.zeros([Num, N, N])
init_regul_all = np.zeros([Num, N, N])

shearlet = ShearletTransform(N, N, [0.5]*n_scales)

print("Initialized, starting loop.")
for index in range(Num):
    np.random.seed(index)
    phantom = images[index,:,:]
    phantom = phantom/np.max(phantom)
    # phantom *= 0.006
    phantom_all[index, :, :] = phantom
    x = torch.Tensor(phantom).to(device)
    
    sinogram = radon.forward(x)
    noise = torch.randn(*sinogram.shape).to(device)
    sinogram += 0.03*torch.max(torch.abs(sinogram))*noise
    
    # filtered_sino = radon.filter_sinogram(sinogram[None,None,:,:], "ram-lak")
    filtered_sino = filter_sinogram(sinogram)
    fbp = radon.backprojection(filtered_sino)
    fbp_all[index, :, :] = fbp.cpu().numpy()

    if index < 600:
        norm2 += np.linalg.norm(noise.cpu().numpy())/600
    
    # # ADMM
    # bp = radon.backward(sinogram)
    # sc = shearlet.forward(bp)
    
    # p_0 = 0.0001
    # p_1 = 0.005

    # u_1 = torch.zeros_like(sc) 
    # z_1 = torch.zeros_like(sc)
    # u_2 = torch.zeros_like(bp) 
    # z_2 = torch.zeros_like(bp)
    
    # ground_truth = x.clone()
    # num_iterations = 50
    # f = torch.zeros_like(bp)
    # for i in range(num_iterations):
    #     cg_y = p_0 * bp + p_1 * shearlet.backward(z_1 - u_1) + (z_2 - u_2)
    #     f = cg(lambda x: p_0 * radon.backward(radon.forward(x)) + (1 + p_1) * x, f.clone(), cg_y, max_iter=50)
    #     sh_f = shearlet.forward(f)
    #     z_1 = shrink(sh_f + u_1, p_0 / p_1) 
    #     z_2 = (f + u_2).clamp_min(0)
    #     u_1 = u_1 + sh_f - z_1 
    #     u_2 = u_2 + f - z_2
    
    x0 = torch.zeros([128, 128]).to("cuda")
    L = 400
    Niter = 500
    alpha = 0.04
    f = tv(x0, radon, sinogram, alpha, L, Niter, f=None, print_flag=False)
    
    init_regul_all[index, :, :] = f.cpu().numpy()
    
    if np.mod(index+1, 100) == 0:
        print(f'{index+1}/{Num}')

np.save('data/norm2', norm2)
np.save('data/phantom', phantom_all)
np.save('data/fbp', fbp_all)
np.save(f"data/init_regul_tv", init_regul_all)

print("Finished.")
# %%

# import numpy as np
# import matplotlib.pyplot as plt

# i = 1
# I = np.load('data_htc2022_simulated/images.npy')[i,:,:]
# x = np.load('data_htc2022_simulated/phantom.npy')[i,:,:]
# y = np.load('data_htc2022_simulated/fbp.npy')[i,:,:]
# z = np.load('data_htc2022_simulated/init_regul.npy')[i,:,:]

# plt.figure()
# plt.imshow(I)
# plt.colorbar()

# plt.figure()
# plt.imshow(x)
# plt.colorbar()

# plt.figure()
# plt.imshow(y)
# plt.colorbar()

# plt.figure()
# plt.imshow(z)
# plt.colorbar()

# # %%
# import scipy.io
# import matplotlib.pyplot as plt
# ct_data = scipy.io.loadmat(f'data_htc2022/htc2022_test_data/htc2022_01a_recon_fbp.mat')
# fbp = ct_data["reconFullFbp"]
# plt.figure()
# plt.imshow(fbp)
# plt.colorbar()

