import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_radon.solvers import cg
import h5py

from my_shearlet import ShearletTransform
from radon_operator import get_TorchRadonOperator, ram_lak_filter

device = "cuda"

def my_grad(self, f: torch.Tensor) -> torch.Tensor:
    fx = torch.cat((f[1:,:], f[-1,:].unsqueeze(0)), dim=0) - f
    fy = torch.cat((f[:,1:], f[:,-1].unsqueeze(1)), dim=1) - f
    return fx, fy

def my_div(self, Px: torch.Tensor, Py: torch.Tensor) -> torch.Tensor:
    fx = Px - torch.cat((Px[0,:].unsqueeze(0), Px[0:-1,:]), dim=0)
    fx[0,:] = Px[0,:]
    fx[-1,:] = -Px[-2,:]

    fy = Py - torch.cat((Py[:,0].unsqueeze(1), Py[:,0:-1]), dim=1)
    fy[:,0] = Py[:,0]
    fy[:,-1] = -Py[:,-2]

    return fx + fy

def shrink(a, b):
    return (torch.abs(a) - b).clamp_min(0) * torch.sgn(a)

n_scales = 3
# angles = (np.linspace(0., 100., n_angles, endpoint=False) - 50.0) / 180.0 * np.pi



data = (h5py.File('randshepp.mat')['data'][:]).transpose([-1,0,1])
x = data[0,:,:]
x /= np.max(x)
x = torch.Tensor(x).to(device)
# f *= 0.006

N = x.shape[0]
# x = torch.zeros([N, N]).to(device)
# x[N//2-N//4:N//2+N//4, N//2-N//4:N//2+N//4] = 1

Nal = 120
# angles = np.linspace(0, np.pi/2, Nal, endpoint=False)
angles = np.linspace(-np.pi/3, np.pi/3, Nal, endpoint=False)
radon = get_TorchRadonOperator(N, angles)

shearlet = ShearletTransform(N, N, [0.5]*n_scales)
sinogram = radon.forward(x)
noise = torch.randn(*sinogram.shape).to(device)
sinogram += 0.05*torch.max(torch.abs(sinogram))*noise

bp = radon.backward(sinogram)

plt.figure()
plt.imshow(radon.backward(ram_lak_filter(sinogram)).cpu().numpy())
plt.colorbar()
plt.savefig("admm_torch_fbp.png")

plt.figure()
plt.imshow(sinogram.cpu().numpy())
plt.colorbar()
plt.savefig("admm_torch_sino.png")


sc = shearlet.forward(bp)
p_0 = 0.0001
p_1 = 0.005
w = 2**shearlet.scales / 400
w = w.view(-1, 1, 1).cuda()

u_1 = torch.zeros_like(sc) 
z_1 = torch.zeros_like(sc)
u_2 = torch.zeros_like(bp) 
z_2 = torch.zeros_like(bp)

ground_truth = x.clone()
num_iterations = 50
f = torch.zeros_like(bp)
for i in range(num_iterations):
    cg_y = p_0 * bp + p_1 * shearlet.backward(z_1 - u_1) + (z_2 - u_2)
    # cg_y = p_0 * bp + p_1 * my_div(z_1 - u_1) + (z_2 - u_2)
    f = cg(lambda x: p_0 * radon.backward(radon.forward(x)) + (1 + p_1) * x, f.clone(), cg_y, max_iter=50)
    sh_f = shearlet.forward(f)
    # sh_f = my_grad(f)
    z_1 = shrink(sh_f + u_1, p_0 / p_1 ) 
    z_2 = (f + u_2).clamp_min(0)
    u_1 = u_1 + sh_f - z_1 
    u_2 = u_2 + f - z_2
    
    print(f"{i+1}/{num_iterations}, {torch.linalg.norm(f - ground_truth)}")
    
plt.figure()
plt.imshow(f.cpu().numpy())
plt.colorbar()
plt.savefig("admm_torch.png")

