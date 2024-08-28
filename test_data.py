import matplotlib.pyplot as plt
import numpy as np
import torch


initial_regularization_method = "data"

print(f"Loading {initial_regularization_method} ...")

index = 1366
x = np.load("data/phantom.npy")[index, :, :]
y = np.load("data/fbp.npy")[index, :, :]
z = np.load("data/init_regul_tv.npy")[index, :, :]

print(f"Plotting {initial_regularization_method} ...")

plt.figure()
plt.imshow(x)
plt.colorbar()
plt.savefig(f"plots/{initial_regularization_method}_phantom.png")

plt.figure()
plt.imshow(y)
plt.colorbar()
plt.savefig(f"plots/{initial_regularization_method}_fbp.png")

plt.figure()
plt.imshow(z)
plt.colorbar()
plt.savefig(f"plots/{initial_regularization_method}_init_regul.png")

print(f"Finished {initial_regularization_method} ...")

print("Finished.")
