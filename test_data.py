import matplotlib.pyplot as plt
import numpy as np
import torch

which = "data"

print(f"Loading {which} ...")

index = 1366
x = np.load(f"data/phantom.npy")[index,:,:]
y = np.load(f"data/fbp.npy")[index,:,:]
z = np.load(f"data/init_regul_tv.npy")[index,:,:]

print(f"Plotting {which} ...")

plt.figure()
plt.imshow(x)
plt.colorbar()
plt.savefig(f"plots/{which}_phantom.png")

plt.figure()
plt.imshow(y)
plt.colorbar()
plt.savefig(f"plots/{which}_fbp.png")

plt.figure()
plt.imshow(z)
plt.colorbar()
plt.savefig(f"plots/{which}_init_regul.png")

print(f"Finished {which} ...")

print("Finished.")