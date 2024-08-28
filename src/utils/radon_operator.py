"""
    The original code was taken from
    https://github.com/matteo-ronchetti/torch-radon.
"""

import numpy as np
import torch
import torch.nn.functional as F

try:
    import scipy.fft

    fftmodule = scipy.fft
except ImportError:
    import numpy.fft

    fftmodule = numpy.fft


# def ram_lak_filter(sinogram):
#     num_projections, num_detectors = sinogram.shape
#     freqs = torch.fft.fftfreq(num_detectors).reshape(1, -1)
#     filter = torch.abs(freqs).to(config.device)
#     filtered_sinogram = torch.zeros_like(sinogram).to(config.device)
#     for i in range(num_projections):
#         projection_fft = torch.fft.fft(sinogram[i, :])
#         filtered_projection_fft = projection_fft * filter
#         filtered_projection = torch.fft.ifft(filtered_projection_fft).real
#         filtered_sinogram[i, :] = filtered_projection *
#                   (2 / torch.sqrt(torch.tensor(num_detectors)))**2
#     return filtered_sinogram

def construct_fourier_filter(size, filter_name):
    """Construct the Fourier filter.

    This computation lessens artifacts and removes a small bias as
    explained in [1], Chap 3. Equation 61.

    Parameters
    ----------
    size: int
        filter size. Must be even.
    filter_name: str
        Filter used in frequency domain filtering. Filters available:
        ram-lak (ramp), shepp-logan, cosine, hamming, hann.

    Returns
    -------
    fourier_filter: ndarray
        The computed Fourier filter.

    References
    ----------
    .. [1] AC Kak, M Slaney, "Principles of Computerized Tomographic
            Imaging", IEEE Press 1988.

    """
    filter_name = filter_name.lower()

    n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=int),
                        np.arange(size / 2 - 1, 0, -2, dtype=int)))
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2

    # Computing the ramp filter from the fourier transform of its
    # frequency domain representation lessens artifacts and removes a
    # small bias as explained in [1], Chap 3. Equation 61
    fourier_filter = 2 * np.real(fftmodule.fft(f))  # ramp filter
    if filter_name == "ramp" or filter_name == "ram-lak":
        pass
    elif filter_name == "shepp-logan":
        # Start from first element to avoid divide by zero
        omega = np.pi * fftmodule.fftfreq(size)[1:]
        fourier_filter[1:] *= np.sin(omega) / omega
    elif filter_name == "cosine":
        freq = np.linspace(0, np.pi, size, endpoint=False)
        cosine_filter = fftmodule.fftshift(np.sin(freq))
        fourier_filter *= cosine_filter
    elif filter_name == "hamming":
        fourier_filter *= fftmodule.fftshift(np.hamming(size))
    elif filter_name == "hann":
        fourier_filter *= fftmodule.fftshift(np.hanning(size))
    else:
        print(
            f"[TorchRadon] Error, unknown filter type '{filter_name}', "
            "available filters are: 'ramp', 'shepp-logan', 'cosine', "
            "'hamming', 'hann'"
        )

    return fourier_filter


def filter_sinogram(sinogram, filter_name="ramp", fourier_filters=None):
    # Pad sinogram to improve accuracy
    # sinogram = sinogram[None,:,:,None]
    # size = sinogram.size(2)
    # n_angles = sinogram.size(1)

    n_angles, size = sinogram.shape

    padded_size = max(
        64, int(2 ** torch.ceil(torch.log2(torch.tensor(2 * size))))
    )
    pad = padded_size - size

    padded_sinogram = F.pad(sinogram.float(), (0, pad))

    # Perform FFT on the padded sinogram
    sino_fft = torch.fft.fft(padded_sinogram, dim=-1)

    # Get filter and apply
    f = construct_fourier_filter(padded_size, filter_name)
    f = torch.Tensor(f).to("cuda")

    filtered_sino_fft = sino_fft * f

    # Perform inverse FFT
    filtered_sinogram = torch.fft.ifft(filtered_sino_fft, dim=-1).real

    # print(filtered_sinogram.shape)
    # Pad removal and rescaling
    filtered_sinogram = (
        filtered_sinogram[:, :-pad] *
        (torch.pi / (2 * n_angles))
    )

    return filtered_sinogram.to(dtype=sinogram.dtype)


if __name__ == "__main__":
    print("Test functionality of Radon operator.")
    import h5py
    import torch_radon
    import matplotlib.pyplot as plt

    print("Loading image.")
    device = "cuda"
    index = 1366
    images = (h5py.File("data/randshepp.mat")["data"][:]).transpose([-1, 0, 1])
    phantom = images[index, :, :]
    phantom = phantom/np.max(phantom)
    x = torch.Tensor(phantom).to(device)
    Nal = 180
    angles = np.linspace(0, np.pi, Nal, endpoint=False)
    NUM_ANGLES = len(angles)

    print("Create operator and data.")
    radon = torch_radon.Radon(
        128, angles, det_count=128, det_spacing=1, clip_to_circle=True
    )

    sinogram = radon.forward(x)
    noise = torch.randn(*sinogram.shape).to(device)
    sinogram += 0.03*torch.max(torch.abs(sinogram))*noise

    print("Filtering and backprojection.")
    sinogram = filter_sinogram(sinogram)
    fbp = radon.backward(sinogram)

    print("Saving image.")
    plt.figure()
    plt.imshow(fbp.cpu().numpy())
    plt.colorbar()
    plt.savefig("bp.png")

    print("Finished.")
