from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pywt
import scipy.io
import torch
from skimage.transform import resize

from src.utils.radon_operator import (RadonOperator, RadonOperator_small, get_matrix, get_real_matrix,
                            ram_lak_filter)


def soft(c: List, a: float):
    """
    Apply the soft thresholding operator to each element in a list of tuples.

    Parameters:
    c (List): A list where the first element is a scalar and the subsequent
    elements are tuples of numerical values.
    a (float): The thresholding parameter.

    Returns:
    List: A new list with the same structure as `c`, where each element in the
    tuples has been soft-thresholded.
    """
    # cnew = [c[0]]
    cnew = [np.sign(c[0])*np.maximum(0, abs(c[0])-alpha)]
    for i in c[1:]:
        c_tmp = []
        for j in i:
            c_tmp.append(np.sign(j)*np.maximum(0, abs(j)-alpha))
        cnew.append(tuple(c_tmp))
    return cnew


def my_multiply(c: List, a: float) -> List:
    """
    Multiply each element in a list of tuples by a scalar.

    Parameters:
    c (List): A list where the first element is a scalar and the subsequent
    elements are tuples of numerical values.
    a (float): The scalar multiplier.

    Returns:
    List: A new list with the same structure as `c`, where each element in the
    tuples has been multiplied by `a`.
    """
    h = [c[0]*a]
    for k in c[1:]:
        h.append(tuple(i*a for i in k))
    return h


def my_subtract(c: List, a: List) -> List:
    """
    Subtract corresponding elements of two lists of tuples.

    Parameters:
    c (List): The first list, where the first element is a scalar and the
    subsequent elements are tuples of numerical values.
    a (List): The second list, with the same structure as `c`.

    Returns:
    List: A new list where each element is the result of subtracting
    corresponding elements of `a` from `c`.
    """
    h = [c[0] - a[0]]
    for k, j in zip(c[1:], a[1:]):
        h.append(tuple(np.subtract(k, j)))
    return h


def my_addition(c: List, a: List) -> List:
    """
    Add corresponding elements of two lists of tuples.

    Parameters:
    c (List): The first list, where the first element is a scalar and the
    subsequent elements are tuples of numerical values.
    a (List): The second list, with the same structure as `c`.

    Returns:
    List: A new list where each element is the result of adding corresponding
    elements of `c` and `a`.
    """
    h = [c[0] + a[0]]
    for k, j in zip(c[1:], a[1:]):
        h.append(tuple(np.add(k, j)))
    return h


def fista(x0: np.ndarray,
          A: Callable,
          B: Callable,
          P: Callable,
          Q: Callable,
          data: np.ndarray,
          alpha: float,
          L: float,
          Niter: int,
          ground_truth: np.ndarray,
          print_flag: bool = True) -> np.ndarray:
    """
    Perform the Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) for
    solving linear inverse problems.

    Parameters:
    x0 (np.ndarray): Initial guess for the solution.
    A (Callable): Forward operator.
    B (Callable): Adjoint (transpose) of the forward operator A.
    P (Callable): Analysis operator of the wavelet transform.
    Q (Callable): Synthesis operator of the wavelet transform.
    data (np.ndarray): Observed data.
    alpha (float): Regularization parameter.
    L (float): Lipschitz constant.
    Niter (int): Number of iterations.
    ground_truth (np.ndarray): Ground truth data for error computation.
    print_flag (bool): Flag to control printing of progress.

    Returns:
    np.ndarray: The computed solution after `Niter` iterations.
    """
    xout = x0
    y = x0
    t_step = 1
    er = np.zeros(Niter)

    for k in range(Niter):
        tprev = t_step
        xprev = xout

        q = A(Q(y)) - data
        p = my_multiply(P(B(q)), 1/L)
        xout = soft(my_subtract(y, p), alpha/L)

        t_step = (1 + np.sqrt(1 + 4*tprev**2))/2

        u = my_subtract(xout, xprev)
        v = my_multiply(u, (tprev - 1) / t_step)
        y = my_addition(xout, v)

        if ground_truth is not None:
            difference = np.abs(Q(xout) - ground_truth)
            squared_difference = difference ** 2
            ground_truth_squared = np.abs(ground_truth) ** 2
            denominator = np.sum(ground_truth_squared)
            result = squared_difference / denominator
            er[k] = np.sum(result)
        if print_flag:  # and ((k-1 % 100 == 0) or k==0):
            if (np.mod(k+1, 100) == 0) or (k == 0):
                norm = np.linalg.norm(Q(xout))
                print(
                    f"Synthesis Iteration: {str(k+1)} / {str(Niter)}, "
                    f"Error: {str(er[k])}, Norm: {norm}"
                )
    return xout


if __name__ == "__main__":
    np.random.seed(42)
    level = 4
    wavelet = "db5"

    P = lambda x: pywt.wavedec2(x, wavelet, mode='symmetric', level=level)
    Q = lambda x: pywt.waverec2(x, wavelet, mode='symmetric')

    # c = P(np.zeros([512, 512]))

    # Load example data
    sample = "01a"
    ct_data = scipy.io.loadmat(
        f'data_htc2022/htc2022_test_data/htc2022_{sample}_limited.mat'
    )
    ct_data = ct_data["CtDataLimited"][0][0]
    sino = ct_data["sinogram"]
    angles = ct_data["parameters"]["angles"][0, 0][0]


    N = 64
    Ns = int(np.sqrt(2)*N)
    Nal = 180
    angles = np.linspace(0, np.pi/2, Nal, endpoint=False)
    # Amat = get_matrix(N, Ns, angles*np.pi/180)
    # angles = np.linspace(0, np.pi/2*(1-1/Nal), Nal, endpoint=True)
    # Amat = get_matrix(N, Ns, angles)
    Aop = RadonOperator_small(angles)
    # f = np.load("data_htc2022_simulated/phantom.npy")[1]
    # f = resize(f, (N, N))

    x = np.linspace(-N//2, N//2-1, N)
    X, Y = np.meshgrid(x, x)
    f = np.zeros([N, N])
    # f[N//2-N//4:N//2+N//4,N//2-N//4:N//2+N//4] = 1
    f[X**2 + Y**2 <= 4*N] = 1

    # A = lambda x: np.matmul(Amat, x.reshape(-1, 1)).reshape(181, 560)
    # B = lambda y: np.matmul(Amat.T, y.reshape(-1, 1)).reshape(512, 512)
    # A = lambda x: Amat.dot(x.reshape(-1, 1)).reshape(len(angles), Ns)
    # B = lambda y: Amat.T.dot(y.reshape(-1, 1)).reshape(N, N)
    A = lambda x: Aop.forward(x)
    B = lambda y: Aop.backward(y)

    g = A(f)
    g += 0.0*np.max(np.abs(g))*np.random.randn(*g.shape)

    fbp = B(ram_lak_filter(torch.Tensor(g)).numpy())

    plt.figure()
    plt.imshow(fbp)
    plt.colorbar()

    L = 9000
    Niter = 1000
    alpha = 0.001
    alpha = 0
    x0 = P(np.zeros([N, N]))
    crec = fista(x0, A, B, P, Q, g, alpha, L, Niter, ground_truth=f)
    rec = Q(crec)

    plt.figure()
    plt.imshow(rec)
    plt.colorbar()
    plt.savefig("ell1_test.png")
    # %%
    import torch
    from pytorch_wavelets import DWTForward, DWTInverse
    import numpy as np
    import matplotlib.pyplot as plt

    xfm = DWTForward(J=3, mode='zero', wave='db1')
    f = torch.Tensor(np.load("data_htc2022_simulated/phantom.npy")[1])[None,None,:,:]
    Yl, Yh = xfm(f)
    ifm = DWTInverse(mode='zero', wave='db1')
    Y = ifm((Yl, Yh))

    plt.imshow(Y.squeeze().numpy())
    # %%
