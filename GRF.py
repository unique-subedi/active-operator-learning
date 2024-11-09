import numpy as np
from scipy.fftpack import dctn


def GRF(alpha, beta, gamma, N):
    # Random variables in KL expansion
    xi = np.random.randn(N, N)

    K1, K2 = np.meshgrid(np.arange(N), np.arange(N))

    # Define the (square root of) eigenvalues of the covariance operator
    coef = alpha**(1/2) *(4*np.pi**2 * (K1**2 + K2**2) + beta)**(-gamma / 2)

    # Construct the KL coefficients
    L = N * coef * xi
    
    #to make sure that the random field is mean 0
    L[0, 0] = 0

    return idctn(L, type =2)
