import numpy as np
from scipy.fftpack import idctn

def gen_cosines(N, k1, k2):
    # Initialize an N x N array with zeros in the frequency domain
    init_array = np.zeros((N, N))
    # Set the appropriate frequency component to 1 (place "spike" at (k1, k2))
    init_array[k1,k2] = 1

    # Perform the inverse DCT to obtain the spatial cosine pattern
    cosines = idctn(init_array, type=2)

    return 1/2*cosines
