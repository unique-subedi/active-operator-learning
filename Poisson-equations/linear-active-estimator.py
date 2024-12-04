import sys
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.fftpack import dctn
from scipy.fftpack import idctn


def Laplacian(N):
    N2 = (N-1) * (N-1)
    h=1/N
  
    # Define diagonals for the discrete Laplacian operator
    main_diag = -4 * np.ones(N2)
    side_diag = np.ones(N2 - 1)
    side_diag[np.arange(1, N-1) * (N-1) - 1] = 0  # Adjust for block boundaries
    up_down_diag = np.ones(N2 - (N-1))

    # Create sparse matrix with specified diagonals
    L = (1/h**2)*diags([main_diag, side_diag, side_diag, up_down_diag, up_down_diag],
              [0, -1, 1, -(N-1), (N-1)], format="csc")

    return L

def gen_cosines(N, k1, k2):
    # Initialize an N x N array with zeros in the frequency domain
    init_array = np.zeros((N, N))
    # Place "spike" at (k1, k2))
    init_array[k1,k2] = 1

    # Perform the inverse DCT to obtain the spatial cosine pattern
    cosines = idctn(init_array, type=2)

    return 1/2*cosines



N = 64
L = Laplacian(N)
alpha = 50**2
beta = 1
gamma=2



# Generate active training data
K = 15 # n=K^2 training samples
trainset_active = []
for k1 in range(0,K):
  for k2 in range(0,K):
    f = gen_cosines(N, k1, k2)
    C= spsolve(L, f[1:, 1:].flatten())
    u=np.zeros((N,N))
    u[1:N,1:N]=C.reshape((N-1,N-1))
    trainset_active.append({'x': f,'y': u})


testset = []
for i in range(0,n_test):
    f = GRF(alpha,beta,gamma,N)
    C= spsolve(L, f[1:, 1:].flatten())
    u=np.zeros((N,N))
    u[1:N,1:N]=C.reshape((N-1,N-1))
    testset.append({'x': f,'y': u})



# Compute the mean squared error of the active estimator for the test-loss

losses_active = np.zeros((n_test, len(trainset_active)))
for i in range(0,n_test):
    f,u_true = testset[i]['x'], testset[i]['y']
    u_pred = np.zeros((N,N))
    f_dct = (1/(N**2)*1/2)*dctn(f, type=2)[:K, :K].flatten()
    for j in range(len(trainset_active)):
        u_pred = u_pred +  trainset_active[j]['y'] * f_dct[j]
        losses_active[i,j] = np.square(u_pred - u_true).mean()/np.square(u_true).mean()
linear_active_losses = np.mean(losses_active, axis=0)
plt.plot(range(1, len(linear_active_losses)+1),linear_active_losses, label='Linear (Active)', color='blue', marker='o', markersize=3, linestyle='-')
plt.xlabel('Training sample size')
plt.ylabel('Relative testing error')
plt.legend()
plt.show()
