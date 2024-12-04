import sys
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import inv, spsolve
from scipy.fftpack import dctn
from scipy.fftpack import idctn
from matplotlib import pyplot as plt



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



N = 64
L = Laplacian(N)
alpha = 50**2
beta = 1
gamma=2

K=15

n_train = K**2
trainset_passive = []
for i in range(0,n_train):
    f = GRF(alpha,beta,gamma,N)
    C= spsolve(L, f[1:, 1:].flatten())
    u=np.zeros((N,N))
    u[1:N,1:N]=C.reshape((N-1,N-1))
    trainset_passive.append({'x': f,'y': u})


n_test=100
testset = []
for i in range(0,n_test):
    f = GRF(alpha,beta,gamma,N)
    C= spsolve(L, f[1:, 1:].flatten())
    u=np.zeros((N,N))
    u[1:N,1:N]=C.reshape((N-1,N-1))
    testset.append({'x': f,'y': u})



pseudoinv = np.zeros((N**2,N**2))
for i in range(0,n_train):
    pseudoinv += np.outer(trainset_passive[i]['x'].flatten(), trainset_passive[i]['x'].flatten())
pseudoinv = np.linalg.pinv(pseudoinv)



losses_passive = np.zeros((n_test, len(trainset_passive)))
for i in range(0,n_test):
    f,u_true = testset[i]['x'], testset[i]['y']
    u_pred = np.zeros((N,N))
    score = pseudoinv @ f.flatten()
    score = score.reshape((N,N))
    for j in range(len(trainset_passive)):
        c = np.sum(trainset_passive[j]['x'] * score)
        u_pred = u_pred + c *  trainset_passive[j]['y']
        losses_passive[i,j] = np.square(u_pred - u_true).mean()/np.square(u_true).mean()


linear_passive_losses = np.mean(losses_passive, axis=0)
plt.plot(range(1, len(linear_passive_losses)+1),linear_passive_losses, label='Linear (Passive)', color='orange', marker='o', markersize=3, linestyle='-')
plt.xlabel('Training sample size')
plt.ylabel('Relative testing error')
plt.legend()
plt.show()



