import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dctn
from scipy.fftpack import idctn
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import inv, spsolve



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

    # Define the grid points
    K1, K2 = np.meshgrid(np.arange(N), np.arange(N))

    # Define the (square root of) eigenvalues of the covariance operator
    coef = alpha**(1/2) *(4*np.pi**2 * (K1**2 + K2**2) + beta)**(-gamma / 2)

    # Construct the KL coefficients
    L = N * coef * xi

    #to make sure that the random field is mean 0
    L[0, 0] = 0

    return idctn(L, type =2)


def solver(L, u_0, Nt, nu):
  u_t =  u_0[1:, 1:].flatten()
  for i in range(Nt):
      u_t = u_t + dt * nu*L.dot(u_t)
  u= np.zeros((N,N))
  u[1:N,1:N]=u_t.reshape((N-1,N-1))
  return u


N = 64
L = Laplacian(N)
alpha = 1
beta = 1
gamma=1.5

Lx = 1  # Length of the rod in x-direction
Ly = 1  # Length of the rod in y-direction
T = 1.0    # Total time for simulation
Nt = 1000  # Number of time steps
nu = 0.01  # Thermal diffusivity

dx = Lx / (N-1)  # Spatial step in x-direction
dy = Ly / (N-1)  # Spatial step in y-direction
dt = T / Nt         # Time step



K=15
# Generate Passive Training Samples
n_train = K**2
trainset_passive = []
for i in range(0,n_train):
    u_0= GRF(alpha,beta,gamma,N)
    u = solver(L, u_0, Nt, nu)
    trainset_passive.append({'x':  u_0,'y': u})


# Generate Test Samples
n_test = 100

testset = []
for i in range(0,n_test):
    u_0 = GRF(alpha,beta,gamma,N)
    u = solver(L, u_0, Nt, nu)
    testset.append({'x':  u_0,'y': u})

# Mean squared error on the test set
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
