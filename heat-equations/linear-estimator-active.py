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

def gen_cosines(N, k1, k2):
    # Initialize an N x N array with zeros in the frequency domain
    init_array = np.zeros((N, N))
    #place "spike" at (k1, k2)
    init_array[k1,k2] = 1

    # Perform the inverse DCT to obtain the spatial cosine pattern
    cosines = idctn(init_array, type=2)

    return 1/2*cosines


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



# Generate active training data
K = 15 # n=K^2 training samples
trainset_active = []
for k1 in range(0,K):
  for k2 in range(0,K):
    u_0 = gen_cosines(N, k1, k2)
    u = solver(L, u_0, Nt, nu)
    trainset_active.append({'x':  u_0,'y': u})


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
losses_active = np.zeros((n_test, len(trainset_active)))
for i in range(0,n_test):
    u_0,u_true = testset[i]['x'], testset[i]['y']
    u_pred = np.zeros((N,N))
    u0_dct = (1/(N**2)*1/2)*dctn(u_0, type=2)[:K, :K].flatten()
    for j in range(len(trainset_active)):
        u_pred = u_pred +  trainset_active[j]['y'] * u0_dct[j]
        losses_active[i,j] = np.square(u_pred - u_true).mean()/np.square(u_true).mean()
    
linear_active_losses = np.mean(losses_active, axis=0)
plt.plot(range(1, len(linear_active_losses)+1),linear_active_losses, label='Linear (Active)', color='blue', marker='o', markersize=3, linestyle='-')
plt.xlabel('Training sample size')
plt.ylabel('Relative testing error')
plt.legend()
plt.show()
