import sys
sys.path.append("data-generation")
import numpy as np
from Laplacian import Laplacian
from GRF import GRF
from cosines import gen_cosines
from scipy.sparse.linalg import spsolve
from scipy.fftpack import dctn
from scipy.fftpack import idctn



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
