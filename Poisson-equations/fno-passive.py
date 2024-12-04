import sys
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.fftpack import dctn
from scipy.fftpack import idctn


from torch.utils.data import DataLoader, Dataset
from neuralop.models import FNO, TFNO
from neuralop.utils import count_model_params
from neuralop import LpLoss
from neuralop.training import Trainer





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



testset = []
for i in range(0,n_test):
    f = GRF(alpha,beta,gamma,N)
    C= spsolve(L, f[1:, 1:].flatten())
    u=np.zeros((N,N))
    u[1:N,1:N]=C.reshape((N-1,N-1))
    testset.append({'x': f,'y': u})



sample_sizes = [i*K for i in range(K+1)]
sample_sizes[0]=1


trainset_passive_torch = []
for s in trainset_passive:
        trainset_passive_torch.append(  {
              'x': torch.tensor(s['x'], dtype = torch.float32).reshape(1,N,N),
              'y': torch.tensor(s['y'],dtype = torch.float32).reshape(1,N,N)
          })


testset_torch = []


for s in testset:
        testset_torch.append(  {
              'x': torch.tensor(s['x'], dtype = torch.float32).reshape(1,N,N),
              'y': torch.tensor(s['y'],dtype = torch.float32).reshape(1,N,N)
          })




FNO_passive_losses = []
for sample_size in sample_sizes:
    print(f"Training with {sample_size} samples...")



    # Create DataLoaders
    trainloader = DataLoader(trainset_passive_torch[:sample_size], batch_size=15, shuffle=True)
    valloaders = {"same_grid": DataLoader(valset_torch, batch_size=20, shuffle=True)}
    testloaders = {'same_grid': DataLoader(testset_torch, batch_size=20, shuffle=True)}


    f_modes = int(N/2)
    model = FNO(n_modes=(f_modes,) * 2, hidden_channels=32, projection_channels=64, num_layers=4,
                in_channels=1, out_channels=1)
    operator = model.to(device)

    optimizer = torch.optim.Adam(operator.parameters(), lr=8e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    l2loss = LpLoss(d=2, p=2, reduce_dims=(0,1))
    train_loss = l2loss
    eval_losses={'l2': l2loss}

    trainer = Trainer(model=operator, n_epochs=200, 
                      device=device,
                      eval_interval=10,
                      log_output=False,
                      use_distributed=False,
                      verbose=True)

    trainer.train(train_loader=trainloader,
                  test_loaders =valloaders,
                  eval_losses=eval_losses,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  regularizer=False,
                  training_loss=train_loss)

    # Evaluation
    testdataset = testloaders['same_grid'].dataset
    test_loss = 0.0
    with torch.no_grad():
        for data in testdataset:
            x = data['x'].to(device)
            y = data['y'].to(device)
            out = operator(x.unsqueeze(0)).to(device)
            test_loss += (torch.mean(torch.square(out - y))/torch.mean(torch.square(y))).item() 

    FNO_passive_losses.append(test_loss / len(testdataset))
    print(f"Test Loss for {sample_size} samples: {FNO_passive_losses[-1]}")



plt.plot(sample_sizes, FNO_passive_losses)
plt.xlabel('Sample size')
plt.ylabel('Test Loss')
plt.title('Test Loss vs. Sample Size')
plt.show()
