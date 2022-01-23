"""
Autograd and OLS in PyTorch
@author: zhang 
"""
# import packages
import numpy as np
import torch, torchvision
from matplotlib import pyplot as plt
#%% Example of autograd
# Create tensors.
a = torch.tensor(-2., requires_grad=True)
b = torch.tensor(5., requires_grad=True)
print(a)
print(b)
y = a + 3*b
print(y)
y.backward()
print('dy/da=', a.grad)
print('dy/db=', b.grad)
# Use .numpy() if you want to get rid of the data format of "tensor"
print('dy/db=', b.grad.numpy())


# try something complex
y = a.pow(2) + 1/(3*b)
a.grad.zero_() #Reset the gradient if reuse a because PyTorch accumulates gradients.
b.grad.zero_()
print(a.grad)
print(b.grad)
y.backward()
print('dy/da=', a.grad)
print('dy/db=', b.grad)

#%% OLS by PyTorch autograd
# simulate design matrix 
n = 200
p = 5
x = torch.randn([n,p])
x[:,0] = 1.
print(x[0:3])
bet = torch.tensor([-2.,-1.,0.,1.,2.]) # true regression coefficients
y = torch.matmul(x,bet) + torch.randn(x.size(0))

# OLS estimation, bet_OLS = (X'X)^{-1}X'y
bet_OLS = torch.matmul(torch.matmul(x.t(),x).inverse(), torch.matmul(x.t(),y))
print(bet_OLS)

# OLS by gradient descent
bet_gd = torch.zeros([p], requires_grad=True)
print(bet_gd)
param = [bet_gd]
optimizer = torch.optim.Adam(param, lr=0.001) # adaptively adjust step sizes 
losses = []

for _ in range(300):
    yhat = torch.matmul(x,bet_gd)
    mse = (y-yhat).pow(2).sum() #or mse = (y-yhat).pow(2).mean()
    optimizer.zero_grad()
    mse.backward(retain_graph=False)
    optimizer.step()
    losses += [mse.item()]

print('Ground truth: ', bet.numpy())
print('Gradient descent: ', bet_gd.detach().numpy())
plt.plot(losses)


#####################################################################
#%% OLS by SGD
# simulate data and make a data loader to sample mini-batches
n = 200
p = 5
x = torch.randn([n,p])
x[:,0] = 1.
bet = torch.tensor([-2.,-1.,0.,1.,2.]) # true regression coefficients
y = torch.matmul(x,bet) + torch.randn(x.size(0))
alldata = [(x[i],y[i]) for i in range(x.size(0))]
data_loader = torch.utils.data.DataLoader(dataset=alldata,
                                          batch_size=20, 
                                          shuffle=True)
bet_sgd = torch.zeros([p], requires_grad=True)
print(bet_sgd)                                
param = [bet_sgd]
#optimizer = torch.optim.Adam(param, lr=0.01) # adaptively adjust step sizes 
# or use adaptive gradient: adagrad
optimizer = torch.optim.Adagrad(param, lr=0.1, lr_decay=0.01) # adaptively adjust step sizes. Trye lr_decay=0.05

niter = 30 # total number of iterations
sgd_losses = []
for epoch in range(niter):
    for i, (x, y) in enumerate(data_loader):
        yhat = torch.matmul(x,bet_sgd)
        mse = (y-yhat).pow(2).mean()
        optimizer.zero_grad()
        mse.backward()
        optimizer.step()
        sgd_losses += [mse.item()]

print('Ground truth: ', bet.numpy())
print('Stochastic gradient descent: ', bet_sgd.detach().numpy())
plt.plot(sgd_losses)










 
