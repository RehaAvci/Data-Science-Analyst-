'''
Pytorch for classification of MNIST digits
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

dataset = datasets.MNIST(root='/home/zhang/Documents/data',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=False)
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=50, 
                                          shuffle=True)


# look at the data
examples = enumerate(data_loader)
batch_idx, (example_data, example_targets) = next(examples)
example_data.shape

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
fig

# global parameters of data and hidden layer dimensions
x_size = 784
h_dim = 200#20#
y_dim=10
device = torch.device("cpu")

# MLP structure: 784 -> h_dim -> h_dim -> 10
class Net(nn.Module):
    def __init__(self, x_size=x_size , h_dim=h_dim, y_dim=y_dim):
        super(Net, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(x_size, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, y_dim),
        ) 
    def forward(self, x):
        h = self.mlp(x)
        output = F.log_softmax(h, dim=1)
        return output

model = Net().to(device)
# weight_decay specifies L2 penalty
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001) 
num_epochs = 10
losses = []
for epoch in range(num_epochs):
    for i, (x, y) in enumerate(data_loader):
        # Forward pass
        x, y = x.to(device).view(-1, x_size), y 
        optimizer.zero_grad()
        output = model(x)
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
        losses.append(loss)
        if i % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f} \tTraining Accuracy: {:.4f}'.format(
                epoch, i * len(x), len(data_loader.dataset),
                100. * i/ len(data_loader), loss.item(),
                (output.max(1)[1]==y).sum().item()/y.size(0)
                )
            )
plt.plot(losses)      
            
# prediction
testset = datasets.MNIST(root='/home/zhang/Documents/data',
                                     train=False,
                                     transform=transforms.ToTensor(),
                                     download=False)
test_loader = torch.utils.data.DataLoader(dataset=testset,
                                          batch_size=len(testset), 
                                          shuffle=False)
for i, (xt, yt) in enumerate(test_loader):
    xt, yt = xt.to(device).view(-1, x_size), yt
    output = model(xt)
    output = output.max(-1)[1]
    accuracy = (output==yt).sum()/len(yt)
    print('prediction accuracy: {:.4f}'.format(accuracy))
    
    
    