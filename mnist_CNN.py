'''
CNN for classification of MNIST digits 
'''
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

train_data = torchvision.datasets.MNIST(
    root = '/home/zhang/Documents/data',
    train = True,                         
    transform = torchvision.transforms.ToTensor(), 
    download = False,            
)
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/home/zhang/Documents/data', train=True, 
                             download=False,
                             transform=torchvision.transforms.ToTensor()),
  batch_size=50, shuffle=True)



examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)
example_data.shape
 
    
# CNN
  
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        ) # output dim = 16*[(28-4)/2]*[(28-4)/2]=16*12*12
        self.conv2 = nn.Sequential(         
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),     
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),                
        ) # output dim = 32*[(12-4)/2]*[(12-4)/2] = 32*4*4
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 4 * 4, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 4 * 4)
        x = x.view(x.size(0), -1)       
        output = F.log_softmax(self.out(x),dim=1)
        return output 


cnn = CNN() 
print(cnn)
# weight_decay specifies L2 penalty
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001, weight_decay=0.001) 
num_epochs = 10
losses = []
for epoch in range(num_epochs):
    for i, (x, y) in enumerate(train_loader):
        #x, y = x.to(device).view(-1, x_size), y 
        optimizer.zero_grad()
        output = cnn(x)
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
        losses.append(loss)
        if i % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f} \tTraining Accuracy: {:.4f}'.format(
                epoch, i * len(x), len(train_loader.dataset),
                100. * i/ len(train_loader), loss.item(),
                (output.max(1)[1]==y).sum().item()/y.size(0)
                )
            )
plt.plot(losses)      
            
# prediction
test_data = torchvision.datasets.MNIST(
    root = '/home/zhang/Documents/data', 
    train = False, 
    transform = torchvision.transforms.ToTensor(),
    download=False
)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/home/zhang/Documents/data', train=False, 
                             download=False,
                             transform=torchvision.transforms.ToTensor()),
  batch_size=10000, shuffle=False)
  
for i, (xt, yt) in enumerate(test_loader):
    output = cnn(xt)
    output = output.max(-1)[1]
    accuracy = (output==yt).sum()/len(yt)
    print('prediction accuracy: {:.4f}'.format(accuracy))
    
    
    