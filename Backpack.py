# Execute this code block to install dependencies when running on colab
try:
    import torch
except:
    from os.path import exists
    from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
    platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
    cuda_output = !ldconfig -p|grep cudart.so|sed -e 's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/'
    accelerator = cuda_output[0] if exists('/dev/nvidia0') else 'cpu'

    !pip install -q http://download.pytorch.org/whl/{accelerator}/torch-1.0.0-{platform}-linux_x86_64.whl torchvision

try: 
    import torchbearer
except:
    !pip install torchbearer

pip install backpack-for-pytorch

# automatically reload external modules if they change
%load_ext autoreload
%autoreload 2

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchbearer
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from backpack import backpack, extend
from backpack.extensions import DiagGGNMC,DiagGGNExact
from backpack.extensions import KFAC, KFLR, KFRA
import matplotlib.pyplot as plt
import time

EPOCH=3

BATCH_SIZE = 128
LEARNING_RATE=0.001
STEP_SIZE = 0.001
DAMPING = 1.0
MAX_ITER = 200
PRINT_EVERY = 50
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


seed = 7
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# convert each image to tensor format
transform = transforms.Compose([
    transforms.ToTensor()  # convert to tensor
])

# load data
trainset = MNIST(".", train=True, download=True, transform=transform)
testset = MNIST(".", train=False, download=True, transform=transform)

# create data loaders
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Sequential(  
      nn.Conv2d(1, 16, 5, 1, 2),      
      nn.ReLU(),    
      nn.MaxPool2d(kernel_size=2),    
    )
    self.conv2 = nn.Sequential( 
      nn.Conv2d(16, 32, 5, 1, 2),  
      nn.ReLU(),  
      nn.MaxPool2d(2),  
    )
    self.out = nn.Linear(32 * 7 * 7, 10)   

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.view(x.size(0), -1)   
    output = self.out(x)
    return output

loss_function = torch.nn.CrossEntropyLoss().to(DEVICE)

def get_accuracy(output, targets):
    predictions = output.argmax(dim=1, keepdim=True).view_as(targets)
    return predictions.eq(targets).float().mean().item()

class DiagGGNOptimizer(torch.optim.Optimizer):
  def __init__(self, parameters, step_size, damping):
    super().__init__(parameters, dict(step_size=step_size, damping=damping))

  def step(self):
    for group in self.param_groups:
      for p in group["params"]:
        step_direction = p.grad / (p.diag_ggn_mc + group["damping"])
        p.data.add_(-group["step_size"], step_direction)

class DiagGGNEOptimizer(torch.optim.Optimizer):
  def __init__(self, parameters, step_size, damping):
    super().__init__(parameters, dict(step_size=step_size, damping=damping))

  def step(self):
    for group in self.param_groups:
      for p in group["params"]:
        step_direction = p.grad / (p.diag_ggn_exact + group["damping"])
        p.data.add_(-group["step_size"], step_direction)






model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 20, 5, 1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2, 2),
    torch.nn.Conv2d(20, 50, 5, 1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2, 2),
    torch.nn.Flatten(),
    torch.nn.Linear(4 * 4 * 50, 500),
    torch.nn.ReLU(),
    torch.nn.Linear(500, 10),
).to(DEVICE)

net_SGD= model
opt_SGD= torch.optim.SGD(net_SGD.parameters(), lr=LEARNING_RATE)


SGD_l_his=[]
SGD_a_his=[]
start = time.time()
for epoch in range(EPOCH):
  print('Epoch:', epoch)
  for step, (b_x, b_y) in enumerate(trainloader):
    b_x, b_y = b_x.to(DEVICE), b_y.to(DEVICE)
    output = net_SGD(b_x)              
    loss = loss_function(output, b_y) 

    loss.backward()

    opt_SGD.step()   

    SGD_a_his.append(get_accuracy(output, b_y))            
    SGD_l_his.append(loss.data.numpy()) 

end = time.time()
SGD_t=end-start

fig = plt.figure()
axes = [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)]

axes[0].plot(SGD_l_his)
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")

axes[1].plot(SGD_a_his)
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")

model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 20, 5, 1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2, 2),
    torch.nn.Conv2d(20, 50, 5, 1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2, 2),
    torch.nn.Flatten(),
    torch.nn.Linear(4 * 4 * 50, 500),
    torch.nn.ReLU(),
    torch.nn.Linear(500, 10),
).to(DEVICE)

net_Adam= model
opt_Adam=torch.optim.Adam(net_Adam.parameters(), lr=0.0001, betas=(0.9, 0.99))



Adam_l_his=[]
Adam_a_his=[]

start = time.time()
for epoch in range(EPOCH):
  print('Epoch:', epoch)
  for step, (b_x, b_y) in enumerate(trainloader):
    b_x, b_y = b_x.to(DEVICE), b_y.to(DEVICE)
    output = net_Adam(b_x)              
    loss = loss_function(output, b_y) 

    loss.backward()

    opt_Adam.step()   

    Adam_a_his.append(get_accuracy(output, b_y))            
    Adam_l_his.append(loss.data.numpy()) 

end = time.time()
Adam_t=end-start

fig = plt.figure()
axes = [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)]

axes[0].plot(Adam_l_his)
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")

axes[1].plot(Adam_a_his)
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")


model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 20, 5, 1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2, 2),
    torch.nn.Conv2d(20, 50, 5, 1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2, 2),
    torch.nn.Flatten(),
    torch.nn.Linear(4 * 4 * 50, 500),
    torch.nn.ReLU(),
    torch.nn.Linear(500, 10),
).to(DEVICE)

net_DiagGGN= model
extend(net_DiagGGN)
extend(loss_function)
opt_DiagGGN= DiagGGNOptimizer(model.parameters(), step_size=STEP_SIZE, damping=DAMPING)
GGN_l_his=[]
GGN_a_his=[]

start=time.time()

for epoch in range(EPOCH):
  print('Epoch:', epoch)
  for step, (b_x, b_y) in enumerate(trainloader):
    b_x, b_y = b_x.to(DEVICE), b_y.to(DEVICE)
    output = net_DiagGGN(b_x)              
    loss = loss_function(output, b_y) 

    with backpack(DiagGGNMC()):
      loss.backward()

    opt_DiagGGN.step()   

    GGN_a_his.append(get_accuracy(output, b_y))            
    GGN_l_his.append(loss.data.numpy()) 

end = time.time()
GGN_t=end-start

fig = plt.figure()
axes = [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)]

axes[0].plot(GGN_l_his)
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")

axes[1].plot(GGN_a_his)
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")


model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 20, 5, 1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2, 2),
    torch.nn.Conv2d(20, 50, 5, 1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2, 2),
    torch.nn.Flatten(),
    torch.nn.Linear(4 * 4 * 50, 500),
    torch.nn.ReLU(),
    torch.nn.Linear(500, 10),
).to(DEVICE)

net_DiagGGNE= model
extend(net_DiagGGNE)
extend(loss_function)
opt_DiagGGNE= DiagGGNEOptimizer(model.parameters(), step_size=STEP_SIZE, damping=DAMPING)
GGNE_l_his=[]
GGNE_a_his=[]

start=time.time()

for epoch in range(EPOCH):
  print('Epoch:', epoch)
  for step, (b_x, b_y) in enumerate(trainloader):
    b_x, b_y = b_x.to(DEVICE), b_y.to(DEVICE)
    output = net_DiagGGNE(b_x)              
    loss = loss_function(output, b_y) 

    with backpack(DiagGGNExact()):
      loss.backward()

    opt_DiagGGNE.step()   

    GGNE_a_his.append(get_accuracy(output, b_y))            
    GGNE_l_his.append(loss.data.numpy()) 

end = time.time()
GGNE_t=end-start

fig = plt.figure()
axes = [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)]

axes[0].plot(GGNE_l_his)
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")

axes[1].plot(GGNE_a_his)
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")

fig = plt.figure(figsize=(10,4))
axes = [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)]

axes[0].plot(SGD_l_his,label='SGD',alpha=0.8)
axes[0].plot(Adam_l_his,label='Adam',alpha=0.8)
axes[0].plot(GGN_l_his,label='DiagGGN_MC',alpha=0.8)
axes[0].plot(GGNE_l_his,label='DiagGGN',alpha=0.8)
axes[0].set_title("Loss")
axes[0].set_xlabel("Iteration")

axes[1].plot(SGD_a_his,label='SGD',alpha=0.8)
axes[1].plot(Adam_a_his,label='Adam',alpha=0.8)
axes[1].plot(GGN_a_his,label='DiagGGN_MC',alpha=0.8)
axes[1].plot(GGNE_a_his,label='DiagGGN',alpha=0.8)
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Iteration")
plt.legend(loc='best')
plt.savefig('bp_1')



label_list = ['SGD','Adam','DiagGGN_MC','DiagGGN']
opt_list = [SGD_t, Adam_t, GGN_t, GGNE_t]
plt.bar(range(len(opt_list)), opt_list,tick_label=label_list)
plt.legend(loc='best')
plt.savefig('bp_2')
