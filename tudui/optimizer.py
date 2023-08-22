import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Linear, Sequential, Flatten
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

darasets = torchvision.datasets.CIFAR10('./conv/data',download=False,train=False,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(darasets,batch_size=1)
class Zowo(nn.Module):
    def __init__(self):
        super(Zowo, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, input):
        output = self.model1(input)
        return output


zowo = Zowo()
print(zowo)


optim = torch.optim.SGD(zowo.parameters(),lr=0.01)
loss = nn.CrossEntropyLoss()

for epech in range(0,20):
    running_loss = 0.00
    for data in dataloader:
        img,target = data
        output = zowo(img)
        loss_result = loss(output,target)
        optim.zero_grad()
        loss_result.backward()
        optim.step()
        running_loss = running_loss+loss_result
    print(running_loss)