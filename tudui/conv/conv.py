import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("data",train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=4)

class Zowo(nn.Module):
    def __init__(self):
        super(Zowo,self).__init__()
        self.conv1 = Conv2d(in_channels=3,out_channels=6,stride=1,padding=0,kernel_size=3)
    def forward(self,x):
        x = self.conv1(x)
        return x

zowo = Zowo()

writer = SummaryWriter("logs")
tag = 0
for data in dataloader:
    imgs, targets = data
    output = zowo(imgs)
    print(imgs.shape)
    print(output.shape)
    writer.add_images("input", imgs, tag)
    output = torch.reshape(output,[-1,3,30,30])
    writer.add_images("output",output, tag)
    tag += 1
