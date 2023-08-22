import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader

datasets = torchvision.datasets.CIFAR10('conv/data',download=False,transform=torchvision.transforms.ToTensor(),train=False)
dataloader = DataLoader(datasets, batch_size=64)

class Zowo(nn.Module):
    def __init__(self):
        super(Zowo,self).__init__()
        self.linear1 = nn.Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

zowo = Zowo()
for data in dataloader:
    img, targ = data
    print(img.shape)
    img = torch.reshape(img,(1,1,1,-1))
    print(img.shape)
    output = zowo(img)
    print(output.shape)
    break