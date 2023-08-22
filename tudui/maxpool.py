import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Zowo(nn.Module):
    def __init__(self):
        super(Zowo,self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride=3,padding=0,dilation=1,ceil_mode=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False)
    def forward(self,x):
        x = self.maxpool2(x)
        return x
zowo = Zowo()
dataset = torchvision.datasets.CIFAR10("conv/data",train=False,download=False
                                       ,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64)
# x = torch.tensor([[1,2,0,3,1],
#                   [0,1,2,3,1],
#                   [1,2,1,0,0],
#                   [5,2,3,1,1],
#                   [2,1,0,1,1]],dtype=torch.float32)
# x = torch.reshape(x,[-1,1,5,5])
# y = zowo(x)
# print(x.shape)
# print(y)
writer = SummaryWriter("log/maxpool")
step = 0
for data in dataloader:
    imgs, targets = data
    output = zowo(imgs)

    writer.add_images("output",output,step)
    writer.add_images("input", imgs, step)

    step += 1
