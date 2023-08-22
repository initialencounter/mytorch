from torch import nn
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./conv/data", train=False, download=False,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)


class Zowo(nn.Module):
    def __init__(self):
        super(Zowo, self).__init__()
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output


zowo = Zowo()
writer = SummaryWriter("./log")
step = 0
for data in dataloader:
    img, targ = data
    writer.add_images('input', img, step)
    output = zowo(img)
    writer.add_images('output', output, step)
    step += 1
