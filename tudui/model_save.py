import torch
from torch import nn


# train_data = torchvision.datasets.ImageNet('../conv/data')
# vgg16_false = torchvision.models.vgg16(pretrained=False)

# torch.save(vgg16_false,"./vgg16_method1.pth")

# torch.save(vgg16_false.state_dict(),"./vgg16_method2.pth")

class Zowo(nn.Module):
    def __init__(self):
        super(Zowo,self).__init__()
        self.conv = nn.Conv2d(2,32,4)

    def forward(self, x):
        return self.conv(x)
zowo = Zowo()
torch.save(zowo,"zowo1.pth")
torch.save(zowo.state_dict(),'zowo2.pth')