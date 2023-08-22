import torch

from torch import nn
class Zowo(nn.Module):
    def __int__(self):
        super().__int__()

    def forword(self,input):
        return input+1

x = torch.tensor(1.0)
zowo = Zowo()
y = zowo.forword(x)

print(y)