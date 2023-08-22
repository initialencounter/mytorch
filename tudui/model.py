import torchvision.datasets
from torch import nn

# train_data = torchvision.datasets.ImageNet('../conv/data')
vgg16_false = torchvision.models.vgg16(pretrained=False)
# vgg16_true = torchvision.models.vgg16(pretrained=True)

print(vgg16_false)
vgg16_false.classifier.add_module("add_linear",nn.Linear(1000,10))
print(vgg16_false)