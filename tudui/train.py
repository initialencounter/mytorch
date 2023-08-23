

from torch.utils.tensorboard import SummaryWriter
from zowo import Zowo
from torch import nn
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torchvision
# 下载训练数据集
datasets_train = torchvision.datasets.CIFAR10("./data/CIFAR10/train",
                                              download=True,
                                              train=True,
                                              transform=torchvision.transforms.ToTensor())
# 下载测试数据集
datasets_test = torchvision.datasets.CIFAR10("./data/CIFAR10/test",
                                             download=True,
                                             train=False,
                                             transform=torchvision.transforms.ToTensor())

from torch.utils.data import DataLoader
# 加载训练数据集
dataloader_train = DataLoader(datasets_train, batch_size=64)
# 加载测试数据集
dataloader_test = DataLoader(datasets_test, batch_size=64)



# 结束训练轮数
end_epcho = 101
# 开始训练轮数
start_epcho = 37
# 记录测试的次数
total_test_step = start_epcho
# 加载预训练模型
zowo = torch.load(f"zowo_{start_epcho-1}.pth")
zowo = zowo.to(device)
# 定义损失函数
loss_fn = nn.CrossEntropyLoss()


# 定义学习率
learning_rate = 5e-4
optimizer = torch.optim.SGD(zowo.parameters(), lr=learning_rate)

# 定义数据可视化
writer = SummaryWriter("./logs_train")

step = start_epcho*len(dataloader_train)
for i in range(start_epcho, end_epcho):
    print(f"--------------第{i}轮训练开始------------")

    zowo.train()
    for data in dataloader_train:
        img, targets = data
        img = img.to(device)
        targets = targets.to(device)
        output = zowo(img)
        loss = loss_fn(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        if step % 100 == 0:
            print(f"训练次数: {step}，loss: {loss.item()}")
            writer.add_scalar("train_loss", loss.item(), step)


    # 测试
    zowo.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in dataloader_test:
            img, targets = data
            img = img.to(device)
            targets = targets.to(device)
            output = zowo(img)
            loss = loss_fn(output, targets)

            total_test_loss += loss.item()
            accuracy = (output.argmax(1) == targets).sum()
            total_accuracy += accuracy

        print(f"第{i}轮测试数据集的整体loss: {total_test_loss}")
        print(f"第{i}轮测试数据集的整体正确率: {total_accuracy / len(dataloader_test)}")
        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        writer.add_scalar("test_accuracy", total_accuracy / len(dataloader_test), total_test_step)
        total_test_step += 1

    torch.save(zowo, "zowo_{}.pth".format(i))
    print("模型已保存")

writer.close()
