import os
import time

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 0.001
batch_size = 32
epochs = 30

transform_train = transforms.Compose([
    # 随机水平翻转
    transforms.RandomHorizontalFlip(),
    # 随机灰度
    transforms.RandomGrayscale(),
    # 将图片转换为tensor
    transforms.ToTensor(),
    # 归一化
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_data = datasets.CIFAR10(root=os.getcwd(), train=True, transform=transform_train, download=False)
test_data = datasets.CIFAR10(root=os.getcwd(), train=False, transform=transform_test, download=False)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=2)


class Net(nn.Module):  # 双层卷积+三层全连接
    def __init__(self):
        super().__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 512, 3)
        # 全连接层
        self.fc1 = nn.Linear(512 * 6 * 6, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):  # torch.Size([16, 1, 32, 32])
        o = x
        o = self.conv1(o)  # torch.Size([16, 64, 30, 30])
        o = nn.ReLU(inplace=True)(o)
        o = nn.MaxPool2d(2, 2)(o)  # torch.Size([16, 512, 15, 111])

        o = self.conv2(o)  # torch.Size([16, 512, 13, 13])
        o = nn.ReLU(inplace=True)(o)
        o = nn.MaxPool2d(2, 2)(o)  # torch.Size([16, 512, 6, 6])

        o = o.reshape(x.size(0), -1)

        o = self.fc1(o)  # 全连接层
        o = nn.ReLU(inplace=True)(o)
        o = nn.Dropout(p=0.5, inplace=False)(o)
        o = self.fc2(o)  # 全连接层
        o = nn.ReLU(inplace=True)(o)
        o = nn.Dropout(p=0.5, inplace=False)(o)
        o = self.fc3(o)  # 全连接层
        return o


start = time.time()
net = Net().to(device)
# net=torch.load('net2.pth')
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
for epoch in range(epochs):
    net.train()
    loss_val = []
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        pre_y = net(x)
        loss = nn.CrossEntropyLoss()(pre_y, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_val.append(loss.item())
        if i % 30 == 0:
            print(r'训练的代数：{} [当前代完成进度：{} / {} ({:.0f}%)]，当前损失函数值: {:.6f}'.
                  format(epoch, i * len(x), len(train_loader.dataset), 100. * i / len(train_loader), np.mean(loss_val)))
middle = time.time()
print('模型训练时间：{:.1f}秒'.format(middle - start))

torch.save(net, 'net2.pth')
# 准确度分析
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        # 通过网络运行图像以计算输出
        outputs = net(images)
        # 概率最高的类是我们选择的预测
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
net.eval()  # 将模型设置为评估模式


# 创建每个类对应二进制标签
def create_binary_labels(labels, target_class):
    binary_labels = [1 if label == target_class else 0 for label in labels]
    return torch.tensor(binary_labels)


# 在每个类上分别评估并绘制 PR 曲线
plt.figure(figsize=(8, 6))
for i, class_name in enumerate(classes):
    print(f"Evaluating for class '{class_name}'...")

    y_true = []
    y_scores = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            binary_labels = create_binary_labels(labels.cpu(), i)
            y_true.extend(binary_labels.numpy())

            softmax_scores = F.softmax(outputs, dim=1)
            confidence_scores = softmax_scores[:, i]
            y_scores.extend(confidence_scores.cpu().numpy())

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    plt.plot(recall, precision, label=f'{class_name} (AUC = {pr_auc:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.title('Precision-Recall Curves for Binary Classification (One-vs-All)')
plt.grid(True)
plt.show()
end = time.time()
print('模型评估时间：{:.1f}秒'.format(end - middle))

correct = 1. * correct / len(test_loader.dataset)
print('在测试集上的预测准确率：{:0.2f}%'.format(100 * correct))
print('运行时间：{:.1f}秒'.format(end - start))
