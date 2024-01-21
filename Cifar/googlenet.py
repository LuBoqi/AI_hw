import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.datasets import CIFAR10
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 定义训练函数
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    loss_val = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data).logits
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_val.append(loss.item())
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


# 定义数据预处理
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
# 加载CIFAR-10数据集
train_dataset = CIFAR10(root='./', train=True, download=True, transform=transform)
test_dataset = CIFAR10(root='./', train=False, download=True, transform=transform)
# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=2)
# 初始化GoogleNet模型
model = models.googlenet(num_classes=10, init_weights=True).to(device)
print(model)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练模型
# num_epochs = 300
# for epoch in range(1, num_epochs + 1):
#     train(model, train_loader, criterion, optimizer, epoch)
#     test_loss = test(model, test_loader)
# torch.save(model, 'googlenet.pth')

model = torch.load('googlenet.pth').to(device)
# 准确度分析
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        # 通过网络运行图像以计算输出
        outputs = model(images)
        # 概率最高的类是我们选择的预测
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
model.eval()  # 将模型设置为评估模式


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
            outputs = model(images)
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
