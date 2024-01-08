import os
import torch
from torchvision.transforms import transforms
import torchvision.datasets as datasets

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# 将数据加载进来，本地已经下载好， root=os.getcwd()为自动获取与源码文件同级目录下的数据集路径
train_data = datasets.CIFAR10(root=os.getcwd(), train=True, transform=transform_train, download=False)
test_data = datasets.CIFAR10(root=os.getcwd(), train=False, transform=transform_test, download=False)
