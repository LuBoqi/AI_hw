import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

train_rate = 0.8
epochs = 1000
learning_rate = 0.001
batch_size = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
header = ['经度', '纬度', '住房年龄中位数', '总房间数', '卧室总数', '人口', '家庭', '中位数收入', '中位数房屋价值']
# 从文件中读取数据
df = pd.read_csv('cal_housing.data', names=header)
print(df.info())
# 对DataFrame中的各个列进行归一化
df = (df - df.min()) / (df.max() - df.min())
print(df.info())
x = df.iloc[:, :-1]
y = df.iloc[:, -1]
x = torch.from_numpy(x.values).float()
y = torch.from_numpy(y.values).float()
index = torch.randperm(len(y))
x, y = x[index], y[index]
train_len = int(len(y) * train_rate)
trainX, trainY = x[:train_len], y[:train_len]
testX, testY = x[train_len:], y[train_len:]
data_set = TensorDataset(x,y)
train_set = TensorDataset(trainX, trainY)
test_set = TensorDataset(testX, testY)
data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


class BPNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 64)
        self.fc2 = torch.nn.Linear(64, 512)
        self.fc3 = torch.nn.Linear(512, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = torch.relu(out)
        out = self.fc2(out)
        out = torch.relu(out)
        out = self.fc3(out)
        return out


# model = torch.load('BP.pth').to(device)
model = BPNet().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_save = []
for epoch in range(1000):
    loss_val = []
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y.view(-1, 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_val.append(loss.item())
    loss_save.append(sum(loss_val) / len(loss_val))
    print(f'Epoch {epoch + 1}, Loss: {sum(loss_val) / len(loss_val)}')

plt.plot(range(1, len(loss_save) + 1), loss_save, label='Training Error')
plt.title('BP Training Curve')
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()
torch.save(model, 'BP.pth')

model.eval()
correct = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        y_pred = y_pred.squeeze()
        correct += (torch.abs(y_pred - y) < 0.1).sum()
print('预测正确率为：', correct.item() / (len(test_loader) * batch_size))
