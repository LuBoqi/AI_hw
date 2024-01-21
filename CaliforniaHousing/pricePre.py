import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, learning_curve

# header = ['longitude', 'latitude', 'housingMedianAge',
#           'totalRooms', 'totalBedrooms', 'population',
#           'households', 'medianIncome', 'medianHouseValue']
# 设置表头
header = ['经度', '纬度', '住房年龄中位数', '总房间数', '卧室总数', '人口', '家庭', '中位数收入', '中位数房屋价值']
# 从文件中读取数据
df = pd.read_csv('cal_housing.data', names=header)
print(df.info())
# 对DataFrame中的各个列进行归一化
df = (df - df.min()) / (df.max() - df.min())
x = df.iloc[:, :-1]
y = df.iloc[:, -1]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)
# 设置Seaborn风格
sns.set(style="whitegrid")
# 初始化模型
model = LinearRegression()
# 计算训练曲线
train_sizes, train_scores, test_scores = learning_curve(
    model, x, y, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10)
)
# 计算均值和标准差
train_scores_mean = -np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
# 绘制训练曲线
plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                 color="blue")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                 color="orange")
plt.plot(train_sizes, train_scores_mean, 'o-', color="blue", label="Training Score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="orange", label="Cross-validation Score")
plt.title('Learning Curve')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()
# 训练最终模型
final_model = LinearRegression()
final_model.fit(x, y)
# 绘制预测结果和真实值的散点图
plt.figure(figsize=(50, 20))
sns.scatterplot(x=y.index, y=y,color='blue',label='Actual')
sns.scatterplot(x=y.index, y=final_model.predict(x),color='red',label='Pred')
plt.legend()
plt.title('Actual vs Predicted Values')
plt.xlabel('Index')
plt.ylabel('Values')
plt.show()

# 初始化梯度下降线性回归模型
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, eta0=0.01, random_state=42)
# 拟合模型
# 存储训练集和测试集上的均方误差
train_errors, test_errors = [], []
# 迭代训练
n_iterations = 1000
for iteration in range(n_iterations):
    sgd_reg.partial_fit(xtrain, ytrain.ravel())
    # 在训练集上计算均方误差
    train_pred = sgd_reg.predict(xtrain)
    train_errors.append(mean_squared_error(ytrain, train_pred))

    # 在测试集上计算均方误差
    test_pred = sgd_reg.predict(xtest)
    test_errors.append(mean_squared_error(ytest, test_pred))
# 绘制训练曲线
plt.plot(range(1, n_iterations + 1), train_errors, label='Training Error')
plt.plot(range(1, n_iterations + 1), test_errors, label='Testing Error')
plt.title('SGD Training Curve')
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

# 打印模型参数
print("截距(intercept):", sgd_reg.intercept_)
print("斜率(coef):", sgd_reg.coef_)
# 在测试集上进行预测
y_pred = sgd_reg.predict(xtest)
# 计算均方误差（Mean Squared Error）
mse = mean_squared_error(ytest, y_pred)
print("均方误差（MSE）:", mse)
# 绘制原始数据和拟合直线
plt.figure(figsize=(50, 20))
plt.scatter(ytest.index, ytest, label='Original Data')
plt.scatter(ytest.index, y_pred, label='SGD Data', color='red')
plt.xlabel('index')
plt.ylabel('y')
plt.legend()
plt.show()
