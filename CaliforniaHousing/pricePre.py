import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# header = ['longitude', 'latitude', 'housingMedianAge',
#           'totalRooms', 'totalBedrooms', 'population',
#           'households', 'medianIncome', 'medianHouseValue']
# 设置表头
header = ['经度', '纬度', '住房年龄中位数', '总房间数', '卧室总数', '人口', '家庭', '中位数收入', '中位数房屋价值']
# 从文件中读取数据
df = pd.read_csv('cal_housing.data', names=header)
print(df.info())
