import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
from pylab import mpl

# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]

# header = ['longitude', 'latitude', 'housingMedianAge',
#           'totalRooms', 'totalBedrooms', 'population',
#           'households', 'medianIncome', 'medianHouseValue']
# 设置表头
header = ['经度', '纬度', '住房年龄中位数', '总房间数', '卧室总数', '人口', '家庭', '中位数收入', '中位数房屋价值']
# 从文件中读取数据
df = pd.read_csv('cal_housing.data', names=header)

# 绘制地图
plt.figure(figsize=(10, 8))
sns.scatterplot(x='纬度', y='经度', data=df)
# 创建地图
m = Basemap(
    projection='merc',
    llcrnrlat=df['纬度'].min() - 1,
    urcrnrlat=df['纬度'].max() + 1,
    llcrnrlon=df['经度'].min() - 1,
    urcrnrlon=df['经度'].max() + 1,
    resolution='i'
)
# 画出海岸线和国家边界
m.drawcoastlines()
m.drawcountries()
# 将经纬度转换为地图坐标
x, y = m(df['经度'].values, df['纬度'].values)
# 在地图上添加散点
scatter = m.scatter(x, y, c=df['中位数房屋价值'], cmap='viridis', s=df['中位数收入']*15, alpha=0.7, edgecolor='k',
                    linewidth=1)
# 添加颜色条
plt.colorbar(scatter, label='中位数房屋价值')
plt.title('地理数据与 中位数房屋价值/中位数收入 关系可视化')
plt.text(0.5, -0.1, '颜色代表中位数房屋价值，圈的大小代表中位数收入', ha='center', va='center', transform=plt.gca().transAxes)
plt.show()


plt.figure(figsize=(10, 6))
# 使用 Seaborn 的 countplot 函数绘制 'housingMedianAge' 的柱状图
df['住房年龄中位数'] = df['住房年龄中位数'].astype(int)
sns.countplot(x='住房年龄中位数', data=df)
# 添加标题和轴标签
plt.title('住房年龄中位数分布')
plt.xlabel('住房年龄中位数')
plt.ylabel('数目')
# 显示图形
plt.show()
