import pandas as pd
import seaborn as sns

header = ['longitude', 'latitude', 'housingMedianAge',
          'totalRooms', 'totalBedrooms', 'population',
          'households', 'medianIncome', 'medianHouseValue']
data = pd.read_csv('cal_housing.data', names=header)
