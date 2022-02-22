import os
from numpy import column_stack
import pandas as pd



data_location = "../kornmo-data-files/raw-data"
soilquality = pd.read_csv(os.path.join(data_location, 'soil-data/jordsmonn.csv'), dtype=str)


soilquality = soilquality.drop(['Unnamed: 0'], axis=1)
print(soilquality.head())

ids = range(0, soilquality.shape[0])

soilquality.insert(0, "id", ids)
print(soilquality.head())

soilquality.to_csv(os.path.join(data_location, 'soil-data/jordsmonn.csv'))
