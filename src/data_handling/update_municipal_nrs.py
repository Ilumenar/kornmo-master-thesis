import os
import pandas as pd

data_location = "E:/Universitetet i Agder/Mikkel Andreas Kvande - kornmo-data-files/raw-data"

print("Reading data")
new_numbers = pd.read_csv(os.path.join(data_location, 'farm-information/new-farm-and-commune-numbers.csv'), delimiter=',', dtype=str)
coords = pd.read_csv(os.path.join(data_location, 'farm-information/centroid_coordinates.csv'), delimiter=',', dtype=str)

print("Chaning the old municipal numbers to the new ones")
coords['kommunenr'] = coords['kommunenr'].apply(lambda x: new_numbers.loc[new_numbers['kommunenr_old'] == x]['kommunenr_new'].iloc[0])


print("Creating csv from new dataset")
coords = coords.filter(['orgnr', 'kommunenr', 'longitude', 'latitude'])
coords.to_csv(os.path.join(data_location, 'farm-information/centroid_coordinates_new.csv'))