import os
import pandas as pd


print("Reading data")
new_numbers = pd.read_csv('../kornmo_old-data-files/raw-data/farm-information/new-farm-and-commune-numbers.csv', delimiter=',', dtype=str)
coords = pd.read_csv('../kornmo_old-data-files/raw-data/farm-information/centroid_coordinates.csv', delimiter=',', dtype=str)

print("Chaning the old municipal numbers to the new ones")
coords['kommunenr'] = coords['kommunenr'].apply(lambda x: new_numbers.loc[new_numbers['kommunenr_old'] == x]['kommunenr_new'].iloc[0])


print("Creating csv from new dataset")
coords = coords.filter(['orgnr', 'kommunenr', 'longitude', 'latitude'])
coords.to_csv('../kornmo_old-data-files/raw-data/farm-information/centroid_coordinates_new.csv')