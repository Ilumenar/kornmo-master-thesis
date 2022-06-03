import geopandas as gpd
import pandas as pd
import fiona
from tqdm import tqdm
import os
import sys

# Python script for converting our gdb-files to csv files.

print("Reading jordsmonn_norge")
layers = fiona.listlayers('../kornmo_old-data-files/raw-data/soil-data/jordsmonn.gdb')
jordsmonn = gpd.GeoDataFrame()

for layer in tqdm(layers, total=len(layers)):
    new_gdf = gpd.read_file('../kornmo_old-data-files/raw-data/soil-data/jordsmonn.gdb', layer=layer)
    jordsmonn = pd.concat([jordsmonn, new_gdf])

ids = range(0, jordsmonn.shape[0])
jordsmonn.insert(0, "id", ids)
jordsmonn = jordsmonn.rename(columns = {'KOMID': 'municipal_nr'})

print("Reading jordkvalitet_norge.gdb")

soilquality = gpd.read_file('../kornmo_old-data-files/raw-data/soil-data/soil-quality.gdb', driver='FileGDB', layer=0)
soilquality = soilquality.dropna()


print("Creating jordsmonn for geometry and municipal_id")
jordsmonn_geometry = jordsmonn[['id', 'municipal_nr', 'geometry']]

print("Removing columns")
columns_to_remove = ['objtype', 'kartleggingsetappe', 'originaldatavert', 'områdeid', 'kopidato', 'navnerom', 'lokalid', 'geometry']
df_for_models = soilquality.drop(columns=columns_to_remove)
df_with_all = soilquality




print("Converting to csv")
df_for_models.to_csv(os.path.join('../kornmo_old-data-files/raw-data/soil-data', 'soilquality_refined.csv'))
df_with_all.to_csv(os.path.join('../kornmo_old-data-files/raw-data/soil-data', 'soilquality.csv'))
jordsmonn.to_csv(os.path.join('../kornmo_old-data-files/raw-data/soil-data', 'jordsmonn.csv'))
jordsmonn_geometry.to_csv(os.path.join('../kornmo_old-data-files/raw-data/soil-data', 'jordsmonn_geometry.csv'))
