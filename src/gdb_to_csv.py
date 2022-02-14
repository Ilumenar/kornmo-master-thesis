import geopandas as gpd
import pandas as pd
import fiona
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import plot_bar
import os

import geopandas

#gdb_soilqual = 'C:/Users/Sigurd/Desktop/jordkvalitet_agder.gdb'
soilquality_path = "E:/Universitetet i Agder/Mikkel Andreas Kvande - kornmo-data-files/raw-data/soil-data"


print("Reading jordsmonn_norge")
layers = fiona.listlayers(os.path.join(soilquality_path, 'jordsmonn.gdb'))
jordsmonn = gpd.GeoDataFrame()

for layer in tqdm(layers, total=len(layers)):
    new_gdf = gpd.read_file(os.path.join(soilquality_path, 'jordsmonn.gdb'), layer=layer)
    jordsmonn = pd.concat([jordsmonn, new_gdf])

print("Reading jordkvalitet_norge.gdb")

soilquality = gpd.read_file(os.path.join(soilquality_path, 'soil-quality.gdb'), driver='FileGDB', layer=0)
soilquality = soilquality.dropna()


print("Creating jordsmonn for geometry and municipal_id")
jordsmonn_geometry = jordsmonn[['KOMID', 'geometry']]

print("Removing columns")
columns_to_remove = ['objtype', 'kartleggingsetappe', 'originaldatavert', 'områdeid', 'kopidato', 'navnerom', 'lokalid', 'geometry']
df_for_models = soilquality.drop(columns=columns_to_remove)
df_with_all = soilquality

print("Converting to csv")
df_for_models.to_csv(os.path.join(soilquality_path, 'soilquality_refined.csv'))
df_with_all.to_csv(os.path.join(soilquality_path, 'soilquality.csv'))
jordsmonn.to_csv(os.path.join(soilquality_path, 'jordsmonn.csv'))
jordsmonn_geometry.to_csv(os.path.join(soilquality_path, 'jordsmonn_geometry.csv'))
