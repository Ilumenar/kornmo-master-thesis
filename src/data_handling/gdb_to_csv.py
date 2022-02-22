import geopandas as gpd
import pandas as pd
import fiona
from tqdm import tqdm
import os


print("Reading jordsmonn_norge")
layers = fiona.listlayers('../kornmo-data-files/raw-data/soil-data/jordsmonn.gdb')
jordsmonn = gpd.GeoDataFrame()

for layer in tqdm(layers, total=len(layers)):
    new_gdf = gpd.read_file('../kornmo-data-files/raw-data/soil-data/jordsmonn.gdb', layer=layer)
    jordsmonn = pd.concat([jordsmonn, new_gdf])

print("Reading jordkvalitet_norge.gdb")

soilquality = gpd.read_file('../kornmo-data-files/raw-data/soil-data/soil-quality.gdb', driver='FileGDB', layer=0)
soilquality = soilquality.dropna()


print("Creating jordsmonn for geometry and municipal_id")
jordsmonn_geometry = jordsmonn[['KOMID', 'geometry']]

print("Removing columns")
columns_to_remove = ['objtype', 'kartleggingsetappe', 'originaldatavert', 'områdeid', 'kopidato', 'navnerom', 'lokalid', 'geometry']
df_for_models = soilquality.drop(columns=columns_to_remove)
df_with_all = soilquality

print("Converting to csv")
df_for_models.to_csv(os.path.join('../kornmo-data-files/raw-data/soil-data', 'soilquality_refined.csv'))
df_with_all.to_csv(os.path.join('../kornmo-data-files/raw-data/soil-data', 'soilquality.csv'))
jordsmonn.to_csv(os.path.join('../kornmo-data-files/raw-data/soil-data', 'jordsmonn.csv'))
jordsmonn_geometry.to_csv(os.path.join('../kornmo-data-files/raw-data/soil-data', 'jordsmonn_geometry.csv'))
