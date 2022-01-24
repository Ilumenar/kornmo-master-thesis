import geopandas as gpd
import pandas as pd
import fiona
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import plot_bar

# Get all the layers from the .gdb file 

gdb_soilqual = 'C:/Users/Sigurd/Desktop/jordkvalitet_agder.gdb'
gdf = gpd.read_file(gdb_soilqual, driver='FileGDB', layer=0)
gdf = gdf.dropna()

gdb_soil = 'C:/Users/Sigurd/Desktop/0000_25833_jordsmonn_gdb.gdb'
layers = fiona.listlayers(gdb_soil)
gdf2 = gpd.GeoDataFrame()
for layer in tqdm(layers, total=len(layers)):
    new_gdf = gpd.read_file(gdb_soil, layer=layer)
    gdf2 = pd.concat([gdf2, new_gdf])


print(gdf2)


columns_to_remove = ['objtype', 'kartleggingsetappe', 'originaldatavert', 'områdeid', 'kopidato', 'navnerom', 'lokalid', '_clipped', 'geometry']
df_for_models = gdf.drop(columns=columns_to_remove)
df_with_all = gdf

df_for_models.to_csv('data/soilquality_refined.csv')
df_with_all.to_csv('data/soilquality.csv')
gdf2.to_csv('data/soil.csv')