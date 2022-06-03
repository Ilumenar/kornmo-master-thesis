import os
from shapely.geometry import mapping, Polygon
from fiona import collection
from fiona.crs import from_epsg
import geopandas as gpd
from tqdm import  tqdm
import pandas as pd
from shapely import wkt

# Python script for creating shapefiles from polygons to display in qgis

def get_polygon_data(nrows=None):
    
    soilquality = pd.read_csv('../kornmo_old-data-files/raw-data/soil-data/jordsmonn_geometry.csv', dtype=str, nrows=nrows)
    soilquality = soilquality.dropna()
    soilquality['geometry'] = soilquality['geometry'].apply(wkt.loads)
    geo_soilquality = gpd.GeoDataFrame(soilquality, crs='epsg:4326')
    geo_soilquality['KOMID'] = geo_soilquality['KOMID'].astype(int)
    
    return geo_soilquality


def shapefile_dataset(soilquality, komid=-1):
    if komid > 0:
        filtered_data = soilquality.loc[soilquality['komid'] == komid]
        polygons = filtered_data['geometry']
    else:
        polygons = soilquality['geometry']
    schema = {'geometry': 'Polygon'}
    with collection('shapefiles/soilquality.shp', "w", crs=from_epsg(25833), driver="ESRI Shapefile", schema=schema) as output:
        for polygon in tqdm(polygons):
            output.write({ 'geometry': mapping(polygon)})

if __name__ == '__main__':
    soilquality = get_polygon_data()
    shapefile_dataset(soilquality)



