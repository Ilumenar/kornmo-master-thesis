import os
from shapely.geometry import mapping, Polygon
from fiona import collection
from fiona.crs import from_epsg
import geopandas as gpd
from tqdm import  tqdm
import pandas as pd
from shapely import wkt


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



# def GeoJSONToFC(shpname,USGSurl):
#     from urllib.request import urlopen
#     import json
#     from shapely.geometry import Point, mapping
#     from fiona import collection
#     from fiona.crs import from_epsg
#     crs = from_epsg(4326)
#     schema = {'geometry': 'Point', 'properties': { 'Place': 'str', 'Magnitude': 'str' }}
#     with collection(shpname, "w", crs=crs, driver="ESRI Shapefile", schema=schema) as output:
#         url = USGSurl
#         weburl = urlopen(url)
#         if weburl.getcode() == 200:
#             data = json.loads(weburl.read())
#         for i in data["features"]:
#             mag, place = i["properties"]["mag"],i["properties"]["place"]
#             x, y = float(i["geometry"]["coordinates"][0]), float(i["geometry"]["coordinates"][1])
#             point = Point(x,y)
#             output.write({'properties':{'Place': place, 'Magnitude': mag}, 'geometry': mapping(point)})

# shpname = "quakes.shp"
# url = "http://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_month.geojson"
# #GeoJSONToFC(shpname,url)



