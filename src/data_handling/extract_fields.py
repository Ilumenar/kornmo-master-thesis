import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
from shapely.geometry import box, mapping, Polygon
from shapely import wkt
import matplotlib.pyplot as plt
import os, sys
from satellite_images import read_sat_images_file
from fiona import collection
from fiona.crs import from_epsg
from pyproj import Proj, Transformer, transform
from utils import boundingBox


data_location = "E:/Universitetet i Agder/Mikkel Andreas Kvande - kornmo-data-files/raw-data"

def get_farmer_elevation(nrows=None):
    farmer_elevation = pd.read_csv(os.path.join(data_location, 'farm-information/farmer_elevation.csv'), delimiter=',', nrows=nrows)
    columns_to_keep = ['orgnr', 'lat', 'lng', 'commune_id']
    farmer_elevation = farmer_elevation.filter(columns_to_keep)
    farmer_elevation['orgnr'] = farmer_elevation['orgnr'].astype(str)
    farmer_elevation['lat'] = farmer_elevation['lat'].astype(float)
    farmer_elevation['lng'] = farmer_elevation['lng'].astype(float)
    farmer_elevation['commune_id'] = farmer_elevation['commune_id'].astype(int)
    return farmer_elevation

def get_polygon_data(nrows=None):
    
    soilquality = pd.read_csv(os.path.join(data_location, 'soil-data/jordsmonn_geometry.csv'), dtype=str)
    soilquality['geometry'] = soilquality['geometry'].apply(wkt.loads)
    geo_soilquality = gpd.GeoDataFrame(soilquality, crs='epsg:4326')
    geo_soilquality['KOMID'] = geo_soilquality['KOMID'].astype(int)
    
    
    return geo_soilquality

#Combines the org numbers from both files
def get_combined_satellite_data():
    sat_images0 = read_sat_images_file('sentinel_100x100_0.h5')
    sat_images1 = read_sat_images_file('sentinel_100x100_1.h5')
    in_first = set(sat_images0)
    in_second = set(sat_images1)

    in_second_but_not_in_first = in_second - in_first

    result = list(sat_images0) + list(in_second_but_not_in_first)

    return result


def get_orgnrs_by_municipal(municipal_nr, orgnr_dataframe):
    return orgnr_dataframe.loc[orgnr_dataframe['commune_id'] == municipal_nr]


def get_polygons_by_municipal(polygons, municipal_nr):
    polygons_by_muni = polygons.loc[polygons['KOMID'] == municipal_nr]
    polygons_list = list(polygons_by_muni['geometry'])
    return polygons_list

if __name__ == '__main__':
    print("Retrieving farmer elevation data, satellite data and polygon data")
    farmer_elevation = get_farmer_elevation()
    polygons = get_polygon_data()

    sat_orgnr = np.array(get_combined_satellite_data())
    farm_orgnr = np.array(list(farmer_elevation['orgnr']))
    intersection = np.intersect1d(sat_orgnr, farm_orgnr)
    
    filtered_farmer_elevation = farmer_elevation[farmer_elevation['orgnr'].isin(intersection)]
    filtered_satellite_data = intersection[:]

    print(f"Farmer elevation shape: {filtered_farmer_elevation.shape}")
    print(f"Length of satellite data after filtering: {len(filtered_satellite_data)}")

    municipal_nrs_farm = np.array(list(set(filtered_farmer_elevation['commune_id'])))
    municipal_nrs_sat = np.array(list(set(polygons['KOMID'])))
    intersection = np.intersect1d(municipal_nrs_farm, municipal_nrs_sat)

    filtered_farmer_elevation = filtered_farmer_elevation[filtered_farmer_elevation['commune_id'].isin(intersection)]
    filtered_polygons = polygons[polygons['KOMID'].isin(intersection)]

    print(f"Farmer elevation shape after second filtering: {filtered_farmer_elevation.shape}")
    print(f"Polygon data shape after second filtering: {filtered_polygons.shape}")
   
    

    municipal_nrs = list(set(filtered_farmer_elevation['commune_id']))
    for municipal_nr in municipal_nrs:
        print(municipal_nr)
        orgnrs_by_municipal_nr = get_orgnrs_by_municipal(municipal_nr, filtered_farmer_elevation)
        print(orgnrs_by_municipal_nr.shape)
        polygons_by_municipal_nr = get_polygons_by_municipal(filtered_polygons, municipal_nr)
        print(len(polygons_by_municipal_nr))
        
        # schema = {'geometry': 'Polygon'}
        # with collection('shapefiles/' + str(municipal_nr) + '.shp', "w", crs=from_epsg(25833), driver="ESRI Shapefile", schema=schema) as output:
        #     for polygon in tqdm(polygons_by_municipal_nr):
        #         output.write({ 'geometry': mapping(polygon)})
        
        
        row = orgnrs_by_municipal_nr.loc[orgnrs_by_municipal_nr['commune_id'] == municipal_nr].iloc[0]
        bounding_box = boundingBox(row['lat'], row['lng'], 1)
        
        polygons_plt = [mapping(poly)['coordinates'][0][0] for poly in polygons_by_municipal_nr]
        
        for coords in polygons_plt:
            for xy in coords:
                print(xy)


        #print(bounding_box)
        transformer = Transformer.from_crs('epsg:25833', 'epsg:4326')
        x0, y0 = transformer.transform(bounding_box[0], bounding_box[1])
        x1, y1 = transformer.transform(bounding_box[2], bounding_box[3])

        print(x0, y0, x1, y1)
        #box = box(x0, y0, x1, y1)
        box = box(bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3])

        for p in polygons_plt:
            plt.plot(*Polygon(p).exterior.xy)

        plt.plot(*box.exterior.xy)
        plt.show()
        # for orgnr in list(orgnrs_by_municipal_nr['orgnr']):
        #     #print(filtered_farmer_elevation.loc[filtered_farmer_elevation['orgnr'] == orgnr].iloc[0])
        #     ...
        #     break
        break
    