import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
from shapely.ops import transform
from shapely import wkt
import os
import pyproj
from satellite_images import read_sat_images_file

data_location = "../kornmo-data-files/raw-data"

def get_farmer_centroid(nrows=None):
    farmer_centroid = pd.read_csv(os.path.join(data_location, 'farm-information/centroid_coordinates_new.csv'), delimiter=',', nrows=nrows)
    columns_to_keep = ['orgnr', 'latitude', 'longitude', 'kommunenr']
    farmer_centroid = farmer_centroid.filter(columns_to_keep)
    farmer_centroid = farmer_centroid.dropna()
    farmer_centroid['orgnr'] = farmer_centroid['orgnr'].astype(str)
    farmer_centroid['latitude'] = farmer_centroid['latitude'].astype(float)
    farmer_centroid['longitude'] = farmer_centroid['longitude'].astype(float)
    farmer_centroid['kommunenr'] = farmer_centroid['kommunenr'].astype(int)
    return farmer_centroid

def get_polygon_data(nrows=None):
    soilquality = pd.read_csv(os.path.join(data_location, 'soil-data/jordsmonn_geometry.csv'), dtype=str, nrows=nrows)
    soilquality = soilquality.dropna()
    soilquality['geometry'] = soilquality['geometry'].apply(wkt.loads)
    geo_soilquality = gpd.GeoDataFrame(soilquality, crs='epsg:4326')
    geo_soilquality['municipal_nr'] = geo_soilquality['municipal_nr'].astype(int)
    return geo_soilquality

def get_disp_eiendommer():
    disp_eien = gpd.read_file(os.path.join(data_location, 'farm-information/farm-properties/disposed-properties-previous-students/disponerte_eiendommer.gpkg'), layer='disponerte_eiendommer')
    disp_eien = disp_eien.dropna()
    disp_eien.drop_duplicates(['orgnr', 'geometry'], keep='first', inplace=True)
    disp_eien['orgnr'] = disp_eien['orgnr'].astype(str)
    return disp_eien

def get_combined_satellite_data():
    sat_images0 = read_sat_images_file('sentinel_100x100_0.h5')
    sat_images1 = read_sat_images_file('sentinel_100x100_1.h5')
    in_first = set(sat_images0)
    in_second = set(sat_images1)

    in_second_but_not_in_first = in_second - in_first

    result = list(sat_images0) + list(in_second_but_not_in_first)

    return result

def filter_data():
    #Getting datasets
    farmer_centroid = get_farmer_centroid()
    field_data = get_polygon_data()
    disp_eien = get_disp_eiendommer()

    #Only keep data that have common orgnrs
    sat_orgnr = np.array(get_combined_satellite_data())
    farm_orgnr = np.array(list(disp_eien['orgnr']))
    intersection = np.intersect1d(sat_orgnr, farm_orgnr)
    filtered_disp_eien = disp_eien[disp_eien['orgnr'].isin(intersection)]

    # print(f"Amount of fields from disposed properties: {filtered_disp_eien.shape}")
    # print(f"Amount of organisation numbers from satellite data: {len(filtered_satellite_data)}")
    # print(f"Amount of fields from jordsmonn: {field_data.shape}")

    #Only keep data that have common municipal numbers
    municipal_nrs, idxs_to_remove = [], []
    orgnrs_to_check = list(set(farmer_centroid['orgnr'].tolist()))
    for index, row in tqdm(filtered_disp_eien.iterrows(), total=filtered_disp_eien.shape[0]):
        if row['orgnr'] in orgnrs_to_check:
            municipal_nrs.append(farmer_centroid.loc[farmer_centroid['orgnr'] == row['orgnr']]['kommunenr'].iloc[0])
        else:
            idxs_to_remove.append(index)
    filtered_disp_eien = filtered_disp_eien.drop(idxs_to_remove)
    #Adds municipal numbers to disposed properties
    filtered_disp_eien.insert(1, "municipal_nr", municipal_nrs)

    return field_data, filtered_disp_eien

def filter_by_municipal(dataframe, municipal_nr):
    return dataframe.loc[dataframe['municipal_nr'] == municipal_nr]

def convert_crs(polygons):
    project = pyproj.Transformer.from_proj(pyproj.Proj('epsg:25833'), pyproj.Proj('epsg:4326'), always_xy=True)
    return [transform(project.transform, poly) for poly in polygons]

def extract_orgnr_per_field(field_data, filtered_disp_eien):
    intersections_df = []
    municipal_nrs = list(set(filtered_disp_eien['municipal_nr'].tolist()))

    for municipal_nr in tqdm(municipal_nrs):

        filtered_disp = filter_by_municipal(filtered_disp_eien, municipal_nr)
        filtered_fields = filter_by_municipal(field_data, municipal_nr)

        polygons_fields = filtered_fields['geometry'].tolist()
        polygons_disp = filtered_disp['geometry'].tolist()
        id_fields = filtered_fields['id'].tolist()
        orgnr_disp = filtered_disp['orgnr'].tolist()
        
        for i, poly_field in enumerate(polygons_fields):
            disp_orgnrs = []
            for j, poly_disp in enumerate(polygons_disp):
                
                if poly_disp.intersects(poly_field):
                    disp_orgnrs.append(orgnr_disp[j])
            if len(disp_orgnrs) > 0:
                intersections_df.append([id_fields[i], municipal_nr, disp_orgnrs])

    intersections_df = pd.DataFrame(intersections_df, columns=['field_id', 'municipal_nr', 'orgnrs'])
    return intersections_df

if __name__ == '__main__':

    print("Fetching Data")
    field_data, filtered_disp_eien = filter_data()

    print("Extracting orgnrs per field")
    intersections_df = extract_orgnr_per_field(field_data, filtered_disp_eien)

    print("Create csv from data")
    intersections_df.to_csv(os.path.join(data_location, 'farm-information/orgnrs_per_field.csv'))
    print(intersections_df.head())
    print(intersections_df.shape)
    