import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm import tqdm

from src.utils import get_disp_eiendommer


def get_data():
    deliveries = pd.read_csv('../../../kornmo-data-files/raw-data/farm-information/grain-deliveries/farmer_deliveries_previous_students.csv')
    deliveries = deliveries.loc[deliveries['year'].isin([2017, 2018, 2019])]
    disposed_prop = get_disp_eiendommer()
    return deliveries, disposed_prop


def create_farmer_planted(deliveries):
    planted = pd.DataFrame(columns=['year', 'orgnr', 'planted'])
    planted_unknown = pd.DataFrame(columns=['year', 'orgnr', 'planted'])

    for index, row in tqdm(deliveries.iterrows(), total=deliveries.shape[0], desc="Creating planted and planted unknown"):
        columns_to_check = row.filter(regex='.*\_sum')
        if np.count_nonzero(columns_to_check.tolist()) == 1:
            planted.loc[index] = row.filter(['year', 'orgnr'])
            planted.loc[index, 'planted'] = columns_to_check.idxmax().split('_')[0]
        else:
            planted_unknown.loc[index] = row.filter(['year', 'orgnr'])

    return planted, planted_unknown


def insert_polygons(df, disposed_prop):
    planted_with_polygon_arr = []
    for i, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Inserting polygons"):
        orgnr = str(int(row['orgnr']))
        year = int(row['year'])
        if orgnr in disposed_prop['orgnr'].values:
            disposed_prop_row = disposed_prop.loc[disposed_prop['orgnr'] == orgnr]
            if year in disposed_prop_row['year'].values:
                for polygon in list(disposed_prop_row.loc[disposed_prop_row['year'] == year]['geometry'].iloc[0].geoms):
                    new_row = list(row)
                    new_row.append(polygon)
                    new_row.append(polygon.area)
                    planted_with_polygon_arr.append(new_row)
            else:
                for polygon in list(disposed_prop_row.sort_values(by=['year'])['geometry'].iloc[0].geoms):
                    new_row = list(row)
                    new_row.append(polygon)
                    new_row.append(polygon.area)
                    planted_with_polygon_arr.append(new_row)

    planted_with_polygon = gpd.GeoDataFrame(planted_with_polygon_arr, columns=['year', 'orgnr', 'planted', 'geometry', 'area'])
    return planted_with_polygon


def main():
    deliveries, disposed_prop = get_data()
    planted, planted_unknown = create_farmer_planted(deliveries)

    planted_with_polygon = insert_polygons(planted, disposed_prop)
    planted_unknown_with_polygon = insert_polygons(planted_unknown, disposed_prop)

    print(f"Created {planted_with_polygon.shape[0]} training data")
    print(f"Created {planted_unknown_with_polygon.shape[0]} validation data")

    planted_with_polygon.to_file('../../../kornmo-data-files/raw-data/crop-grain_classification-data/training_data.gpkg', driver="GPKG")
    planted_unknown_with_polygon.to_file('../../../kornmo-data-files/raw-data/crop-grain_classification-data/validation_data.gpkg', driver="GPKG")

    print("Done")


if __name__ == "__main__":
    main()

