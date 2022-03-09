import pandas as pd
import statistics
import requests
from tqdm import tqdm
from src.utils import get_disp_eiendommer, convert_crs


TERRAIN_API_URL = 'https://api.mapbox.com/v4/mapbox.mapbox-terrain-v2/tilequery/'
MAPBOX_API_KEY = "pk.eyJ1IjoibWFzdGVydGhlc2lzIiwiYSI6ImNrem85bG11dDF3OWoyb2xsbXd6NmtwcWoifQ.KY6Ar4sI88UYi2vhg21lPg"


def create_farmers_information_dataset():
    all_farmers = pd.DataFrame(columns=['orgnr', 'longitude', 'latitude', 'elevation'])
    disposed_properties = get_disp_eiendommer()
    unique_farms = disposed_properties.drop_duplicates('orgnr')['orgnr'].tolist()

    for orgnr in tqdm(unique_farms):
        row = disposed_properties.loc[disposed_properties['orgnr'] == orgnr].sort_values(by=['year']).iloc[0]
        coordinates = convert_crs([row['geometry'].centroid])[0]

        all_farmers.loc[orgnr, 'orgnr'] = orgnr
        all_farmers.loc[orgnr, 'longitude'] = coordinates.x
        all_farmers.loc[orgnr, 'latitude'] = coordinates.y

    all_farmers.reset_index(drop=True, inplace=True)

    # ----------------------------------- Getting Mapbox elevations ----------------------------------------------------
    payload = {
        'radius': '0',
        'limit': '20',
        'dedupe': 'false',
        'geometry': 'polygon',
        'access_token': MAPBOX_API_KEY,
    }

    p_bar = tqdm(total=len(all_farmers), iterable=all_farmers.iterrows())
    for index, farm in p_bar:
        features = ""
        try:
            r = requests.get(f"{TERRAIN_API_URL}{farm['longitude']},{farm['latitude']}.json", params=payload)
            json = r.json()
            features = json['features']
            properties = map(lambda x: x['properties'], features)
            properties = filter(lambda x: 'ele' in x, properties)
            elevations = list(map(lambda x: x['ele'], properties))
            all_farmers.loc[index, 'elevation'] = statistics.mean(elevations)
        except:
            print(f"ERROR! Bad request or result for farm: {farm}, with data: {features}")

    if all_farmers.isna().sum().sum() != 0:
        print(f"ERROR! Found NaN in final dataset:")
        print(all_farmers.isna().sum())

    all_farmers.to_csv("../../kornmo-data-files/raw-data/farm-information/geographic-location/all-farmers-with-location.csv")


if __name__ == '__main__':
    create_farmers_information_dataset()
