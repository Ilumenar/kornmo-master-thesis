import requests
import pandas as pd
import numpy as np
import sys
from utils import append_df_to_csv, WEATHER_TYPES


# ------------------- 2. Collects measurement data fom each station, for each day in the given year --------------------

def get_frost_data(frost_client_id, source_id, ref_time, element, time_resolution=''):
    # Check Frost if the source has data on given season and element
    params = {
        'sources': [source_id],
        'elements': element,
        'referencetime': ref_time
    }

    if time_resolution:
        params['timeresolutions'] = time_resolution

    observations_url = 'https://frost.met.no/observations/v0.jsonld'
    request = requests.get(observations_url, params, auth=(frost_client_id, ''))
    json = request.json()

    if request.status_code == 200:
        data = json['data']
        return data
    else:
        return np.nan


def download_raw_weather_data(weather_type, from_date, to_date, growth_season, frost_sources_df, frost_elements, time_resolution, client_id):
    file_path = f'../../../kornmo-data-files/raw-data/weather-data/raw/{weather_type}/{weather_type}_raw_{from_date}_to_{to_date}.csv'

    import os
    if os.path.exists(file_path):
        print("Deleted existing file before downloading")
        os.remove(file_path)
    else:
        print("The file does not exist")

    count_sources = frost_sources_df.shape[0]
    for index, frost_source in frost_sources_df.iterrows():
        # Get weather for each station
        # Append it to a csv
        ref_time = f'{from_date}/{to_date}'
        station_id = frost_source['id']
        frost_data = get_frost_data(client_id, station_id, ref_time, frost_elements, time_resolution)
        data = {'station_id': station_id, 'data': frost_data, 'growth_season': growth_season}

        df = pd.DataFrame([data])
        append_df_to_csv(df, file_path)

        sys.stdout.write('\r')
        sys.stdout.write(
            f'[{"=" * (index // 20)}{" " * ((count_sources - index) // 20)}] {round((index / count_sources) * 100, 1)}%')
        sys.stdout.flush()


def get_raw_weather_readings_to_file(from_date, to_date, growth_season, weather_type, client_id):
    frost_sources = pd.read_csv('../../../kornmo-data-files/raw-data/weather-data/frost_weather_sources.csv')
    frost_elements = ""
    time_resolution = ""

    print("---- Downloading raw readings ----")
    print(f"Weather type: {weather_type}")
    if weather_type == WEATHER_TYPES.PRECIPITATION:
        frost_elements = 'sum(precipitation_amount P1D)'
        time_resolution = 'P1D'

    elif weather_type == WEATHER_TYPES.TEMPERATURE:
        frost_elements = 'air_temperature'
        time_resolution = ''

    elif weather_type == WEATHER_TYPES.DAYDEGREE0:
        frost_elements = 'integral_of_excess(mean(air_temperature P1D) P1D 0.0)'
        time_resolution = ''

    elif weather_type == WEATHER_TYPES.DAYDEGREE5:
        frost_elements = 'integral_of_excess(mean(air_temperature P1D) P1D 5.0)'
        time_resolution = ''

    elif weather_type == WEATHER_TYPES.GROUND:
        frost_elements = 'state_of_ground'
        time_resolution = ''

    elif weather_type == WEATHER_TYPES.SUNLIGHT:
        frost_elements = 'sum(duration_of_sunshine P1D)'
        time_resolution = ''

    else:
        print(f"\nFound unsupported weather type: {weather_type}!")

    download_raw_weather_data(weather_type, from_date, to_date, growth_season, frost_sources,
                              frost_elements, time_resolution, client_id)

    print(f"\n Done with getting raw {weather_type} for {growth_season}\n")
