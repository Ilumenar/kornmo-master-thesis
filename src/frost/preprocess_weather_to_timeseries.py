import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
from kornmo.frost.utils import explode_on_column, with_dict_as_columns, with_col_as_type_datetime, append_df_to_csv
from warnings import simplefilter
from collections import Counter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


# ------------------------- 3. Processes every measurement for each sensor as a time series data -----------------------

def get_day_index(start_date_str, end_date_str):
    s_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    e_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    delta_days = (e_date - s_date).days

    return [s_date + timedelta(days=i) for i in range(delta_days)]


def get_single_observation(observations):
    # Preferably us the same offset on all, if not available just use the first.
    pt6h_obs = list(filter(lambda obs: obs['timeOffset'] == 'PT6H', observations))
    return pt6h_obs[0] if pt6h_obs else observations[0]


def with_observation_as_dict(data_df):
    data_df['observations'] = data_df['observations'].apply(get_single_observation)
    return data_df


def process_temperature_columns(station_df, station_exploded_df, date, index):
    same_date = station_exploded_df["referenceTime"].dt.date == date.date()
    measurements_of_current_date = station_exploded_df.loc[same_date]

    daily_observation_min = np.nan
    daily_observation_max = np.nan
    daily_observation_mean = np.nan

    if not measurements_of_current_date.empty:
        daily_observation_min = measurements_of_current_date.value.min()
        daily_observation_max = measurements_of_current_date.value.max()
        daily_observation_mean = round(measurements_of_current_date.value.mean(), 1)

    station_df[f'day_{index}_min'] = daily_observation_min
    station_df[f'day_{index}_max'] = daily_observation_max
    station_df[f'day_{index}_mean'] = daily_observation_mean

    return station_df


# For processing precipitation and daydegrees .csv rows
def process_simple_columns(station_df, station_exploded_df, date, index):
    same_date = station_exploded_df["referenceTime"].dt.date == date.date()
    measurements_of_current_date = station_exploded_df.loc[same_date]

    daily_observation_value = np.nan
    if not measurements_of_current_date.empty:
        amount_measurements = len(measurements_of_current_date)
        if amount_measurements > 1:
            print(f"Found more than 1 measurement on date {date}")

        # Takes first measurement, no matter what
        daily_observation_value = measurements_of_current_date['value'].iloc[0]

    station_df[f'day_{index}'] = daily_observation_value
    return station_df


def translate_ground_value(value):

    if value == 0 or value == 1 or value == 2 or value == 3 or value == 4:
        return value

    # Glaze on ground
    elif value == 5:
        return 5

    # The one extremely dry value, which is wrong
    elif value == 9:
        return 4

    # Ice on ground
    elif value == 10:
        return 6

    # Some snow on ground
    elif value == 11 or value == 15:
        return 7

    # Mostly snow on ground
    elif value == 12 or value == 16:
        return 8

    # Completely snow on ground
    elif value == 13 or value == 14 or value == 17 or value == 18 or value == 19:
        return 9

    # Missing measurement
    elif value == 31 or value == -1:
        return -1

    else:
        print(f"Found an unsupported value: {value}, of type {type(value)}!")


def find_average_status(values):
    occurrences = Counter(values)

    elements = list(occurrences.values())
    keys = list(occurrences.keys())

    if elements[0] != max(elements):

        first_element = elements[0]
        first_key = keys[0]

        for i in range(len(elements)):
            if elements[i] == max(elements):
                elements[0] = elements[i]
                keys[0] = keys[i]
                elements[i] = first_element
                keys[i] = first_key

    if elements[0] >= (len(values) - 1):
        return keys[0]

    # If 50/50 between two types:
    elif elements[0] >= (len(values) / 2) and len(occurrences.keys()) == 2:

        # If mix of moist and wet, return wet
        moist_wet = [1, 2]
        if set(keys).issubset(moist_wet):
            return 2

        # If mix of snow amount, return mostly snow
        snow = [7, 8, 9]
        if set(keys).issubset(snow):
            return 8

        # If mix of glaze and ice, return ice
        ice_glaze = [5, 6]
        if set(keys).issubset(ice_glaze):
            return 6

    # If now, there is a type with two occurrences, use it.
    if elements[0] >= 2:
        return keys[0]

    # If by now, these are the three values, return moist
    something_moist_wet = [0, 1, 2]
    if set(keys).issubset(something_moist_wet):
        return 1

    print(f"No soultion for the values: {values}, with occurance {occurrences}")
    print(f"They had the following elements: {elements} and keys: {keys}")

    return -1


def process_ground_columns(station_df, station_exploded_df, date, index):
    same_date = station_exploded_df["referenceTime"].dt.date == date.date()
    measurements_of_current_date = station_exploded_df.loc[same_date]

    if not measurements_of_current_date.empty:
        all_amounts = []

        for _, value in measurements_of_current_date['value'].iteritems():
            translated_value = translate_ground_value(value)
            if translated_value != -1:
                all_amounts.append(translated_value)

        if len(all_amounts) >= 1:
            dominant_value = find_average_status(all_amounts)

            if dominant_value != -1:
                station_df[f'day_{index}'] = dominant_value
                return station_df

    station_df[f'day_{index}'] = np.nan
    return station_df


def preprocess_weather_data(start_date, end_date, weather_type, WEATHER_TYPES):
    new_filepath = f'../../../kornmo-data-files/raw-data/weather-data/processed/{weather_type}/{weather_type}_processed_{start_date}_to_{end_date}.csv'
    source_filename = f'../../../kornmo-data-files/raw-data/weather-data/raw/{weather_type}/{weather_type}_raw_{start_date}_to_{end_date}.csv'

    total_row_count = sum(1 for _ in open(source_filename))
    # Process 50 rows at a time
    chunksize = 300
    chunks_processed = 0
    max_chunks = total_row_count // chunksize

    # Before starting, remove any existing data (if any)
    if os.path.exists(new_filepath):
        print(f"Removed existing file {new_filepath}")
        os.remove(new_filepath)

    print("---- Splitting frost data into columns ----")
    print(f"Frost data type: {weather_type}")
    print(f"Total rows to process: {total_row_count}")
    print(f"Splitting the work into chunks of {chunksize}")
    print(f"Results found in path: {new_filepath}")

    for chunk in pd.read_csv(source_filename, chunksize=chunksize):
        # print(chunk)
        df = chunk.dropna().reset_index(drop=True)
        df["data"] = df["data"].apply(eval)

        stations_exploded_df = (df.pipe(explode_on_column, 'data')
                                .pipe(with_dict_as_columns, 'data')
                                .pipe(with_observation_as_dict)
                                .pipe(with_dict_as_columns, 'observations')
                                .pipe(with_col_as_type_datetime, 'referenceTime')
                                )
        daily_index = get_day_index(start_date, end_date)
        stations_exploded_df = stations_exploded_df.groupby('station_id')

        all_station_readings = []
        for index, station_exploded_df in stations_exploded_df:
            station_df = station_exploded_df.head(1)[['station_id', 'growth_season', 'elementId', 'unit']]
            for index, date in enumerate(daily_index):

                if weather_type == WEATHER_TYPES.TEMPERATURE:
                    station_df = process_temperature_columns(station_df, station_exploded_df, date, index)

                elif weather_type == WEATHER_TYPES.PRECIPITATION:
                    station_df = process_simple_columns(station_df, station_exploded_df, date, index)

                elif weather_type == WEATHER_TYPES.DAYDEGREE0 or weather_type == WEATHER_TYPES.DAYDEGREE5:
                    station_df = process_simple_columns(station_df, station_exploded_df, date, index)

                elif weather_type == WEATHER_TYPES.GROUND:
                    station_df = process_ground_columns(station_df, station_exploded_df, date, index)

                elif weather_type == WEATHER_TYPES.SUNLIGHT:
                    station_df = process_simple_columns(station_df, station_exploded_df, date, index)

                else:
                    print(f"No processing function for {weather_type}")

            all_station_readings.append(station_df)

        seasonal_data_df = pd.concat(all_station_readings, ignore_index=True)
        append_df_to_csv(seasonal_data_df, new_filepath)

        # Status printer
        chunks_processed += 1
        rows_processed = chunks_processed * chunksize
        status_percentage = round((rows_processed / total_row_count) * 100, 1)
        sys.stdout.write('\r')
        sys.stdout.write(f'[{"=" * chunks_processed}{" " * (max_chunks - chunks_processed)}] {status_percentage}%')
        sys.stdout.flush()

    print(f"\n Done with pre-processing {weather_type} for {start_date}\n")
