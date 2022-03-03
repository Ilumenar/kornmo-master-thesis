from interpolate_with_nn import *
from interpolate_by_proximity import assign_to_farmer_and_fill_by_proximity
from preprocess_weather_to_timeseries import preprocess_weather_data
from raw_weather_readings import get_raw_weather_readings_to_file
from utils import WEATHER_TYPES
from frost_weather_sources import get_frost_weather_sources_to_file
from add_features_to_dataset import add_feature_to_main_dataset
from precipitation_interpolation_nn import interpolation_precipitation_nn
from sunlight_interpolation_nn import create_and_train_interpolation_nn
from daydegree5_interpolation_nn import interpolation_daydegree5_nn


# Download the weather readings for each of the sources of a given type
#
# raw_frost_readings_to_file(start_date, end_date, growth_season, WEATHER_TYPES.TEMPERATURE, client_id)
# raw_frost_readings_to_file(start_date, end_date, growth_season, WEATHER_TYPES.DAYDEGREE0, client_id)
# raw_frost_readings_to_file(start_date, end_date, growth_season, WEATHER_TYPES.DAYDEGREE5, client_id)
# raw_frost_readings_to_file(start_date, end_date, growth_season, WEATHER_TYPES.GROUND, client_id)
#

# Process the readings
# These files can be used for interpolation / filling in the blanks by
# looking at the 2nd closest, etc.
#
# preprocess_weather(start_date, end_date, WEATHER_TYPES.TEMPERATURE)
# preprocess_weather(start_date, end_date, WEATHER_TYPES.DAYDEGREE0)
# preprocess_weather(start_date, end_date, WEATHER_TYPES.DAYDEGREE5)
# preprocess_weather(start_date, end_date, WEATHER_TYPES.GROUND)

# assign_to_farmer_and_fill_by_proximity(start_date, end_date, WEATHER_TYPES.PRECIPITATION)
# assign_to_farmer_and_fill_by_proximity(start_date, end_date, WEATHER_TYPES.SUNLIGHT)

# generate_interpolated_daydegree5_for_year(growth_season)

# interpolation_daydegree5_nn()


def first_run_processing(key):
    # Get all the sources from FROST
    get_frost_weather_sources_to_file(key)

    # Todo: Create a farmers dataset with all farms to use and their relevant information


def get_start_end_date(year):
    return f'{year}-03-01', f'{year}-10-01'


def get_and_process_feature_data(all_years, weather_feature):
    for growth_season in all_years:
        start_date, end_date = get_start_end_date(growth_season)

        # Download readings for each of the weather station sources
        get_raw_weather_readings_to_file(start_date, end_date, growth_season, weather_feature, client_id)

        # Process each reading in a day by day structure
        preprocess_weather_data(start_date, end_date, weather_feature)

        # Replace NaN values through closest neighbors or interpolation
        # Seccure that there are no NaN values

    # Train a NN for interpolation with correct farmers
    create_and_train_interpolation_nn(weather_feature)

    for growth_season in all_years:
        # Predict interpolation values
        generate_interpolated_feature(growth_season, weather_feature)


if __name__ == '__main__':
    client_id = 'c114a6ef-9081-4c42-b41b-0b3344a08ac4'
    secret = '8756c739-6d4e-47ad-b893-28d80b218df3'
    years = [2017, 2018, 2019, 2020, 2021]

    first_run_processing(client_id)

    get_and_process_feature_data(years, WEATHER_TYPES.SUNLIGHT)
    get_and_process_feature_data(years, WEATHER_TYPES.PRECIPITATION)

    add_feature_to_main_dataset(WEATHER_TYPES.SUNLIGHT)
    add_feature_to_main_dataset(WEATHER_TYPES.PRECIPITATION)


"""
Progress status of Frost data processing:

main_frost.py - Need to add the methods for each of the features, focusing on sunlight and precipitation for now.
frost_weather_sources.py - Completed
utils.py - Nothing to do here
raw_weather_readings.py - Completed
preprocess_weather_to_timeseries.py - Completed


"""
