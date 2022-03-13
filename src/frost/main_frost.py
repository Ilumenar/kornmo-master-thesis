import os

from interpolate_with_nn import *
from preprocess_weather_to_timeseries import preprocess_weather_data
from raw_weather_readings import get_raw_weather_readings_to_file
from utils import WEATHER_TYPES
from frost_weather_sources import get_frost_weather_sources_to_file
from add_features_to_dataset import add_feature_to_main_dataset
from singlevalue_interpolation_nn import create_and_train_singlevalue_interpolation_nn
from multivalue_interpolation_nn import create_and_train_multivalue_interpolation_nn
from scripts.create_farmers_coordinates import create_farmers_information_dataset
from interpolate_multivalue_with_nn import generate_interpolated_multivalue_for_year
from interpolate_singlevalue_with_nn import generate_interpolated_singlevalue_for_year
from nan_preprocessing import remove_nan_and_validate


def first_run_processing(key):
    # Getting all FROST weather sources
    if os.path.exists('../../../kornmo-data-files/raw-data/weather-data/frost_weather_sources.csv'):
        print(f"Dataset with all FROST weather sources already exists")
    else:
        print(f"Dataset with all FROST weather sources does not exist. Greating one now, hold on tight.")
        get_frost_weather_sources_to_file(key)

    # Getting all farmers with their geographic information
    if os.path.exists("../../../kornmo-data-files/raw-data/farm-information/all-farmers-with-location.csv"):
        print(f"Dataset for all unique farmers with coordinates already exists")
    else:
        print(f"Dataset for all unique farmers with coordinates does not exist. Greating one now, hold on tight.")
        create_farmers_information_dataset()


def find_lower_upper_bound(weather_feature):
    # Todo: Finish all bounds
    if weather_feature == WEATHER_TYPES.PRECIPITATION:
        return 0, 100

    elif weather_feature == WEATHER_TYPES.TEMPERATURE:
        return -30, 30

    elif weather_feature == WEATHER_TYPES.DAYDEGREE5:
        return 0, 1

    elif weather_feature == WEATHER_TYPES.DAYDEGREE0:
        return 0, 1

    elif weather_feature == WEATHER_TYPES.GROUND:
        return 0, 9

    elif weather_feature == WEATHER_TYPES.SUNLIGHT:
        return 0, 1

    else:
        print(f"Received unvalid weather feature ({weather_feature}) while getting lower and upper bounds")
        return 0, 0


def get_start_end_date(year):
    return f'{year}-03-01', f'{year}-10-01'


def get_and_process_feature_data(all_years, weather_feature):
    lower_bound, upper_bound = find_lower_upper_bound(weather_feature)

    """
    for growth_season in all_years:
        start_date, end_date = get_start_end_date(growth_season)

        # Download readings for each of the weather station sources
        get_raw_weather_readings_to_file(start_date, end_date, growth_season, weather_feature, client_id)

        # Process each reading in a day by day structure
        preprocess_weather_data(start_date, end_date, weather_feature)

        # Replace NaN values, clean and validate dataset
        remove_nan_and_validate(weather_feature, start_date, end_date)
    """

    # Train a NN for interpolation with correct farmers
    if weather_feature == WEATHER_TYPES.TEMPERATURE:
        create_and_train_multivalue_interpolation_nn(weather_feature, lower_bound, upper_bound)
    else:
        create_and_train_singlevalue_interpolation_nn(weather_feature, lower_bound, upper_bound)

    # Predict interpolation values
    for growth_season in all_years:
        if weather_feature == WEATHER_TYPES.TEMPERATURE:
            generate_interpolated_multivalue_for_year(growth_season, weather_feature, lower_bound, upper_bound)
        else:
            generate_interpolated_singlevalue_for_year(growth_season, weather_feature, lower_bound, upper_bound, 0)



if __name__ == '__main__':
    client_id = 'c114a6ef-9081-4c42-b41b-0b3344a08ac4'
    secret = '8756c739-6d4e-47ad-b893-28d80b218df3'
    years = [2017, 2018, 2019, 2020, 2021]

    # first_run_processing(client_id)

    create_and_train_singlevalue_interpolation_nn('sunlight', 0, 1)
    generate_interpolated_singlevalue_for_year(2017, 'sunlight', 0, 1, 0)



    # get_and_process_feature_data(years, WEATHER_TYPES.SUNLIGHT)
    # get_and_process_feature_data(years, WEATHER_TYPES.PRECIPITATION)
    # get_and_process_feature_data(years, WEATHER_TYPES.TEMPERATURE)
    # get_and_process_feature_data(years, WEATHER_TYPES.DAYDEGREE0)
    # get_and_process_feature_data(years, WEATHER_TYPES.DAYDEGREE5)
    # get_and_process_feature_data(years, WEATHER_TYPES.GROUND)

    # add_feature_to_main_dataset(WEATHER_TYPES.SUNLIGHT)
    # add_feature_to_main_dataset(WEATHER_TYPES.PRECIPITATION)


"""
Progress status of Frost data processing:

main_frost.py - Need to add the methods for each of the features, focusing on sunlight and precipitation for now.
frost_weather_sources.py - Completed
utils.py - Nothing to do here
raw_weather_readings.py - Completed
preprocess_weather_to_timeseries.py - Completed
create_farmers_coordinates.py - Completed
nan_preprocessing.py - Completed
singlevalue_interpolation_nn.py
multivalue_interpolation_nn.py
interpolate_singlevalue_with_nn.py
interpolate_multivalue_with_nn.py
weather_interpolation_utils.py - Completed

"""
