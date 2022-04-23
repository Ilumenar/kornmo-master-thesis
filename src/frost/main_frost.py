import os
from preprocess_weather_to_timeseries import preprocess_weather_data
from raw_weather_readings import get_raw_weather_readings_to_file
from utils import WEATHER_TYPES
from frost_weather_sources import get_frost_weather_sources_to_file
from singlevalue_interpolation_nn import create_and_train_singlevalue_interpolation_nn
from multivalue_interpolation_nn import create_and_train_multivalue_interpolation_nn
from scripts.create_farmers_coordinates import create_farmers_information_dataset
from interpolate_multivalue_with_nn import generate_interpolated_multivalue_for_year
from interpolate_singlevalue_with_nn import generate_interpolated_singlevalue_for_year
from nan_preprocessing import remove_nan_and_validate

from find_measurements_by_proximity import assign_to_farmer_and_fill_by_proximity


def get_frost_sources_and_create_farmers_dataset(key):
    # Getting all FROST weather sources
    if os.path.exists('../../../kornmo-data-files/raw-data/weather-data/frost_weather_sources.csv'):
        print(f"Dataset with all FROST weather sources already exists")
    else:
        print(f"Dataset with all FROST weather sources does not exist. Greating one now, hold on tight.")
        get_frost_weather_sources_to_file(key)

    # Getting all farmers with their geographic information
    if os.path.exists("../../../kornmo-data-files/raw-data/farm-information/farmers-with-coordinates-and-soil_quality.csv"):
        print(f"Dataset for all unique farmers with coordinates already exists")
    else:
        print(f"Dataset for all unique farmers with coordinates does not exist. Greating one now, hold on tight.")
        create_farmers_information_dataset()


def find_lower_upper_bound(weather_feature):
    if weather_feature == WEATHER_TYPES.PRECIPITATION:
        return 0, 100

    elif weather_feature == WEATHER_TYPES.TEMPERATURE:
        return -30, 30

    elif weather_feature == WEATHER_TYPES.GROUND:
        return 0, 9

    else:
        return 0, 1


def get_start_end_date(year):
    return f'{year}-03-01', f'{year}-10-01'


def get_and_process_feature_data(all_years, weather_feature):
    for growth_season in all_years:
        start_date, end_date = get_start_end_date(growth_season)

        # Download readings for each of the weather station sources
        get_raw_weather_readings_to_file(start_date, end_date, growth_season, weather_feature, client_id)

        # Process each reading in a day by day structure
        preprocess_weather_data(start_date, end_date, weather_feature)


def clean_and_validate_dataset(all_years, weather_feature):
    for growth_season in all_years:
        start_date, end_date = get_start_end_date(growth_season)

        # Replace NaN values, clean and validate dataset
        remove_nan_and_validate(weather_feature, start_date, end_date)


def create_and_train_interpolation_nn(all_years, weather_feature):
    lower_bound, upper_bound = find_lower_upper_bound(weather_feature)

    # Train a NN for interpolation with correct farmers
    if weather_feature == WEATHER_TYPES.TEMPERATURE:
        create_and_train_multivalue_interpolation_nn(weather_feature, lower_bound, upper_bound, all_years)
    else:
        create_and_train_singlevalue_interpolation_nn(weather_feature, lower_bound, upper_bound, all_years)


def interpolate_measurements_by_distance(growth_season, weather_feature):
    lower_bound, upper_bound = find_lower_upper_bound(weather_feature)

    if weather_feature == WEATHER_TYPES.TEMPERATURE:
        generate_interpolated_multivalue_for_year(growth_season, weather_feature, lower_bound, upper_bound, 0)
    else:
        generate_interpolated_singlevalue_for_year(growth_season, weather_feature, lower_bound, upper_bound, 0)


def find_measurement_by_proximity(all_years, weather_feature):
    for growth_season in all_years:
        start_date, end_date = get_start_end_date(growth_season)
        assign_to_farmer_and_fill_by_proximity(start_date, end_date, weather_feature)


if __name__ == '__main__':
    client_id = 'c114a6ef-9081-4c42-b41b-0b3344a08ac4'
    secret = '8756c739-6d4e-47ad-b893-28d80b218df3'
    years = [2017, 2018, 2019, 2020]

    # Pick your desired weather type
    weather_type = WEATHER_TYPES.DAYDEGREE5

    # Download all frost sources, and create dataset of farmers to use.
    # get_frost_sources_and_create_farmers_dataset(client_id)

    # Download all measurements for each year, and procces them into timeseries
    # get_and_process_feature_data(years, weather_type)

    # Replace all Nan values and validate the dataset before further use
    # clean_and_validate_dataset(years, weather_type)

    # Create training dataset and train a NN for measurement interpolation
    # create_and_train_interpolation_nn(years, weather_type)

    # Predict the measurement through the NN for a specific year
    # interpolate_measurements_by_distance(2018, weather_type)

    # Or find measurements by proximity. Only used for Ground
    # find_measurement_by_proximity(years, weather_type)

