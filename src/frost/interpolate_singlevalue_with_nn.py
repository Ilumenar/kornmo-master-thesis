import pandas as pd
import numpy as np
from utils import distance
from keras.models import load_model
import weather_interpolation_utils as wiu
from tqdm import tqdm
from utils import WEATHER_TYPES


def get_number_of_days(readings):
    n_days = 0
    while True:
        if any(f'day_{n_days}' in col_name for col_name in readings.columns):
            n_days += 1
        else:
            return n_days


def get_k_closest_stations_singlevalue(sensors: pd.DataFrame, distances: pd.DataFrame, lat, lng, masl, k: int, weather_feature):
    if weather_feature == WEATHER_TYPES.SUNLIGHT:
        closest = sensors.merge(distances, how='outer', left_on='station_id', right_on='id').head(k)
    else:
        closest = distances.head(5 * k).merge(sensors, how='inner', left_on='id', right_on='station_id').head(k)

    if len(closest) < k:
        print(f"Could not find {k} stations: {closest}")

    series = pd.Series(dtype='float64')
    station_count = 0
    for station_id, station in closest.iterrows():
        series[f'{station_count}_lat_diff'] = station.lat - lat
        series[f'{station_count}_lng_diff'] = station.lng - lng
        series[f'{station_count}_masl_diff'] = station.masl - masl
        series[f'{station_count}_value'] = station.value
        station_count += 1

    return series


def generate_interpolated_singlevalue_for_year(growth_season, weather_feature, lower_bound, upper_bound, starting_index):

    singlevalue_model = load_model(f'nn_interpolation_models/{weather_feature}_model.h5')
    # Tensorflow outputs some garbage on the first use, which ruins the progress bars, so let's get it over with.
    singlevalue_model.predict(np.zeros(shape=(1, 12)))

    readings = pd.read_csv(f"../../../kornmo-data-files/raw-data/weather-data/processed/{weather_feature}/{weather_feature}_processed_{growth_season}-03-01_to_{growth_season}-10-01.csv")
    sensors = pd.read_csv(f"../../../kornmo-data-files/raw-data/weather-data/frost_weather_sources.csv", index_col="id")[['lng', 'lat', 'masl']]
    readings = readings.join(sensors, "station_id")
    readings = readings.reset_index()

    farmers = pd.read_csv(f"../../../kornmo-data-files/raw-data/farm-information/all-farmers-with-location.csv")[['orgnr', 'longitude', 'latitude', 'elevation']]

    farmer_distances = {}
    p_bar = tqdm(farmers.iterrows(), total=len(farmers))
    p_bar.set_description("Calculating distances")
    for idx, farmer in p_bar:
        farmer_sensors = sensors.copy()
        farmer_sensors['distance'] = sensors.apply(lambda ws: distance((farmer.latitude, farmer.longitude), (ws.lat, ws.lng)), axis=1)
        farmer_sensors = farmer_sensors.sort_values(by=['distance'])
        farmer_distances[farmer.orgnr] = farmer_sensors[['distance']]

    n_days = get_number_of_days(readings)

    if starting_index != 0:
        farmers = pd.read_csv(f"../../../kornmo-data-files/raw-data/weather-data/nn_interpolated/{weather_feature}/{weather_feature}_interpolated_{growth_season}-03-01_to_{growth_season}-10-01.csv")

    for day in range(starting_index, n_days):
        p_bar.set_description_str(f"Interpolating {weather_feature} {growth_season}, day {day} of {n_days - 1}")
        readings_for_day = readings[["station_id", "lat", "lng", "masl", f"day_{day}"]]
        readings_for_day = readings_for_day.rename(columns={f"day_{day}": "value"})

        nn_input = pd.DataFrame()
        for index, farmer in farmers.iterrows():
            closest_sensors = get_k_closest_stations_singlevalue(
                readings_for_day,
                farmer_distances[farmer.orgnr],
                farmer.latitude,
                farmer.longitude,
                farmer.elevation,
                3,
                weather_feature
            )

            nn_input = pd.concat([nn_input, closest_sensors.to_frame().T], ignore_index=True)

        nn_input = wiu.normalize_singlevalue_inputs(nn_input, lower_bound, upper_bound)
        nn_prediction = singlevalue_model.predict(nn_input.to_numpy())
        farmers[f'day_{day}'] = wiu.denormalize_prediction(nn_prediction.flatten(), lower_bound, upper_bound)

        farmers.to_csv(f"../../../kornmo-data-files/raw-data/weather-data/nn_interpolated/{weather_feature}/{weather_feature}_interpolated_{growth_season}-03-01_to_{growth_season}-10-01.csv", float_format='%.1f')
        print(f"Done interpolating {weather_feature} {growth_season}, day {day} of {n_days - 1}")

    farmers.to_csv(f"../../../kornmo-data-files/raw-data/weather-data/nn_interpolated/{weather_feature}/{weather_feature}_interpolated_{growth_season}-03-01_to_{growth_season}-10-01.csv", float_format='%.1f')
    print(f"Done with interpolating all values for {weather_feature} in {growth_season}")
