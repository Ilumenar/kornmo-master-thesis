import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import utils
import weather_interpolation_utils as wiu
from keras.layers import Dense, Dropout
from keras.models import Sequential
import matplotlib.pyplot as plt
import tensorflow as tf


def get_k_closest(sensors: pd.DataFrame, lat, lng, k: int):
    sensors['distance'] = sensors.apply(
        lambda ws: utils.distance((lat, lng), (ws.lat, ws.lng)), axis=1
    )

    new_sensors = sensors.sort_values(by=['distance']).head(k)
    if len(new_sensors) != k:
        print(f"Did not find K closest sensors for sensors: {sensors}")
    return new_sensors


def make_singlevalue_dataset_entry(lat, lng, masl, actual, closest: pd.DataFrame) -> pd.Series:
    day_series = pd.Series(dtype='float64')
    station_count = 0
    for station_id, station in closest.iterrows():
        day_series[f'{station_count}_lat_diff'] = station.lat - lat
        day_series[f'{station_count}_lng_diff'] = station.lng - lng
        day_series[f'{station_count}_masl_diff'] = station.masl - masl
        day_series[f'{station_count}_value'] = station.value
        station_count += 1
    day_series['station_x_actual'] = actual

    return day_series


def create_singlevalue_training_data(weather_feature) -> pd.DataFrame:
    frost_sources = pd.read_csv('../../../kornmo-data-files/raw-data/weather-data/frost_weather_sources.csv', index_col=['id'])

    df: pd.DataFrame = pd.DataFrame()

    for year in range(2017, 2021):
        new_sensors: pd.DataFrame = pd.read_csv(f'../../../kornmo-data-files/raw-data/weather-data/processed/{weather_feature}/{weather_feature}_processed_{year}-03-01_to_{year}-10-01.csv', index_col='station_id').dropna()

        new_sensors = new_sensors.join(frost_sources, 'station_id')

        for day in range(214):
            print(f'day {day}, {year}')
            for station_id, station in new_sensors.iterrows():
                day_sensors = new_sensors[["lat", "lng", "masl", f"day_{day}"]]
                day_sensors = day_sensors.rename(columns={f"day_{day}": "value"}).drop(station_id)
                closest = get_k_closest(day_sensors, station.lat, station.lng, 3)
                series: pd.Series = make_singlevalue_dataset_entry(
                    station.lat,
                    station.lng,
                    station.masl,
                    station[f'day_{day}'],
                    closest
                )

                df = pd.concat([df, series.to_frame().T], ignore_index=True)

    if df.isna().sum().sum() != 0:
        print(f"There are {df.isna().sum().sum()} NaN values in the dataset for {weather_feature}")
        print(df.isna().sum())

    df.to_csv(f'nn_interpolation_models/{weather_feature}_training_data.csv')

    return df


def train_singlevalue_interpolation_model(train_x, train_y, val_x, val_y, weather_feature):
    model = Sequential()
    model.add(Dense(units=512, activation="relu", input_dim=len(train_x[0])))
    model.add(Dropout(0.2))
    model.add(Dense(units=512, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=32, activation="relu"))
    model.add(Dense(units=1))

    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, min_delta=0)
    history = model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        callbacks=[early_stopping],
        batch_size=4096,
        epochs=1000,
        verbose=2,
    )

    model.save(f'nn_interpolation_models/{weather_feature}_model.h5')
    print(f"Interpolation model for {weather_feature} is trained and saved to file")

    plt.xlabel('Epoch')
    plt.ylabel("Loss")
    plt.plot(history.history['loss'], label="loss")
    plt.plot(history.history['val_loss'], label="val_loss")
    plt.legend()
    plt.show()

    return model


def create_and_train_singlevalue_interpolation_nn(weather_feature, lower_bound, upper_bound):

    data = create_singlevalue_training_data(weather_feature)

    data = wiu.normalize_singlevalue_inputs(data, lower_bound, upper_bound)
    data = wiu.normalize_singlevalue_actual(data, lower_bound, upper_bound)

    train, val = train_test_split(shuffle(data), test_size=0.2)
    val, test = train_test_split(val, test_size=0.2)
    train_x = train.drop('station_x_actual', axis=1).to_numpy()
    train_y = train['station_x_actual'].to_numpy()

    val_x = val.drop('station_x_actual', axis=1).to_numpy()
    val_y = val['station_x_actual'].to_numpy()

    model = train_singlevalue_interpolation_model(train_x, train_y, val_x, val_y, weather_feature)
