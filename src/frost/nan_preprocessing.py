import pandas as pd
from sklearn.metrics import r2_score
import math
import numpy as np
import warnings
from utils import WEATHER_TYPES

warnings.simplefilter('ignore', np.RankWarning)

ACCEPTABLE_MISSING_PERCENTAGE = 0.75


def interpolate_row(row):
    x = list(range(1, len(row) + 1))
    y = row.tolist()
    vals_to_interpolate = []

    for i, val in enumerate(y[:]):
        if math.isnan(val):
            vals_to_interpolate.append(i)
            y.remove(val)
            x.remove(i + 1)
    if len(vals_to_interpolate) == 0:
        return

    r2_scores = []
    for degree in range(2, 20):
        model = np.poly1d(np.polyfit(x, y, degree))
        r2_scores.append(r2_score(y, model(x)))

    best_degree = r2_scores.index(max(r2_scores)) + 2
    best_model = np.poly1d(np.polyfit(x, y, best_degree))
    for val in vals_to_interpolate:
        interpolated_value = best_model(val)
        y.insert(val, interpolated_value)

    return pd.Series(y)


def remove_nan_and_validate(weather_type, start_date, end_date):
    processes_dataset = pd.read_csv(
        f'../../../kornmo-data-files/raw-data/weather-data/processed/{weather_type}/{weather_type}_processed_{start_date}_to_{end_date}.csv'
    )

    if weather_type == WEATHER_TYPES.TEMPERATURE:
        processes_dataset.fillna(value={"unit": "degC"}, inplace=True)

    measurement_columns = processes_dataset.filter(regex="day_.*").columns
    measurement_days = len(measurement_columns)

    for index, row in processes_dataset.iterrows():
        row = row.filter(regex="day_.*")
        missing = row.isna().sum().sum()
        coverage = missing / measurement_days

        if (row == 0).sum() + missing == measurement_days:
            processes_dataset.drop([index], inplace=True)

        elif coverage >= ACCEPTABLE_MISSING_PERCENTAGE:
            processes_dataset.drop([index], inplace=True)

        elif missing != 0:
            row.reset_index(drop=True, inplace=True)
            interpolated_row = interpolate_row(row)

            for i in range(measurement_days):
                processes_dataset.loc[index, measurement_columns[i]] = interpolated_row[i]

    if processes_dataset.isna().sum().sum() != 0:
        print(f"There are {processes_dataset.isna().sum().sum()} NaN values in the dataset for {weather_type} {start_date}")
        print(processes_dataset.isna().sum())
    else:
        processes_dataset.reset_index(drop=True, inplace=True)
        processes_dataset.to_csv(
            f'../../../kornmo-data-files/raw-data/weather-data/cleaned/{weather_type}/{weather_type}_processed_{start_date}_to_{end_date}.csv'
        )
        print(f"Dataset for {weather_type} {start_date} valid and cleared for interpolation")
