import pandas as pd
from sklearn.metrics import r2_score
import math
import numpy as np
import warnings
from tqdm import tqdm
from utils import WEATHER_TYPES
import matplotlib.pyplot as plt


warnings.simplefilter('ignore', np.RankWarning)

ACCEPTABLE_MISSING_PERCENTAGE = 0.70


def interpolate_row(row):
    row_min = min(row)
    row_max = max(row)

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

    # best_degree = r2_scores.index(max(r2_scores)) + 2
    print(r2_scores[1])
    best_degree = 3
    best_model = np.poly1d(np.polyfit(x, y, best_degree))
    for val in vals_to_interpolate:
        interpolated_value = best_model(val)
        y.insert(val, interpolated_value)

    y = pd.Series(y)
    new_row_min = min(y)
    new_row_max = max(y)
    if row_min != new_row_min or row_max != new_row_max:
        print(f"Found unequal values. Min: {row_min}, {new_row_min}. Max: {row_max}, {new_row_max}. Using degree: {best_degree}")

    else:
        print(f"Correctly used degree: {best_degree}")

    return y


def get_random_value(row):
    data = row.dropna().tolist()
    num_features = len(set(data))

    count, _, _ = plt.hist(data, num_features)
    sum_count = np.sum(count)
    probabilities = [x / sum_count for x in count]

    random_value = np.random.choice(np.arange(0, num_features), p=probabilities)

    return random_value


def remove_nan_and_validate(weather_type, start_date, end_date):
    processes_dataset = pd.read_csv(
        f'../../../kornmo-data-files/raw-data/weather-data/processed/{weather_type}/{weather_type}_processed_{start_date}_to_{end_date}.csv'
    )

    if weather_type == WEATHER_TYPES.TEMPERATURE:
        processes_dataset.fillna(value={"unit": "degC"}, inplace=True)

    measurement_columns = processes_dataset.filter(regex="day_.*").columns
    measurement_days = len(measurement_columns)

    min_value = min(processes_dataset.filter(regex="day_.*"))
    max_value = min(processes_dataset.filter(regex="day_.*"))

    p_bar = tqdm(processes_dataset.iterrows())

    for index, row in p_bar:
        p_bar.set_description(f"Calculating for index {index}")
        row = row.filter(regex="day_.*")
        missing = row.isna().sum().sum()
        missing_percentage = missing / measurement_days

        if (row == 0).sum() + missing == measurement_days:
            processes_dataset.drop([index], inplace=True)

        elif missing_percentage >= ACCEPTABLE_MISSING_PERCENTAGE:
            processes_dataset.drop([index], inplace=True)

        elif missing != 0:
            row.reset_index(drop=True, inplace=True)

            for i in range(row.isna().sum().sum()):
                new_value = get_random_value(row)
                row.fillna(value=new_value, limit=1, inplace=True)

            for i in range(measurement_days):
                processes_dataset.loc[index, measurement_columns[i]] = row[i]

    new_min_value = min(processes_dataset.filter(regex="day_.*"))
    new_max_value = min(processes_dataset.filter(regex="day_.*"))

    if processes_dataset.isna().sum().sum() != 0:
        print(f"There are {processes_dataset.isna().sum().sum()} NaN values in the dataset for {weather_type} {start_date}")
        print(processes_dataset.isna().sum())

    elif min_value != new_min_value or max_value != new_max_value:
        print(f"Min or Max not equal after interpolation. Min: {min_value} != {new_min_value}. Max: {max_value} != {new_max_value}")

    else:
        processes_dataset.reset_index(drop=True, inplace=True)
        processes_dataset.to_csv(
            f'../../../kornmo-data-files/raw-data/weather-data/cleaned/{weather_type}/{weather_type}_cleaned_{start_date}_to_{end_date}.csv'
        )
        print(f"Dataset for {weather_type} {start_date} valid and cleared for interpolation")
