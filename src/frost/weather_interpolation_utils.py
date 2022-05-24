from pandas import DataFrame
from kornmo.kornmo_utils import normalize, denormalize


def normalize_singlevalue_inputs(nn_input: DataFrame, lower, upper) -> DataFrame:
    for i in range(3):
        nn_input[f'{i}_masl_diff'] = normalize(nn_input[f'{i}_masl_diff'], -1000, 1000)
        nn_input[f'{i}_value'] = normalize(nn_input[f'{i}_value'], lower, upper)
    return nn_input


def normalize_singlevalue_actual(nn_actual: DataFrame, lower, upper) -> DataFrame:
    nn_actual['station_x_actual'] = normalize(nn_actual['station_x_actual'], lower, upper)
    return nn_actual


def normalize_multivalue_inputs(nn_input: DataFrame, lower, upper) -> DataFrame:
    for i in range(3):
        nn_input[f'{i}_masl_diff'] = normalize(nn_input[f'{i}_masl_diff'], -1000, 1000)
        nn_input[f'{i}_min'] = normalize(nn_input[f'{i}_min'], lower, upper)
        nn_input[f'{i}_mean'] = normalize(nn_input[f'{i}_mean'], lower, upper)
        nn_input[f'{i}_max'] = normalize(nn_input[f'{i}_max'], lower, upper)
    return nn_input


def normalize_multivalue_actual(nn_actual: DataFrame, lower, upper) -> DataFrame:
    nn_actual['station_x_min'] = normalize(nn_actual['station_x_min'], lower, upper)
    nn_actual['station_x_mean'] = normalize(nn_actual['station_x_mean'], lower, upper)
    nn_actual['station_x_max'] = normalize(nn_actual['station_x_max'], lower, upper)
    return nn_actual


def denormalize_prediction(nn_prediction: DataFrame, lower, upper) -> DataFrame:
    return denormalize(nn_prediction, lower, upper)


def get_number_of_days(readings):
    n_days = 0
    while True:
        if any(f'day_{n_days}' in col_name for col_name in readings.columns):
            n_days += 1
        else:
            return n_days
