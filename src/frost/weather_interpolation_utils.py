from pandas import DataFrame


def normalize(df, lower: float = None, upper: float = None) -> DataFrame:
    """
    :param df: The DataFrame where all columns will be normalized.
    :param lower: if present, together with upper, this value will correspond to the normalized value of 0.
    :param upper: if present, together with lower, this value will correspond to the normalized value of 1.
    :return: The new normalized DataFrame
    """

    if lower is None:
        lower = df.min()
    if upper is None:
        upper = df.max()

    return (df - lower) / (upper - lower)


def denormalize(df, lower: float, upper: float) -> DataFrame:
    """
    Denormalizes DataFrame
    :param df: The DataFrame where all columns will be denormalized.
    :param lower: The denormalized value of 0
    :param upper: The denormalized value of 1
    :return: The denormalized DataFrame
    """

    return df * (upper - lower) + lower


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
