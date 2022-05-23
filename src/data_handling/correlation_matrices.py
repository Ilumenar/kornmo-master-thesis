import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def jordsmonn_to_numeric():
    soil_quality = pd.read_csv('../../../kornmo-data-files/raw-data/soil-data/jordsmonn.csv', low_memory=False)

    # Removing unuseful features
    soil_quality.drop([
        'Unnamed: 0', 'ORIGINALDA', 'ORIGINALDATAVERT', 'KOPIDATO', 'OBJTYPE', 'NAVNEROM', 'MÅLEMETODE', 'NØYAKTIGHET',
        'VERIFISERINGSDATO', 'fme_rejection_code', '_GEOMETRY_NAME', '_PART_NUMBER', 'KARTSIGNATUR', 'JORDKVALITET',
        'geometry'
    ], axis=1, inplace=True)

    # Changing NaN to 0
    soil_quality = soil_quality.fillna(0)

    # Change each non-numeric feature to numeric
    for feature in soil_quality.columns:
        print("Feature: " + feature)
        if soil_quality[feature].dtype != "float64" and soil_quality[feature].dtype != "int64":
            values = list(soil_quality[feature].unique())
            soil_quality[feature] = soil_quality[feature].apply(lambda x: values.index(x))

    # Check that they are all numeric
    for feature in soil_quality.columns:
        if soil_quality[feature].dtype != "float64" and soil_quality[feature].dtype != "int64":
            print("Found non-numeric feature")
            print(soil_quality[feature])

    soil_quality.to_csv('../../kornmo_old-data-files/raw-data/soil-data/jordsmonn-numeric.csv')
    print("Done converting dataset to numeric")


def correlation_matrix():
    soil_quality = pd.read_csv('../../../kornmo-data-files/raw-data/soil-data/jordsmonn-numeric.csv', low_memory=False)

    soil_quality.drop(['Unnamed: 0', 'DK_ASPARG', 'DK_BETER', 'DK_BLOMK', 'DK_BONNER', 'DK_GRAS_N',
        'DK_GRAS_V', 'DK_GULROT', 'DK_HODEK', 'DK_KALROT', 'DK_KINAK', 'DK_LOK', 'DK_MAIS', 'DK_MANDEL',
        'DK_POTET_N', 'DK_POTET_V', 'DK_PURRE', 'DK_ROSENK', 'DK_SALAT',
        'DK_SELLERI', 'DK_TIDLIG', 'DK_VORLOK', 'DK_VRAPS', 'DK_VRYBS', 'AA_ASPARG', 'AA_BETER', 'AA_BLOMK',
        'AA_BONNER', 'AA_GRAS_N',
        'AA_GRAS_V', 'AA_GULROT', 'AA_HODEK', 'AA_KALROT', 'AA_KINAK',
        'AA_LOK', 'AA_MAIS', 'AA_MANDEL',
        'AA_POTET_N', 'AA_POTET_V', 'AA_PURRE', 'AA_ROSENK', 'AA_SALAT',
        'AA_SELLERI', 'AA_TIDLIG', 'AA_VORLOK', 'AA_VRAPS', 'AA_VRYBS'], axis=1, inplace=True)

    matrix = soil_quality.corr().round(2)

    fig, ax = plt.subplots(figsize=(20, 20))
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    sns.heatmap(matrix, annot=True, mask=mask, vmax=1, vmin=-1, center=0, cmap='vlag')
    plt.show()

    print("Done plotting")


# jordsmonn_to_numeric()
correlation_matrix()
