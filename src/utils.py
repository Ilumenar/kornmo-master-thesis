import matplotlib.pyplot as plt

def plot_bar(gdf, column):
    plt.figure()
    gdf.groupby(column)[column].count().plot(kind='bar')
    plt.show()