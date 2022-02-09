import matplotlib.pyplot as plt
import numpy as np


def plot_bar(gdf, column):
    plt.figure()
    gdf.groupby(column)[column].count().plot(kind='bar')
    plt.show()


def highlight_optimized_natural_color(B):
    g = 0.6
    B = [B[3] * g, B[2] * g, B[1] * g]
    return [B[0]**(1/3) - 0.035, B[1]**(1/3) - 0.035, B[2]**(1/3) - 0.035]
    
true_color = lambda x: [x[3], x[2], x[1]]

def to_rgb(img, map_func=true_color, normalize=True):
    shape = img.shape
    newImg = []    
    for row in range(0, shape[0]):
        newRow = []
        for col in range(0, shape[1]):
            newRow.append(map_func(img[row][col]))
        newImg.append(newRow)
    if normalize:
        normImg = normalize_img(newImg, 1)
        return normImg
    else:
        return newImg

def plot_image(image):
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    #image = image/np.amax(image)
    #image = np.clip(image, 0, 1)
    ax.imshow(image)

    ax.set_xticks([])
    ax.set_yticks([])

def normalize_img(img, new_max):
    min = np.min(img)
    max = np.max(img)
    new_img = []
    for i, row in enumerate(img):
        new_row = []
        for j, pixel in enumerate(row):
            new_rgb = []
            for color in pixel:
                #zi = (xi – min(x)) / (max(x) – min(x)) * 1,000
                new_color = (color - min)/(max - min) * new_max
                new_rgb.append(new_color)
            new_row.append(new_rgb)
        new_img.append(new_row)
    return np.array(new_img)