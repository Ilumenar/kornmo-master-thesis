from storage import SentinelDataset
import os
from src.utils import to_rgb
import matplotlib.pyplot as plt
import numpy as np
from satellite_images import read_sat_images_file

if __name__ == "__main__":
    sat_img_path = 'E:/MasterThesisData/Satellite_Images/'
    file_name = 'sentinel_100x100_1.h5'
    print("Creating dataset. This might take some time")
    sd = SentinelDataset(os.path.join(sat_img_path, file_name))

    # Checking what orgnrs exists in file
    images = read_sat_images_file(file_name)
    print(images.keys())

    orgnr = 811555762
    year = 2017
    if sd.contains(orgnr, year):
        data = sd.get_images(orgnr, year)
        data = [to_rgb(img) for img in data]
        ncols = 5
        nrows = 6
        aspect_ratio = 1
        subplot_kw = {'xticks': [], 'yticks': [], 'frame_on': False}
        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5 * ncols * aspect_ratio, 5 * nrows), subplot_kw=subplot_kw)

        for idx, image in enumerate(data):
            ax = axs[idx // ncols][idx % ncols]
            ax.imshow(np.clip(image, 0, 1))

        plt.tight_layout()
        plt.show()
    else:
        print(f"Dataset {file_name} does not contain images for {orgnr} for year {year}")
