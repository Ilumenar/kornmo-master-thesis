
import h5py
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

def read_images(filename, split):
    sat_images = {}
    with h5py.File(os.path.join('E:/MasterThesisData', filename), "r") as f:
        images = f['images']
        for i, orgnum in enumerate(tqdm(images.keys(), total=split)):
            if(split == i):
                break
            img_dicts = {}
            for year in images[orgnum]:
                img = images[orgnum][year][()]
                img_dicts[year] = img
            sat_images[orgnum] = img_dicts
    f.close()
    return sat_images




