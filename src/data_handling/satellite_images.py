
import h5py
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

def read_images(filename, split):
    sat_images = {}
    with h5py.File(os.path.join('E:/MasterThesisData/Satellite_Images', filename), "r") as f:
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

def read_sat_images_file(filename):
    images = {}
    with h5py.File(os.path.join('E:/MasterThesisData/Satellite_Images', filename), "r") as f:
        h5images = f['images']
        keys = list(h5images.keys())
        for orgnr in tqdm(keys):
            images[orgnr] = len(h5images[orgnr].keys())
    f.close()

    return images

def get_images_by_orgnr(orgnr):
    sat_images = {}
    for filename in ['sentinel_100x100_0.h5', 'sentinel_100x100_1.h5']:
        with h5py.File(os.path.join('E:/MasterThesisData/Satellite_Images', filename), "r") as f:
            if orgnr in list(f['images'].keys()):
                images = f['images'][orgnr]
                for year in images:
                    img = images[year][()]
                    sat_images[year] = img
    f.close()
    return sat_images


