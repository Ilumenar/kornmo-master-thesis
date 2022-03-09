import h5py
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image
img_location = 'E:/MasterThesisData/Satellite_Images'

def read_images(filename, split=-1):
    sat_images = {}
    with h5py.File(os.path.join(img_location, filename), "r") as f:
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
    with h5py.File(os.path.join(img_location, filename), "r") as f:
        h5images = f['images']
        keys = list(h5images.keys())
        for orgnr in tqdm(keys):
            images[orgnr] = len(h5images[orgnr].keys())
    f.close()

    return images

def get_images_by_orgnr(orgnr):
    sat_images = {}
    for filename in ['sentinel_100x100_0.h5', 'sentinel_100x100_1.h5']:
        with h5py.File(os.path.join(img_location, filename), "r") as f:
            if orgnr in list(f['images'].keys()):
                images = f['images'][orgnr]
                for year in images:
                    img = images[year][()]
                    sat_images[year] = img
    f.close()
    return sat_images


def read_jordsmonn_h5():
    for filename in ['nibio_disposed_properties_masks.h5']: # , 'nibio_jordsmonn_100x100.h5']:
        with h5py.File(os.path.join('../../../kornmo-data-files/raw-data/farm-information/farm-properties', filename), "r") as f:
            all_orgnr = {}
            print(f.keys())

            h5images = f['masks']
            keys = list(h5images.keys())

            for orgnr in tqdm(keys):
                all_orgnr[orgnr] = orgnr

            # orgnr = ["811675792", "812075322", "812686992", "812856472"]
            orgnr = ["970391959"]

            for nr in orgnr:
                if nr in list(f['masks'].keys()):
                    images = f['masks'][nr]
                    year = "2019"
                    img = images[year][()]
                    img = (img * 255).astype(np.uint8)
                    i = Image.fromarray(img)
                    i.show()

            f.close()


read_jordsmonn_h5()

