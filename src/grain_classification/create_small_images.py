import math
import sys

import geopandas as gpd
import os
import h5py
from PIL import Image, ImageDraw
from shapely.geometry import box
from tqdm import tqdm

from src.mask.geo_point_translator import GeoPointTranslator
from src.satellite_images.storage import SentinelDataset
from src.utils import boundingBox, convert_crs


NEW_IMAGES_PATH = 'E:/MasterThesisData/Satellite_Images/small_images_all.h5'
# NEW_MASKS_PATH = 'E:/MasterThesisData/Satellite_Images/small_masks_train.h5'
SATELLITE_IMAGES_PATH = 'E:/MasterThesisData/Satellite_Images/combined_uncompressed.h5'
POLYGONS_PATH = '../../../kornmo-data-files/raw-data/crop-classification-data/all_data.gpkg'


def create_h5_file(filename):
    if not os.path.exists(filename):
        with h5py.File(filename, "a") as file:
            file.create_group("images")
    else:
        os.remove(filename)
        with h5py.File(filename, "a") as file:
            file.create_group("images")


def insert_data(filename, key, data):
    with h5py.File(filename, "a") as file:
        file.create_dataset(name=key, data=data, compression="gzip", compression_opts=2)


def read_data():
    satellite_imgs = SentinelDataset(SATELLITE_IMAGES_PATH)
    training_polys = gpd.read_file(POLYGONS_PATH)
    centroid_coords = gpd.read_file('../../../kornmo-data-files/raw-data/farm-information/farm-properties/bounding-boxes-previous-students/disponerte_eiendommer_bboxes.shp')
    return satellite_imgs, centroid_coords, training_polys


def crop_img(img, center_x, center_y, size):
    left = center_x - size / 2
    top = center_y - size / 2

    if center_x - size / 2 < 0:
        offset_x = 0 - (center_x - size/2)
        left = center_x - size/2 + offset_x
    elif center_x + size/2 >= 100:
        offset_x = (center_x + size/2) - 100
        left = center_x - size / 2 - offset_x

    if center_y - size / 2 < 0:
        offset_y = 0 - (center_y - size / 2)
        top = center_y - size / 2 + offset_y
    elif center_y + size / 2 >= 100:
        offset_y = (center_y + size / 2) - 100
        top = center_y - size / 2 - offset_y

    left = int(left)
    top = int(top)
    new_img = []
    for i in range(left, left+size):
        row = []
        for j in range(top, top+size):
            row.append(img[i][j])
        new_img.append(row)
    return new_img


def generate_mask(polygon, size=16):
    y_max, x_max = size, size
    mask_img = Image.new('1', (x_max, y_max), 0)

    bbox = boundingBox(polygon.centroid.y, polygon.centroid.x, 1)
    bbox_poly = box(bbox[0], bbox[1], bbox[2], bbox[3])
    point_translator = GeoPointTranslator(bbox_poly, size)

    shapes = [polygon.exterior.coords[:]]
    for shape in shapes:
        field_polygon = []
        for point in shape:
            xy = point_translator.lat_lng_to_screen_xy(point[1], point[0])
            x = xy['x']
            y = xy['y']
            field_polygon.append((x, y_max - y))
        ImageDraw.Draw(mask_img).polygon(field_polygon, outline=1, fill=1)

    return mask_img


def crop_images(images, polygons, coords, orgnr, year, size=16):

    bbox = boundingBox(coords.centroid.y, coords.centroid.x, 1)
    bbox_poly = box(bbox[0], bbox[1], bbox[2], bbox[3])
    point_translator = GeoPointTranslator(bbox_poly)
    for i, row in polygons.iterrows():
        polygon = convert_crs([row['geometry']])[0]
        cropped_images_per_polygon = []
        point = point_translator.lat_lng_to_screen_xy(polygon.centroid.y, polygon.centroid.x)
        for image in images:
            cropped_image = crop_img(image, math.floor(point['x']), math.floor(point['y']), size)
            cropped_images_per_polygon.append(cropped_image)

        # mask = generate_mask(polygon)
        # cropped_mask = crop_img(np.asarray(mask), math.floor(point['x']), math.floor(point['y']), size)
        # Image.fromarray(np.array(mask)).show()

        # insert_data(NEW_MASKS_PATH, f'masks/{orgnr}{i}/{year}', mask)
        insert_data(NEW_IMAGES_PATH, f'images/{orgnr}{i}/{year}', cropped_images_per_polygon)


def create_small_images(satellite_images, centroid_coords, training_polys):
    for data in tqdm(satellite_images.to_iterator(), total=satellite_images.__len__()):
        orgnr = int(data[0])
        year = int(data[1])
        try:
            # print(orgnr)
            images = satellite_images.get_images(orgnr, year)
            polygons = training_polys.loc[training_polys['orgnr'] == orgnr].loc[training_polys['year'] == year]
            if orgnr in set(centroid_coords['orgnr']):
                coords = convert_crs([centroid_coords.loc[centroid_coords['orgnr'] == orgnr]['geometry'].iloc[0]])[0]
                crop_images(images, polygons, coords, orgnr, year)
        except IndexError as e:
            print(orgnr)
            sys.exit(0)
        # insert_images(NEW_IMAGES_PATH, f'{orgnr}/{year}', cropped_images)


def main():
    satellite_images, centroid_coords, training_polys = read_data()

    create_h5_file(NEW_IMAGES_PATH)
    # create_h5_file(NEW_MASKS_PATH)

    create_small_images(satellite_images, centroid_coords, training_polys)


if __name__ == '__main__':
    main()
