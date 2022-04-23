import os
import h5py
import geopandas as gpd
from tqdm import tqdm
from shapely.geometry import box
from PIL import Image, ImageDraw

from src.mask.geo_point_translator import GeoPointTranslator
from src.utils import boundingBox, convert_crs


def create_mask_file(filename):
    if not os.path.exists(filename):
        with h5py.File(filename, "a") as file:
            file.create_group("masks")
    else:
        os.remove(filename)


def insert_mask(filename, key, mask):
    with h5py.File(filename, "a") as file:
        file.create_dataset(name=f"masks/{key}", data=mask, compression="gzip", compression_opts=2)


def generate_mask_image(bbox, polygon):
    y_max = 100
    x_max = 100
    mask_img = Image.new('1', (x_max, y_max), 0)

    geo_translator = GeoPointTranslator(bbox)

    shapes = [polygon.exterior.coords[:]]
    for shape in shapes:
        field_polygon = []
        for point in shape:
            xy = geo_translator.lat_lng_to_screen_xy(point[1], point[0])
            x = xy['x']
            y = xy['y']
            field_polygon.append((x, y_max - y))
        ImageDraw.Draw(mask_img).polygon(field_polygon, outline=1, fill=1)

    return mask_img


def insert_masks_to_h5(filename, data):
    keys_inserted = []
    for i, row in tqdm(data.iterrows(), total=data.shape[0], desc=f"Inserting masks..."):

        key = f"{int(row['orgnr'])}/{int(row['year'])}"
        if key not in keys_inserted:

            polygon = convert_crs([row['geometry']])[0]
            bbox = boundingBox(polygon.centroid.y, polygon.centroid.x, 1)
            bbox = box(bbox[0], bbox[1], bbox[2], bbox[3])
            mask_img = generate_mask_image(bbox, polygon)
            insert_mask(filename, key, mask_img)
            keys_inserted.append(key)


def main():
    filename_train = '../../../kornmo-data-files/raw-data/crop-classification-data/train_data_masks.h5'
    create_mask_file(filename_train)

    print("Reading training data")
    train_data_masks = gpd.read_file('../../../kornmo-data-files/raw-data/crop-classification-data/training_data.gpkg')

    insert_masks_to_h5(filename_train, train_data_masks)


    with h5py.File(filename_train, "a") as file:
        print(f"Inserted training data for {len(list(file['masks'].keys()))} farms")



if __name__ == '__main__':
    main()




