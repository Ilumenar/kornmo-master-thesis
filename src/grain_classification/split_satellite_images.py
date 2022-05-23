import geopandas as gpd
from tqdm import tqdm

from kornmo.sentinel.storage import SentinelDataset, SentinelDatasetIterator


data_path = '../../../kornmo-data-files/raw-data/crop-classification-data/'


def create_dataset(all_images: SentinelDataset, new_images: SentinelDataset, data):
    for i, row in tqdm(data.iterrows(), total=data.shape[0]):
        orgnr = int(row['orgnr'])
        year = int(row['year'])
        if all_images.contains(orgnr, year) and not new_images.contains(orgnr, year):
            imgs = all_images.get_images(orgnr, year)
            new_images.store_images(imgs, orgnr, year)


def main():
    sd = SentinelDataset('E:/MasterThesisData/Satellite_Images/combined_uncompressed.h5')
    print(f"Loaded {sd.__len__()} labels")
    train_data = gpd.read_file(f"{data_path}/training_data.gpkg")
    val_data = gpd.read_file(f"{data_path}/validation_data.gpkg")
    print(f"Loaded {train_data.shape[0]} training data and {val_data.shape[0]} validation data")

    sd_train = SentinelDataset('E:/MasterThesisData/Satellite_Images/satellite_images_train.h5', create_if_missing=True)
    sd_validation = SentinelDataset('E:/MasterThesisData/Satellite_Images/satellite_images_val.h5', create_if_missing=True)

    create_dataset(sd, sd_train, train_data)
    create_dataset(sd, sd_validation, val_data)


if __name__ == '__main__':
    main()
