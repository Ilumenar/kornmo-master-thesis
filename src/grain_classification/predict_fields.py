from tqdm import tqdm
from keras.models import load_model
import pandas as pd
import tensorflow as tf
from src.satellite_images.storage import SentinelDataset
from src.utils import to_rgb

PREDICTED_VALUES_PATH = '../../../kornmo-data-files/raw-data/crop-classification-data/week_1_16/predicted_values.csv'
MODEL_PATH = 'training/models/classification_1-16.hdf5'
CLASSES = ['bygg', 'rug', 'havre', 'rughvete', 'hvete']


def read_data():
    all_fields = SentinelDataset('E:/MasterThesisData/Satellite_Images/small_images_all.h5')
    model = load_model('training/models/classification_1-16.hdf5')
    all_fields_it = all_fields.to_iterator()

    def val_generator():
        for orgnr, year, imgs, label in all_fields_it:
            imgs = imgs[0:16]
            yield imgs

    val_dataset = tf.data.Dataset.from_generator(
        val_generator,
        output_types=tf.dtypes.float64,
        output_shapes=(16, 16, 16, 12)
    )

    return val_dataset, model


def predict_values(val_dataset, model, day):
    # for data in tqdm(fields.to_iterator(), total=fields.__len__()):
    #     orgnr = int(data[0])
    #     year = int(data[1])
    #     images = data[2][()][0:day]
    #     print(images.shape)
    #     model.summary()
    #     crop_type = model.predict(images)
    #     print(crop_type)
    #     break
    predicted_values = model.predict(val_dataset.batch(32).prefetch(2), verbose=1)


def main():
    print("Reading data...")
    val_dataset, model = read_data()
    predict_values(val_dataset, model, 16)


if __name__ == '__main__':
    main()
