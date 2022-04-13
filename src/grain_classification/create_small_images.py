from src.satellite_images.storage import SentinelDataset


def read_data():
    satellite_imgs = SentinelDataset('E:/MasterThesisData/Satellite_Images/satellite_images_train.h5')


def crop_img(img, left, top, size):
    new_img = []
    for i in range(left, left+size):
        row = []
        for j in range(top, top+size):
            row.append(img[i][j])
        new_img.append(row)
    return new_img
