from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow import keras
from utils import Utils
import numpy as np
from sklearn.model_selection import train_test_split


class DataLoader:

    def __init__(self):
        self.utils = Utils()

    def get_images(self, train_dir, img_height, img_width):

        # The images in the dataset are not used directly. Instead, only augmented images are provided to the model.
        data = keras.utils.image_dataset_from_directory(self.utils.getInputPath(), shuffle=True, image_size=(img_height, img_width))

        data_iterator = data.as_numpy_iterator()
        batch = data_iterator.next

        # normalization and map label
        data = data.map(lambda x, y: (x / 255, y))
        data.as_numpy_iterator().next()

        train_size = int(len(data) * .7)
        val_size = int(len(data) * .2)
        test_size = int(len(data) * .1)

        train = data.take(train_size)
        val = data.skip(train_size).take(val_size)
        test = data.skip(train_size + val_size).take(test_size)

        return train, val, test

    def preprocessImages(self, data_set):
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.flow(data_set)

        return datagen

    def print_array_info(v):
        print("{} is of type {} with shape {} and dtype {}".format(v,
                                                                   eval("type({})".format(v)),
                                                                   eval("{}.shape".format(v)),
                                                                   eval("{}.dtype".format(v))
                                                                   ))
