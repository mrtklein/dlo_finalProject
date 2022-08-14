from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from utils import Utils
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import random
import numpy as np
from sklearn.model_selection import train_test_split


class DataLoader:

    def __init__(self):
        self.utils = Utils()
        self.X_train = []
        self.X_val = []
        self.y_train = []
        self.y_val = []
        self.labels = ['paper', 'scissors', 'rock', 'rest']

    def get_images(self, train_dir, img_height, img_width):

        dataset = []
        count = 0
        for label in self.labels:
            folder = os.path.join(train_dir, label)
            for image in os.listdir(folder):
                img = load_img(os.path.join(folder, image), target_size=(img_height, img_width))
                img = img_to_array(img)
                img = img / 255.0
                dataset.append((img, count))
            print(f'\rCompleted: {label}', end='')
            count += 1
        random.shuffle(dataset)
        X, y = zip(*dataset)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(np.array(X), np.array(y),
                                                                              test_size=0.22, random_state=42)
        print("Größe der Trainingsdaten:" + np.unique(self.y_train, return_counts=True),
              "Größe der Validierungsdaten:" + np.unique(self.y_val, return_counts=True))

        return self.X_train, self.X_val, self.y_train, self.y_val

    def preprocessImages(self):
        datagen = ImageDataGenerator(horizontal_flip=True,
                                     vertical_flip=True,
                                     rotation_range=20,
                                     zoom_range=0.2,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.1,
                                     fill_mode="nearest")

        testgen = ImageDataGenerator()

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(self.X_train)
        testgen.fit(self.X_val)

        nb = len(self.labels)

        y_train = np.eye(nb)[self.y_train]
        y_val = np.eye(nb)[self.y_val]

        return datagen, testgen, y_train, y_val

    def print_array_info(v):
        print("{} is of type {} with shape {} and dtype {}".format(v,
                                                                   eval("type({})".format(v)),
                                                                   eval("{}.shape".format(v)),
                                                                   eval("{}.dtype".format(v))
                                                                   ))
