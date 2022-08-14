from keras.preprocessing.image import ImageDataGenerator
from utils import Utils


class DataLoader:

    def __init__(self):
        self.utils = Utils()

    def get_images(self, batch_size, img_height, img_width):
        data_dir = self.utils.getInputPath()

        # Data preprocessing with https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
        augs_gen = ImageDataGenerator(
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1. / 255,
            horizontal_flip=True,
            vertical_flip=True,
            validation_split=0.2
        )

        train_gen = augs_gen.flow_from_directory(
            data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True,
            subset='training'
        )

        val_gen = augs_gen.flow_from_directory(
            data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False,
            subset='validation'
        )

        print("Labels: " + str(list(train_gen.class_indices.keys())))
        print("number_of_train_sample " + str(len(train_gen)))
        print("number_of_validation_sample " + str(len(val_gen)))

        return train_gen, val_gen

    def print_array_info(self, v):
        print("{} is of type {} with shape {} and dtype {}".format(v,
                                                                   eval("type({})".format(v)),
                                                                   eval("{}.shape".format(v)),
                                                                   eval("{}.dtype".format(v))
                                                                   ))
