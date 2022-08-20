from keras.preprocessing.image import ImageDataGenerator
from utils import Utils


class DataLoader:

    def __init__(self):
        self.utils = Utils()

    def get_images(self, batch_size, img_height, img_width):
        data_dir = self.utils.getInputPath()

        # Data preprocessing
        # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
        # https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/
        # Augmentation Generation => it means we use a different transformation of each image in each epoch
        augs_gen = ImageDataGenerator(
            width_shift_range=0.2, # => random positive and negative Shifts. 20% of the width of the image
            height_shift_range=0.2,
            rotation_range=90,  # => random rotations via the rotation_range argument
            brightness_range=[0.5, 1.2], #=>  randomly darkening images, brightening images (<1.0=>darken,>1.0 brightness)
            horizontal_flip=False,  # => Randomly flip inputs horizontally.
            vertical_flip=False,
            zoom_range=0.2,  # zoom in or out in images,
            rescale=1. / 255,  # => Normalization
            validation_split=0.2,
        )

        # flow_from_directory
        # Return one Batch of augmented images foreach iteration. The labels are mapped by the corresponding subfolders

        # Training
        train_gen = augs_gen.flow_from_directory(
            data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42,  # random seed for shuffling and transformations.
            # save_to_dir=self.utils.getInputBatchesPath(), # => shows preprocessed images
            subset='training'
        )

        # Validation
        val_gen = augs_gen.flow_from_directory(
            data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False,
            subset='validation'
        )

        print("Labels: " + str(list(train_gen.class_indices.keys())))
        print("Number of train batches: " + str(len(train_gen)))
        print("Number of validation batches: " + str(len(val_gen)))

        return train_gen, val_gen

    def print_array_info(self, v):
        print("{} is of type {} with shape {} and dtype {}".format(v,
                                                                   eval("type({})".format(v)),
                                                                   eval("{}.shape".format(v)),
                                                                   eval("{}.dtype".format(v))
                                                                   ))
