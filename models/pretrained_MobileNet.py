from keras.applications.mobilenet import MobileNet
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, ReduceLROnPlateau
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.models import Sequential

from utils import Utils


class Pretrained_MobileNet:


    def __init__(self):
        self.utils = Utils()

    def get_model(self, img_height, img_width, summary=True):
        model_base = MobileNet(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
        model = Sequential()
        model.add(model_base)
        model.add(GlobalAveragePooling2D())
        model.add(Dropout(0.2))
        model.add(Dense(3, activation='softmax'))

        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        if summary:
            model.summary()

        return model

