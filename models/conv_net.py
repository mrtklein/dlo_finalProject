from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, ReduceLROnPlateau
from keras.layers import Dropout, Dense
from keras.models import Sequential

from utils import Utils


class ConvNet:
    def __init__(self) -> None:
        self.utils = Utils()

    def get_model(self, img_height, img_width, summary=True):
        model = Sequential([
            Conv2D(16, (3, 3), 1, activation='relu',
                   input_shape=(img_height, img_width, 3)),
            MaxPooling2D(),
            # Dropout(0.2),
            Conv2D(32, (3, 3), 1, activation='relu'),
            MaxPooling2D(),
            Conv2D(16, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            # Dropout(0.2),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(4, activation='softmax')  # 4 categories as output channel
        ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        if summary:
            model.summary()

        return model

    def getCallBacks(self):
        # -------Callbacks-------------#
        #  checkpoints will be saved with the epoch number and the validation loss in the filename
        # best_model_weights = self.utils.getModelDirPath()+'weights.{epoch:02d}-{val_loss:.2f}.hdf5'

        best_model_weights = self.utils.getModelDirPath() + 'best_model.hdf5'

        log_dir = self.utils.getLogPath()

        checkpoint = ModelCheckpoint(
            best_model_weights,
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max',
            save_weights_only=False
        )
        earlystop = EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,
            patience=10,
            verbose=1,
            mode='auto',
            restore_best_weights=True
        )
        tensorboard = TensorBoard(
            log_dir=log_dir,
        )

        csvlogger = CSVLogger(
            filename="training_csv.log",
            separator=",",
            append=False
        )

        # lrsched = LearningRateScheduler(step_decay,verbose=1)
        #
        # reduce = ReduceLROnPlateau(
        #     monitor='val_loss',
        #     factor=0.5,
        #     patience=40,
        #     verbose=1,
        #     mode='auto',
        #     cooldown=1
        # )
        return [checkpoint, tensorboard, csvlogger], best_model_weights
