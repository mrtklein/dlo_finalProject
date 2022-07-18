from keras.applications.mobilenet import MobileNet
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, ReduceLROnPlateau
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.models import Sequential

from utils import Utils


class Pretrained_MobileNet:
    def get_model(self, img_height, img_width, summary=True):
        model_base = MobileNet(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
        model = Sequential()
        model.add(model_base)
        model.add(GlobalAveragePooling2D())
        model.add(Dropout(0.2))
        model.add(Dense(3, activation='softmax'))

        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        if summary:
            model.summary()

        return model

    def getCallBacks(self):
        # -------Callbacks-------------#
        best_model_weights = './base.model'
        utils = Utils()
        log_dir = utils.getLogPath()

        checkpoint = ModelCheckpoint(
            best_model_weights,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min',
            save_weights_only=False,
            period=1
        )
        earlystop = EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,
            patience=10,
            verbose=1,
            mode='auto'
        )
        tensorboard = TensorBoard(
            log_dir=log_dir,
            histogram_freq=0,
            batch_size=16,
            write_graph=True,
            write_grads=True,
            write_images=False,
        )

        csvlogger = CSVLogger(
            filename="training_csv.log",
            separator=",",
            append=False
        )

        # lrsched = LearningRateScheduler(step_decay,verbose=1)

        reduce = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=40,
            verbose=1,
            mode='auto',
            cooldown=1
        )
        return [checkpoint, tensorboard, csvlogger, reduce], best_model_weights
