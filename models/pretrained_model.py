from keras.applications.densenet import DenseNet201
from keras.applications.resnet import ResNet50
from keras.applications.vgg16 import VGG16
from keras.optimizer_v1 import SGD, Adam
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, ReduceLROnPlateau
from keras.layers import Dropout, Dense
from keras.models import Sequential, Model

from utils import Utils


class Pretrained_Model:
    def __init__(self) -> None:
        self.utils = Utils()

    def createModel(self, backbone, lr=1e-4, dense=128, drpout1=0.5, drpout2=0.25):
        """
        easily adjust the learning rate, the dense layer and the dropout layer.

        :param backbone:
        :param lr:
        :param dense:
        :param drpout1:
        :param drpout2:
        :return:
        """
        EPOCHS = 50
        INIT_LR = 1e-1
        BS = 128
        model = Sequential()
        model.add(backbone)
        model.add(Dropout(drpout1))
        model.add(Dense(dense, activation='relu'))
        #     model.add(LeakyReLU(alpha=0.1))
        #     model.add(BatchNormalization())
        model.add(Dropout(drpout2))
        model.add(Dense(4, activation='softmax'))

        model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=Adam(learning_rate=lr),
            metrics=['accuracy']
        )

        return model

    def createBackboneModel(self, denseNet=False, resnet=False, vgg=False, ):
        """
        For the purpose of this project, the VGG-16 model is selected and instantiated with weights trained on ImageNet.
        The top layer which is the classification layer is excluded and the input shape of the image is set to 224x224x3
        :param denseNet:
        :param resnet:
        :param vgg:
        :return:
        """
        if denseNet:
            backbone = DenseNet201(weights='imagenet', include_top=False,
                                   input_shape=(224, 224, 3))
        if resnet:
            backbone = ResNet50(weights='imagenet', include_top=False,
                                input_shape=(224, 224, 3))
        if vgg:
            backbone = VGG16(weights='imagenet', include_top=False,
                             input_shape=(224, 224, 3))
            output = backbone.layers[-1].output
        output = keras.layers.Flatten()(output)
        backboneModel = Model(backbone.input, outputs=output)
        for layer in backboneModel.layers:
            layer.trainable = False
            return backboneModel

    def get_model(self, img_height, img_width, summary=True):
        model = Sequential([
            Conv2D(16, (3, 3), 1, activation='relu',
                   input_shape=(img_height, img_width, 3)),  # => Output Feature Maps 222x222x3
            MaxPooling2D(),
            # Dropout(0.2),
            Conv2D(32, (3, 3), 1, activation='relu'),
            MaxPooling2D(),
            Conv2D(16, 3, padding='same', activation='relu'),
            # "same" results in padding with zeros evenly to the left/right or up/down of the input. When padding="same" and strides=1, the output has the same size as the input.
            MaxPooling2D(),
            # Dropout(0.2),
            Flatten(),
            # Vollständig verbundene Schicht mit einer ReLU-Aktivierungsfunktion
            # hinzufügen
            Dense(256, activation='relu'),
            # Vollständig verbundene Schicht mit einer Sigmoid-Aktivierungsfunktion hinzufügen
            Dense(4, activation='softmax')  # 4 categories as output channel
        ])

        opt = SGD(lr=1e-4, momentum=0.99)
        opt1 = Adam(lr=2e-4)

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
            patience=20,
            verbose=1,
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
        return [checkpoint, earlystop, tensorboard, csvlogger], best_model_weights
