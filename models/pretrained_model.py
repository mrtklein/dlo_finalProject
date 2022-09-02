from keras.applications.densenet import DenseNet201
from keras.applications.resnet import ResNet50
from keras.applications.vgg16 import VGG16
from keras.optimizer_v2.adam import Adam
from keras.optimizer_v2.gradient_descent import SGD
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

    def getModel(self, backbone, lr=1e-4, dense=128, drpout1=0.5, drpout2=0.25):
        """
        easily adjust the learning rate, the dense layer and the dropout layer.

        :param backbone:
        :param lr:
        :param dense:
        :param drpout1:
        :param drpout2:
        :return:
        """
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


    def createBaseModel(self, resnet=False, vgg=False):
        if resnet:
            backbone = ResNet50(weights='imagenet', include_top=False,
                                input_shape=(224, 224, 3))
        if vgg:
            backbone = VGG16(weights='imagenet', include_top=False,
                             input_shape=(224, 224, 3))
        for layer in backbone.layers:
            layer.trainable = False
        avg = keras.layers.GlobalAveragePooling2D()(backbone.output)
        baseModel = keras.Model(inputs=backbone.input, outputs=avg)

        return baseModel

    def createBackboneModel(self, denseNet=False, resnet=False, vgg=False):
        """
        For the purpose of this project, the VGG-16 model is selected and instantiated with weights trained on ImageNet.
        The top layer which is the classification layer is excluded and the input shape of the image is set to 224x224x3
        :param denseNet:
        :param resnet:
        :param vgg:
        :return:
        """
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

    def getCallBacks(self, variant='None'):
        # -------Callbacks-------------#
        #  checkpoints will be saved with the epoch number and the validation loss in the filename
        # best_model_weights = self.utils.getModelDirPath()+'weights.{epoch:02d}-{val_loss:.2f}.hdf5'

        best_model_weights = self.utils.getModelDirPath() + 'best_model_vgg_' + variant + '.hdf5'

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
        # Learning Rate Reducer
        learn_control = ReduceLROnPlateau(
            monitor='val_accuracy',
            patience=5,
            verbose=1, factor=0.3,
            min_lr=1e-7)  # Checkpoint
        return [checkpoint, earlystop, tensorboard, csvlogger, learn_control], best_model_weights
