import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from datasets.data_loader import DataLoader as Dataset
from datasets.visualization import Visualizer
from models.conv_net import ConvNet
from models.pretrained_model import Pretrained_Model
from utils import Utils
import pandas as pd


class Trainer:
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.img_height = config.img_height
        self.img_width = config.img_width
        self.data = Dataset()
        self.cnnModel = ConvNet()
        self.model_pretrained = Pretrained_Model()
        self.utils = Utils()
        self.visualizer = Visualizer()

    def train(self, plot=True):
        train_dataset, valid_dataset = self.data.get_images(self.batch_size, self.img_height, self.img_width)

        imgs, labels = next(train_dataset)
        self.visualizer.plot_batch(imgs, titles=labels,
                                   filename="Batch_Augmentation" + str(self.img_height) + "x" + str(
                                       self.img_height) + ".png")

        # model = self.cnnModel.get_model(self.img_height, self.img_width)

        backbone = self.model_pretrained.createBackboneModel(vgg=True)
        backbone.summary()

        model = self.model_pretrained.createModel(backbone, lr=1e-4, drpout1=0.3, drpout2=0.2)
        model.summary()

        callbacks, best_model_weights = self.model_pretrained.getCallBacks()

        history = model.fit(
            train_dataset,
            epochs=self.epochs,
            validation_data=valid_dataset,
            callbacks=callbacks,
            verbose=1
        )

        if plot:
            self.plot_history(history)

        self.saveModel(best_model_weights, model, valid_dataset)

    def fitandPlot(self, model, dataAug, train_data, valid_dataset, EPOCHS=20, INIT_LR=1e-1, BS=64):
        # Learning Rate Reducer
        learn_control = ReduceLROnPlateau(
            monitor='val_accuracy',
            patience=5,
            verbose=1, factor=0.2,
            min_lr=1e-7)  # Checkpoint
        filepath = "weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath,
                                     monitor='val_accuracy',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='max')

        history = model.fit(
            train_data,
            epochs=EPOCHS,
            validation_data=valid_dataset,
            callbacks=[learn_control, checkpoint])

        self.plot_history(history)

    def plot_history(self, history):
        print("History keys: " + str(history.history.keys()))

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        self.visualizer.drawHistory(acc, loss, val_acc, val_loss)

    def saveModel(self, best_model_weights, model, valid_dataset):
        model.load_weights(best_model_weights)

        model_score = model.evaluate_generator(valid_dataset)
        print("Model Test Loss:", model_score[0])
        print("Model Test Accuracy:", model_score[1])

        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)

        #  checkpoints will be saved with the epoch number and the validation loss in the filename
        model.save(self.utils.getModelDirPath() + 'model-saved.hdf5')
        print("Weights Saved")

    def loadLogFile(self, path):
        df = pd.read_csv(path, index_col='epoch')
        print(df)
        self.visualizer.drawHistory(df.accuracy, df.loss, df.val_accuracy, df.val_loss)
