import numpy as np
from keras.models import load_model
from tensorflow.python.ops.confusion_matrix import confusion_matrix

from datasets.data_loader import DataLoader as Dataset
from datasets.visualization import Visualizer
from models.conv_net import ConvNet
from models.pretrained_model import Pretrained_Model
from utils import Utils
import pandas as pd
from keras.utils.vis_utils import plot_model


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

        backbone = self.model_pretrained.createBackboneModel(resnet=True)
        backbone.summary()
        plot_model(backbone, to_file='DLO_Graphs/resnet-backbone.png', show_shapes=True, show_layer_names=True)

        model = self.model_pretrained.createModel(backbone, lr=1e-4, drpout1=0.3, drpout2=0.2)
        model.summary()
        plot_model(model, to_file='DLO_Graphs/resnet-model.png', show_shapes=True, show_layer_names=True)

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

    def showConfusionMatrix(self, model_path):
        train_dataset, valid_dataset = self.data.get_images(self.batch_size, self.img_height, self.img_width)

        model = load_model(model_path)

        target_predicted = model.predict(valid_dataset)
        target_predicted = np.argmax(target_predicted, axis=1)
        # Konfusionsmatrix erzeugen
        matrix = confusion_matrix(valid_dataset.labels, target_predicted)
        dataframe = pd.DataFrame(matrix, index=valid_dataset.class_indices.keys(),
                                 columns=valid_dataset.class_indices.keys())
        self.visualizer.showHeatmap(dataframe)

    def saveWrongPredictions(self, model_path, wrong_rock=True):
        train_dataset, valid_dataset = self.data.get_images(self.batch_size, self.img_height, self.img_width)

        model = load_model(model_path)

        wrong_predicted = self.getWrongPredictions(model, valid_dataset)

        print(f"There are {str(len(wrong_predicted))} wrong predictions.")

        if wrong_rock:
            wrong_rock = list(filter(lambda wrong_predicted: wrong_predicted['prediction'] == 'rock', wrong_predicted))
            count = 0
            for img in wrong_rock:
                img_file = img['image']
                y_pred = img['prediction']
                probability = img['probability']
                y_target = img['actual']
                count = count + 1
                self.visualizer.show_save_ImgPredictVsActual(img_file, y_pred, probability, y_target,
                                                             self.utils.getWrong_predictedDirPath() + "wrong_rock" + str(
                                                                 count))

    def getWrongPredictions(self, model, valid_dataset):
        wrong_predicted = []
        batch_idx = 0
        while batch_idx < len(valid_dataset):
            try:
                images, labels = next(valid_dataset)
                target_predicted = model.predict(images, batch_size=1)
                predicted_class_indices = np.argmax(target_predicted, axis=1)

                for i in range(len(predicted_class_indices)):
                    y_pred = self.visualizer.getImageTitelCatNumber(predicted_class_indices[i])
                    pred_probability = '{0:.3f}'.format(max(target_predicted[i]))
                    y_target = self.visualizer.getImageTitel(labels[i])

                    if y_pred != y_target:
                        wrong_predicted.append(
                            {
                                "image": images[i],
                                "prediction": y_pred,
                                "probability": pred_probability,
                                "actual": y_target
                            })
                        print(f"Image {i} in batch {batch_idx} of {len(valid_dataset)}")
                batch_idx = batch_idx + 1
            except StopIteration:
                print("StopIteration")
                break
        return wrong_predicted
