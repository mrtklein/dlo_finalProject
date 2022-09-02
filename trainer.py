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

    def train(self, plot=True, variant=""):
        train_dataset, valid_dataset = self.data.get_images(self.batch_size, self.img_height, self.img_width)

        imgs, labels = next(train_dataset)
        self.visualizer.plot_batch(imgs, titles=labels,
                                   filename="Batch.png")

        # base_model = self.model_pretrained.createBaseModel(vgg=True)
        # base_model.summary()
        # plot_model(base_model, to_file='DLO_Graphs/vgg-model-tf.png', show_shapes=True, show_layer_names=True)

        backbone = self.model_pretrained.createBackboneModel(vgg=True)
        backbone.summary()
        plot_model(backbone, to_file='DLO_Graphs/vgg-backbone.png', show_shapes=True, show_layer_names=True)

        model = self.model_pretrained.getModel(backbone, dense1=256, dense2=128, lr=1e-4, drpout1=0.6, drpout2=0.2)
        model.summary()
        plot_model(model, to_file='DLO_Graphs/model_vgg_dropout.png', show_shapes=True, show_layer_names=True)

        callbacks, best_model_weights = self.model_pretrained.getCallBacks(variant=variant)

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

        learning_rate = history.history['lr']
        self.visualizer.drawHistory(acc, loss, val_acc, val_loss, learning_rate)

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

    def showConfusionMatrix(self, model_path, filename):
        train_dataset, valid_dataset = self.data.get_images(self.batch_size, self.img_height, self.img_width)

        model = load_model(model_path)

        target_predicted = model.predict(valid_dataset)
        target_predicted = np.argmax(target_predicted, axis=1)
        # Konfusionsmatrix erzeugen
        matrix = confusion_matrix(valid_dataset.labels, target_predicted)
        dataframe = pd.DataFrame(matrix, index=valid_dataset.class_indices.keys(),
                                 columns=valid_dataset.class_indices.keys())
        self.visualizer.showHeatmap(dataframe, filename=filename)

    def evaluate_saveWrongPredictions(self, model_path, wrong_rock=True, modelname='unknown'):
        train_dataset, valid_dataset = self.data.get_images(self.batch_size, self.img_height, self.img_width)

        model = load_model(model_path)

        # Evaluate the model on the test data using `evaluate`
        print("Evaluate on test data")
        results = model.evaluate(valid_dataset, batch_size=32)

        wrong_predicted = self.getWrongPredictions(model, valid_dataset)

        with open(modelname + '_evaluation.txt', 'w') as f:
            print(f"MODEL: {modelname} "
                  f"\n"
                  f"test loss, test acc: {results}."
                  f"\n"
                  f"There are " + str(len(wrong_predicted)) + "wrong predictions.", file=f)

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
                                                             self.utils.getWrong_predictedDirPath()
                                                             + "MODEL_" + modelname + "__wrong_rock" + str(count))

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
