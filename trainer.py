import matplotlib.pyplot as plt
from datasets.data_loader import DataLoader as Dataset
from datasets.visualization import Visualizer
from models.conv_net import ConvNet
from utils import Utils


class Trainer:
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.img_height = config.img_height
        self.img_width = config.img_width
        self.data = Dataset()
        self.cnnModel = ConvNet()
        self.utils = Utils()
        self.visualizer=Visualizer()

    def train(self, plot=True):
        train_dataset, valid_dataset = self.data.get_images(self.batch_size, self.img_height, self.img_width)

        self.visualizer.visualize_raw_data("rock", "0050")

        model = self.cnnModel.get_model(self.img_height, self.img_width)

        callbacks, best_model_weights = self.cnnModel.getCallBacks()

        history = model.fit(
            train_dataset,
            epochs=self.epochs,
            validation_data=valid_dataset,
            callbacks=callbacks,
            verbose=1
        )

        if plot:
            self.plot_default(history)

        self.saveModel(best_model_weights, model, valid_dataset)

    def plot_default(self, history):
        print("History keys: " + str(history.history.keys()))

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(self.epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, color='teal', label='Training Accuracy')
        plt.plot(epochs_range, val_acc, color='orange', label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, color='teal', label='Training Loss')
        plt.plot(epochs_range, val_loss, color='orange', label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

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
