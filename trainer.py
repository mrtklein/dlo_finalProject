import matplotlib.pyplot as plt
from models.pretrained_ResNet import InceptionResNetV2 as Mdl
from datasets.data_loader import DataLoader as Dataset
from models.pretrained_MobileNet import Pretrained_MobileNet
from utils import Utils


class Trainer:
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.img_height = config.img_height
        self.img_width = config.img_width
        self.data = Dataset()
        self.mobileNet = Pretrained_MobileNet()
        self.utils=Utils()
        # self.mdl = Mdl()

    def train(self, plot=True):
        train_dataset, valid_dataset = self.data.get_images(self.batch_size,self.img_height,self.img_width)

        self.show_samples()

        labels=list(train_dataset.class_indices.keys())

        model = self.mobileNet.get_model(self.img_height, self.img_width)
        callbacks, best_model_weights = self.mobileNet.getCallBacks()

        steps_per_epoch = 5
        history = model.fit(
            train_dataset,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=valid_dataset,
            validation_steps=steps_per_epoch,
            callbacks=callbacks
        )

        if plot:
            self.plot_default(history)

        self.saveModel(best_model_weights, model, steps_per_epoch, valid_dataset)

    def showImages(self, train_dataset):
        for img in train_dataset.next():
            print(img.shape) #(32, 224, 224, 3)
            plt.imshow(img[0])
            plt.show()

    def show_samples(self,array_of_images):
        n = array_of_images.shape[0]
        total_rows = 1 + int((n - 1) / 5)
        total_columns = 5
        fig = plt.figure()
        gridspec_array = fig.add_gridspec(total_rows, total_columns)

        for i, img in enumerate(array_of_images):
            row = int(i / 5)
            col = i % 5
            ax = fig.add_subplot(gridspec_array[row, col])
            ax.imshow(img)

        plt.show()

    def saveModel(self, best_model_weights, model, steps_per_epoch, valid_dataset):
        model.load_weights(best_model_weights)
        model_score = model.evaluate_generator(valid_dataset, steps=steps_per_epoch)
        print("Model Test Loss:", model_score[0])
        print("Model Test Accuracy:", model_score[1])
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # model.save("model.h5")
        model.save("model_extend.h1")
        print("Weights Saved")

    def plot_default(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(self.epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()
