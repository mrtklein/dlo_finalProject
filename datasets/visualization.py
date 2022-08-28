from matplotlib import pyplot as plt
from utils import Utils
import seaborn as sns
import numpy as np


class Visualizer:
    def __init__(self):
        self.utils = Utils()

    def visualize_raw_data(self, handsign: str, image_number: str):
        data_dir = self.utils.getInputPath()
        plt.imshow(plt.imread("%s" % data_dir + "Train\\" + "%s" % handsign + "\\img" + "%s" % image_number + ".jpg"))
        plt.title('Folder: %s; ' % handsign + 'Image Number: %s' % image_number)
        plt.show()

    def visualize_data_set(self, dataset, datalabel, img_number: int):
        for i in range(15):
            plt.figure(figsize=(15, 9))
            plt.subplot(5, 5, i + 1)
            plt.imshow(dataset[img_number])
            plt.title('Label:  {labels[datalabel[img_number]]}')
            img_number += 1

    def plot_batch(self, ims, figsize=(12, 6), rows=4, titles=None, filename='batchFig.png'):
        """

        :param ims: batch of data
        :param figsize:
        :param rows:
        :param titles: One-Hot Encoded Titles of the images
        :param filename: Name of the file which will be saved
        """
        if type(ims[0]) is np.ndarray:
            ims = np.array(ims)
            if (ims.shape[-1] != 3):
                ims = ims.transpose((0, 2, 3, 1))
        f = plt.figure(figsize=figsize)
        cols = len(ims) // rows if len(ims) % 2 == 0 else len(ims) // rows + 1
        for i in range(rows * cols):
            sp = f.add_subplot(rows, cols, i + 1)
            sp.axis('Off')
            if titles is not None:
                curr_title = self.getImageTitel(titles[i])
                sp.set_title(curr_title, fontsize=10)
            plt.imshow(ims[i])
        plt.savefig(filename)
        plt.show()

    def getImageTitel(self, one_hot_encode_title):
        if all(one_hot_encode_title == [0, 0, 0, 1]):
            return "scissors"
        if all(one_hot_encode_title == [0, 0, 1, 0]):
            return "rock"
        if all(one_hot_encode_title == [0, 1, 0, 0]):
            return "rest"
        if all(one_hot_encode_title == [1, 0, 0, 0]):
            return "paper"

    def drawHistory(self, acc, loss, val_acc, val_loss):
        epochs_range = range(len(acc))
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, color='teal', label='Training Accuracy')
        plt.plot(epochs_range, val_acc, color='orange', label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.xlabel("Epochs",
                   family='serif',
                   color='black',
                   weight='normal',
                   size=16,
                   labelpad=6)
        plt.ylabel("Accuracy in %",
                   family='serif',
                   color='black',
                   weight='normal',
                   size=16,
                   labelpad=6)
        plt.title('Training and Validation Accuracy')
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, color='teal', label='Training Loss')
        plt.plot(epochs_range, val_loss, color='orange', label='Validation Loss')
        plt.legend(loc='upper right')
        plt.xlabel("Epochs",
                   family='serif',
                   color='black',
                   weight='normal',
                   size=16,
                   labelpad=6)
        plt.ylabel("Loss in %",
                   family='serif',
                   color='black',
                   weight='normal',
                   size=16,
                   labelpad=6)
        plt.title('Training and Validation Loss')
        plt.savefig(
            "Result__" + "Val_acc" + str(round(max(val_acc), 2)) + "_Val_loss" + str(round(max(val_loss), 2)) + ".png")
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        plt.show()

    def showHeatmap(self, dataframe):
        sns.heatmap(dataframe, annot=True, fmt="g", cmap="Blues")
        plt.title("Konfusionsmatrix"), plt.tight_layout()
        plt.ylabel("Echte Klasse"), plt.xlabel("Vorhergesagte Klasse")
        plt.savefig("Confusion_matrix_resnet.png")
        plt.show()
