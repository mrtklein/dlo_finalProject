from matplotlib import pyplot as plt
from utils import Utils
from matplotlib import pyplot
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
