import matplotlib.pyplot as plt
from utils import Utils
from matplotlib import pyplot


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

    def plotImage(self, batch):
        fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
        for idx, img in enumerate(batch[0][:4]):
            ax[idx].imshow(img.astype(int))
            ax[idx].title.set_text(batch[1][idx])
        plt.show()

    def plotImagePyPlot(self, it):
        # generate samples and plot
        for i in range(9):
            # define subplot
            pyplot.subplot(330 + 1 + i)
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            # plot raw pixel data
            pyplot.imshow(image)
        # show the figure
        pyplot.show()
