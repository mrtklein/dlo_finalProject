import matplotlib.pyplot as plt


class Visualizer():

    def __init__(self) -> None:
        super().__init__()

    def showImages(self, train_dataset):
        for img in train_dataset.next():
            print(img.shape)  # (32, 224, 224, 3)
            plt.imshow(img[0])
            plt.show()

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
