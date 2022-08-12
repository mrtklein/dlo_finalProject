import matplotlib.pyplot as plt
from utils import Utils

class Image_process:
    def __init__(self):
        self.utils = Utils()
        
    def visualize(self, handsign: str, image_number: str):
        data_dir = self.utils.getInputPath()
        plt.imshow(plt.imread("%s" % data_dir + "Train\\" + "%s" % handsign + "\\img"+"%s" %image_number + ".jpg"))
        plt.title('Folder: %s; ' % handsign + 'Image Number: %s' % image_number)
        plt.show()
