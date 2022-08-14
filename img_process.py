import matplotlib.pyplot as plt
from utils import Utils

class Image_process:
    def __init__(self):
        self.utils = Utils()
        
    def visualize_raw_data(self, handsign: str, image_number: str):
        data_dir = self.utils.getInputPath()
        plt.imshow(plt.imread("%s" % data_dir + "Train\\" + "%s" % handsign + "\\img"+"%s" %image_number + ".jpg"))
        plt.title('Folder: %s; ' % handsign + 'Image Number: %s' % image_number)
        plt.show()
    
    def visualize_data_set(self, dataset, datalabel, img_number: int):
        for i in range (15):
            plt.figure(figsize = 15, 9)
            plt.subplot(5,5,i+1)
            plt.imshow(dataset[img_number])
            plt.title('Label:  {labels[datalabel[img_number]]}')
            img_number += 1
            


