import argparse
import os
import sys
import tensorflow as tf

from trainer import Trainer
from utils import Utils


def main():
    is_windows = sys.platform.startswith('win')
    if is_windows:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--img_height', type=int, default=224)
    parser.add_argument('--img_width', type=int, default=224)
    config = parser.parse_args()

    trainer = Trainer(config)

    trainer.train(variant="more_dropout")
    # trainer.loadLogFile('training_csv.log')
    # trainer.showConfusionMatrix(Utils().getModelDirPath() + 'best_model_vgg_brightness.hdf5', "vgg_Confusion_matrix_brightness.png")

    # trainer.showConfusionMatrix(Utils().getModelDirPath() + 'best_model_vgg_brightness.hdf5', "vgg_Confusion_matrix_brightness.png")
    # trainer.showConfusionMatrix(Utils().getModelDirPath() + 'best_model_cnn.hdf5', "cnn_Confusion_matrix.png")
    #
    # trainer.evaluate_saveWrongPredictions(Utils().getModelDirPath() + 'best_model_cnn.hdf5', wrong_rock=False,
    #                                       modelname='cnn')
    # trainer.evaluate_saveWrongPredictions(Utils().getModelDirPath() + 'best_model_vgg.hdf5', wrong_rock=True,
    #                                       modelname='vgg')
    # trainer.evaluate_saveWrongPredictions(Utils().getModelDirPath() + 'best_model_resnet.hdf5', wrong_rock=True,
    #                                       modelname='resnet')


if __name__ == "__main__":
    main()
