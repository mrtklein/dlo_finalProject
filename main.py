import argparse
import os
import sys
import tensorflow as tf

from trainer import Trainer


def main():
    is_windows = sys.platform.startswith('win')
    if is_windows:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--img_height', type=int, default=224)
    parser.add_argument('--img_width', type=int, default=224)
    config = parser.parse_args()

    trainer = Trainer(config)

    trainer.train()
    # trainer.loadLogFile('training_csv.log')


if __name__ == "__main__":
    main()
