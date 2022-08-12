import argparse
import os
import sys

from trainer import Trainer


def main():
    is_windows = sys.platform.startswith('win')
    if is_windows:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--img_height', type=int, default=200)
    parser.add_argument('--img_width', type=int, default=300)
    config = parser.parse_args()

    trainer = Trainer(config)

    trainer.train()



if __name__ == "__main__":
    main()
