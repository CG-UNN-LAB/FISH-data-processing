import argparse
import logging as log
import os

from matplotlib import pyplot as plt

from chromosome_detection import ChromosomeCellDetector
from io_utils import read_czi_image
import matplotlib.image as mpimg
import numpy as np


def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', '--input',
        type=str,
        dest='input',
        help='Path to input image (.czi .png .jpg)',
        required=True)
    parser.add_argument(
        '-c', '--confidence',
        dest='confidence',
        type=float,
        help='Threshold for object prediction',
        default=0.5)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    log.basicConfig(format='[%(levelname)s]:%(message)s', level=log.INFO)
    args = cli_argument_parser()

    log.info(f'Read input image {args.input}')
    if args.input.endswith(".czi"):
        image, _ = read_czi_image(args.input)

        folder_name = "..\\Photo_Console\\Photo"  # Имя папки;
        if not os.path.exists(folder_name):  # Создать папку, если ее нет;
            os.makedirs(folder_name)
        Image_name = os.path.splitext(os.path.basename(args.input))[0]  # Для получения имени файла
        plt.imsave(folder_name + "\\" + Image_name + ".png", image)
    else:
        image = mpimg.imread(args.input)
        image = (255 * image).astype(np.uint8)  # normalize the data to 0-255
        image = np.ascontiguousarray(image)

    detector = ChromosomeCellDetector(image)
    log.info(f'{"Perform segmentation"}')
    detector.find_cells(confidence=args.confidence)
    log.info(f'{"Perform chromosomes detection"}')
    detector.detect_chromosomes()

    fig, ax = plt.subplots(1, 1, figsize=(16, 16), dpi=300)
    ax = detector.plot(ax)
    fig.patch.set_visible(False)
    ax.axis('off')

    fname = f'..\\Photo_Console\\{os.path.basename(args.input).split(".")[0]}.png'
    fig.savefig(fname, dpi='figure', bbox_inches='tight', format='png')
