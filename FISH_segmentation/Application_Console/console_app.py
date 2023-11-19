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


def rgba2rgb(rgba, background=(255, 255, 255)):
    row, col, ch = rgba.shape
    if ch == 3:
        return rgba
    assert ch == 4, 'RGBA image has 4 channels.'
    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]
    a = np.asarray(a, dtype='float32') / 255.0
    R, G, B = background
    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B
    return np.asarray(rgb, dtype='uint8')


if __name__ == '__main__':
    log.basicConfig(format='[%(levelname)s]:%(message)s', level=log.INFO)
    args = cli_argument_parser()
    log.info(f'Read input image {args.input}')

    if args.input.endswith(".czi"):
        image, _ = read_czi_image(args.input)
        folder_name_wp = "..\\Photo_Console\\Photo_czi\\Without_predict"  # Имя папки;
        if not os.path.exists(folder_name_wp):  # Создать папку, если ее нет;
            os.makedirs(folder_name_wp)
        Image_name = os.path.splitext(os.path.basename(args.input))[0]  # Для получения имени файла
        plt.imsave(folder_name_wp + "\\" + Image_name + ".png", image)

        folder_name_with_p = "..\\Photo_Console\\Photo_czi\\With_predict"
        if not os.path.exists(folder_name_with_p):
            os.makedirs(folder_name_with_p)

    if args.input.endswith(".png"):
        folder_name_wp = "..\\Photo_Console\\Photo_png\\Without_predict"  # Имя папки;
        if not os.path.exists(folder_name_wp):  # Создать папку, если ее нет;
            os.makedirs(folder_name_wp)
        image = mpimg.imread(args.input)
        image = (255 * image).astype(np.uint8)  # normalize the data to 0-255
        image = np.ascontiguousarray(image)
        image = rgba2rgb(image)
        Image_name = os.path.splitext(os.path.basename(args.input))[0]  # Для получения имени файла
        plt.imsave(folder_name_wp + "\\" + Image_name + ".png", image)

        folder_name_with_p = "..\\Photo_Console\\Photo_png\\With_predict"
        if not os.path.exists(folder_name_with_p):
            os.makedirs(folder_name_with_p)

    if args.input.endswith(".jpg"):
        folder_name_wp = "..\\Photo_Console\\Photo_jpg\\Without_predict"  # Имя папки;
        if not os.path.exists(folder_name_wp):  # Создать папку, если ее нет;
            os.makedirs(folder_name_wp)
        image = mpimg.imread(args.input)
        image = np.ascontiguousarray(image)
        image = rgba2rgb(image)
        Image_name = os.path.splitext(os.path.basename(args.input))[0]  # Для получения имени файла
        plt.imsave(folder_name_wp + "\\" + Image_name + ".png", image)

        folder_name_with_p = "..\\Photo_Console\\Photo_jpg\\With_predict"
        if not os.path.exists(folder_name_with_p):
            os.makedirs(folder_name_with_p)

detector = ChromosomeCellDetector(image)
log.info(f'{"Perform segmentation"}')
detector.find_cells(confidence=args.confidence)
log.info(f'{"Perform chromosomes detection"}')
detector.detect_chromosomes()
log.info(f'{detector.GreenChromosome} Green Chromosome(s), {detector.RedChromosome} Red Chromosome(s)')
fig, ax = plt.subplots(1, 1, figsize=(16, 16), dpi=300)
ax = detector.plot(ax)
# fig.patch.set_visible(False)
# ax.axis('off')

fname = f'{folder_name_with_p}\\{Image_name}.png'
fig.savefig(fname, dpi='figure', format='png', transparent=True)

output_csv_path = "output.csv"
log.info(f'{"Save information of prediction to .csv"}')
detector.write_to_csv(output_csv_path, folder_name_with_p, Image_name)
