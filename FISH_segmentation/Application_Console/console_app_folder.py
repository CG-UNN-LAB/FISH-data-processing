import argparse
import logging as log
import os
import fnmatch
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from chromosome_detection import ChromosomeCellDetector
from io_utils import read_czi_image
from pathlib import Path


def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', '--input',
        type=str,
        dest='input',
        help='Path to folder with image ',
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


def find_images_by_extension(root_folder, extension):
    matches = []
    for root, files in os.walk(root_folder):
        for basename in files:
            if fnmatch.filter(basename, extension):
                matches.append(os.path.join(root, basename))

    return matches


if __name__ == '__main__':
    log.basicConfig(format='[%(levelname)s]:%(message)s', level=log.INFO)
    args = cli_argument_parser()

    log.info(f'Found all images in input directory: {args.input}')

    images_paths_czi = []
    for file_path in Path(args.input).glob('**/*.czi'):
        images_paths_czi.append(file_path)
        log.info(f'Found image in czi format! Path: {file_path}')

    images_paths_png = []
    for file_path in Path(args.input).glob('**/*.png'):
        images_paths_png.append(file_path)
        log.info(f'Found image in png format! Path: {file_path}')

    images_paths_jpg = []
    for file_path in Path(args.input).glob('**/*.jpg'):
        images_paths_jpg.append(file_path)
        log.info(f'Found image in jpg format! Path: {file_path}')

    for image_path_czi in images_paths_czi:
        try:
            image, _ = read_czi_image(image_path_czi)

            folder_name_for_image_without_predict = "..\\Photo_Console_Find_All\\Photo_CZI\\Without_predict"
            if not os.path.exists(folder_name_for_image_without_predict):
                os.makedirs(folder_name_for_image_without_predict)
            Image_name = os.path.splitext(os.path.basename(image_path_czi))[0]
            plt.imsave(folder_name_for_image_without_predict + "\\" + Image_name + ".png", image)

            detector = ChromosomeCellDetector(image)
            log.info(f'{"Perform segmentation"}')
            detector.find_cells(confidence=args.confidence)

            log.info(f'{"Perform chromosomes detection"}')
            detector.detect_chromosomes()
            log.info(f'{detector.GreenChromosome} Green Chromosome(s), {detector.RedChromosome} Red Chromosome(s)')
        except Exception:
            log.exception(f'Open CV crashes on this image! Path to image:{image_path_czi}')

        fig, ax = plt.subplots(1, 1, figsize=(16, 16), dpi=300)
        ax = detector.plot(ax)
        fig.patch.set_visible(False)
        ax.axis('off')

        folder_name_for_image_with_predict = "..\\Photo_Console_Find_All\\Photo_CZI\\With_predict"
        if not os.path.exists(folder_name_for_image_with_predict):
            os.makedirs(folder_name_for_image_with_predict)
        fname = f'..\\Photo_Console_Find_All\\Photo_CZI\\With_predict\\{Image_name}.png'
        fig.savefig(fname, dpi=300, format='png')

    for image_path_png in images_paths_png:
        try:
            image = mpimg.imread(image_path_png)
            image = (255 * image).astype(np.uint8)  # normalize the data to 0-255
            image = np.ascontiguousarray(image)
            image = rgba2rgb(image)
            folder_name_for_image_without_predict = "..\\Photo_Console_Find_All\\Photo_PNG\\Without_predict"
            if not os.path.exists(folder_name_for_image_without_predict):
                os.makedirs(folder_name_for_image_without_predict)
            Image_name = os.path.splitext(os.path.basename(image_path_png))[0]
            plt.imsave(folder_name_for_image_without_predict + "\\" + Image_name + ".png", image)
            detector = ChromosomeCellDetector(image)
            log.info(f'{"Perform segmentation"}')
            detector.find_cells(confidence=args.confidence)

            log.info(f'{"Perform chromosomes detection"}')
            detector.detect_chromosomes()
            log.info(f'{detector.GreenChromosome} Green Chromosome(s), {detector.RedChromosome} Red Chromosome(s)')
        except Exception:
            log.exception(f'Open CV crashes on this image! Path to image:{image_path_png}')

        fig, ax = plt.subplots(1, 1, figsize=(16, 16), dpi=300)
        ax = detector.plot(ax)
        fig.patch.set_visible(False)
        ax.axis('off')

        folder_name_for_image_with_predict = "..\\Photo_Console_Find_All\\Photo_PNG\\With_predict"
        if not os.path.exists(folder_name_for_image_with_predict):
            os.makedirs(folder_name_for_image_with_predict)
        fname = f'..\\Photo_Console_Find_All\\Photo_PNG\\With_predict\\{Image_name}.png'
        fig.savefig(fname, dpi=300, format='png')

    for image_path_jpg in images_paths_jpg:
        try:
            image = mpimg.imread(image_path_jpg)
            image = np.ascontiguousarray(image)
            folder_name_for_image_without_predict = "..\\Photo_Console_Find_All\\Photo_JPG\\Without_predict"
            if not os.path.exists(folder_name_for_image_without_predict):
                os.makedirs(folder_name_for_image_without_predict)
            Image_name = os.path.splitext(os.path.basename(image_path_jpg))[0]
            plt.imsave(folder_name_for_image_without_predict + "\\" + Image_name + ".png", image)
            detector = ChromosomeCellDetector(image)
            log.info(f'{"Perform segmentation"}')
            detector.find_cells(confidence=args.confidence)
            log.info(f'{"Perform chromosomes detection"}')
            detector.detect_chromosomes()
            log.info(f'{detector.GreenChromosome} Green Chromosome(s), {detector.RedChromosome} Red Chromosome(s)')
        except Exception:
            log.exception(f'Open CV crashes on this image! Path to image:{image_path_jpg}')

        fig, ax = plt.subplots(1, 1, figsize=(16, 16), dpi=300)
        ax = detector.plot(ax)
        fig.patch.set_visible(False)
        ax.axis('off')

        folder_name_for_image_with_predict = "..\\Photo_Console_Find_All\\Photo_JPG\\With_predict"
        if not os.path.exists(folder_name_for_image_with_predict):
            os.makedirs(folder_name_for_image_with_predict)
        fname = f'..\\Photo_Console_Find_All\\Photo_JPG\\With_predict\\{Image_name}.png'
        fig.savefig(fname, dpi=300, format='png')
