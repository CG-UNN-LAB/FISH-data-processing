import argparse
import logging as log
import os

from io_utils import read_czi_file_as_pil_image
from ultralytics import YOLO

def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input',
                        type=str,
                        dest='input',
                        help='Path to input image (.czi .png .jpg)',
                        required=True)
    parser.add_argument('-t', '--threshold',
                        dest='threshold',
                        type=float,
                        help='Threshold for object prediction',
                        default=0.5)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    log.basicConfig(format='[%(levelname)s]:%(message)s', level=log.INFO)
    args = cli_argument_parser()

    log.info(f'Load Segmentation model')
    FileModelPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..\\Model\\my_yolov8_model_core_segmentation.pt")
    model = YOLO(FileModelPath)

    log.info(f'Read input image {args.input}')
    image = read_czi_file_as_pil_image(args.input)
    
    log.info(f'Perform segmentation')
    predictions = model.predict(
        image, 
        show = False, 
        classes=[0,1], 
        save = True,
        project='..\\Photo_Console',
        name=f'{os.path.basename(args.input)}',
        show_labels = False,
        show_conf = False,
        save_txt = True,
        stream=False,
        conf=args.threshold,
        line_thickness=1)
